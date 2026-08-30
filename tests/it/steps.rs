//! Step recording tests: the `(pc, op)` base track, the detail overlay, and switching full
//! recording on and off while a transaction executes.

use alloy_primitives::{hex, Address, Bytes, U256};
use alloy_rpc_types_trace::geth::GethDefaultTracingOptions;
use revm::{
    bytecode::opcode,
    context::TxEnv,
    context_interface::{ContextTr, TransactTo},
    database::CacheDB,
    database_interface::EmptyDB,
    inspector::JournalExt,
    interpreter::{
        interpreter::EthInterpreter, CallInputs, CallOutcome, CreateInputs, CreateOutcome,
        Interpreter,
    },
    primitives::hardfork::SpecId,
    state::{AccountInfo, Bytecode},
    Context, InspectEvm, Inspector, MainBuilder, MainContext,
};
use revm_inspectors::tracing::{
    types::TraceMemberOrder, StepRecording, TracingInspector, TracingInspectorConfig,
};

/// A loop that counts down from 10 000, executing seven opcodes per iteration:
///
/// ```text
/// PUSH2 0x2710          // counter
/// JUMPDEST              // offset 3
/// PUSH1 1
/// SWAP1
/// SUB
/// DUP1
/// PUSH1 3
/// JUMPI
/// STOP
/// ```
const LOOP_CODE: [u8; 13] = hex!("6127105b600190038060035700");
const LOOP_ITERATIONS: usize = 10_000;
/// `PUSH2`, then seven opcodes per iteration, then `STOP`.
const LOOP_STEPS: usize = 1 + 7 * LOOP_ITERATIONS + 1;
const CONTRACT: Address = Address::repeat_byte(0x42);
const CALLER: Address = Address::repeat_byte(0x01);

/// Runs `inspector` over a call to `CONTRACT` holding `code`, plus the given extra accounts.
fn run_code<INSP>(inspector: INSP, code: Bytes, extra: &[(Address, Bytes)]) -> INSP
where
    INSP: for<'a> Inspector<
        Context<
            revm::context::BlockEnv,
            TxEnv,
            revm::context::CfgEnv,
            CacheDB<EmptyDB>,
            revm::Journal<CacheDB<EmptyDB>>,
            (),
        >,
    >,
{
    let mut db = CacheDB::new(EmptyDB::default());
    for (address, code) in core::iter::once(&(CONTRACT, code)).chain(extra) {
        db.insert_account_info(
            *address,
            AccountInfo { code: Some(Bytecode::new_raw(code.clone())), ..Default::default() },
        );
    }
    db.insert_account_info(
        CALLER,
        AccountInfo { balance: U256::from(u64::MAX), ..Default::default() },
    );
    let context = Context::mainnet().with_db(db).modify_cfg_chained(|c| c.spec = SpecId::CANCUN);
    let mut evm = context.build_mainnet_with_inspector(inspector);
    let res = evm
        .inspect_tx(TxEnv {
            caller: CALLER,
            gas_limit: 10_000_000,
            kind: TransactTo::Call(CONTRACT),
            ..Default::default()
        })
        .unwrap();
    assert!(res.result.is_success(), "{res:?}");
    evm.into_inspector()
}

fn run<INSP>(inspector: INSP) -> INSP
where
    INSP: for<'a> Inspector<
        Context<
            revm::context::BlockEnv,
            TxEnv,
            revm::context::CfgEnv,
            CacheDB<EmptyDB>,
            revm::Journal<CacheDB<EmptyDB>>,
            (),
        >,
    >,
{
    run_code(inspector, Bytes::from_static(&LOOP_CODE), &[])
}

fn step_ordering_entries(tracer: &TracingInspector) -> usize {
    tracer.traces().nodes()[0]
        .ordering
        .iter()
        .filter(|item| matches!(item, TraceMemberOrder::Step(_)))
        .count()
}

#[test]
fn pc_and_op_records_every_step_and_no_details() {
    let tracer = run(TracingInspector::new(
        TracingInspectorConfig::none().set_step_recording(StepRecording::PcAndOp),
    ));
    let trace = &tracer.traces().nodes()[0].trace;

    assert_eq!(trace.step_count(), LOOP_STEPS);
    assert_eq!(trace.steps.detailed_len(), 0);
    assert_eq!(step_ordering_entries(&tracer), 0);
    // The first step is the `PUSH2` at pc 0, the last the `STOP` at pc 12.
    let first = trace.iter_steps().next().unwrap();
    assert_eq!((first.pc, first.op.get()), (0, opcode::PUSH2));
    let last = trace.iter_steps().next_back().unwrap();
    assert_eq!((last.pc, last.op.get()), (12, opcode::STOP));
    assert!(trace.iter_steps().all(|step| step.detail().is_none()));
    // Five bytes per step, plus whatever capacity the vectors grew to.
    assert!(trace.steps.bytes() >= 5 * LOOP_STEPS);
    assert!(trace.steps.bytes() <= 2 * 5 * LOOP_STEPS);
}

#[test]
fn full_records_a_detail_for_every_step() {
    // Through the legacy bool setter, which maps onto `Full`.
    let tracer = run(TracingInspector::new(TracingInspectorConfig::none().set_steps(true)));
    let trace = &tracer.traces().nodes()[0].trace;

    assert_eq!(trace.step_count(), LOOP_STEPS);
    assert_eq!(trace.steps.detailed_len(), LOOP_STEPS);
    assert_eq!(step_ordering_entries(&tracer), LOOP_STEPS);
    for (idx, step) in trace.iter_detailed_steps().enumerate() {
        assert_eq!(step.step_index, idx);
        assert_eq!(trace.detailed_step(idx).unwrap().pc, step.pc);
        // `step_end` filled the detail: every opcode of the loop but `STOP` costs gas.
        assert!(step.gas_cost > 0 || step.op.get() == opcode::STOP);
    }
    assert!(trace.last_detailed_step().unwrap().status.is_some(), "`STOP` records its status");
}

/// Switches the tracer between `PcAndOp` and `Full` from inside the step hook, the way a
/// cheatcode does from inside a call.
struct Switching {
    tracer: TracingInspector,
    seen: usize,
    window: core::ops::Range<usize>,
}

impl<CTX: ContextTr<Journal: JournalExt>> Inspector<CTX, EthInterpreter> for Switching {
    fn initialize_interp(&mut self, interp: &mut Interpreter, context: &mut CTX) {
        self.tracer.initialize_interp(interp, context);
    }

    fn step(&mut self, interp: &mut Interpreter, context: &mut CTX) {
        let level = if self.window.contains(&self.seen) {
            StepRecording::Full
        } else {
            StepRecording::PcAndOp
        };
        self.tracer.update_config(|config| config.set_step_recording(level));
        self.seen += 1;
        self.tracer.step(interp, context);
    }

    fn step_end(&mut self, interp: &mut Interpreter, context: &mut CTX) {
        self.tracer.step_end(interp, context);
    }

    fn call(&mut self, context: &mut CTX, inputs: &mut CallInputs) -> Option<CallOutcome> {
        self.tracer.call(context, inputs)
    }

    fn call_end(&mut self, context: &mut CTX, inputs: &CallInputs, outcome: &mut CallOutcome) {
        self.tracer.call_end(context, inputs, outcome);
    }

    fn create(&mut self, context: &mut CTX, inputs: &mut CreateInputs) -> Option<CreateOutcome> {
        self.tracer.create(context, inputs)
    }

    fn create_end(
        &mut self,
        context: &mut CTX,
        inputs: &CreateInputs,
        outcome: &mut CreateOutcome,
    ) {
        self.tracer.create_end(context, inputs, outcome);
    }
}

#[test]
fn full_recording_switched_on_mid_transaction_overlays_a_window() {
    let window = 100..250;
    let switching = run(Switching {
        tracer: TracingInspector::new(
            TracingInspectorConfig::none().set_step_recording(StepRecording::PcAndOp),
        ),
        seen: 0,
        window: window.clone(),
    });
    let trace = &switching.tracer.traces().nodes()[0].trace;

    // The base track is complete across the switch.
    assert_eq!(trace.step_count(), LOOP_STEPS);
    // The detail overlay covers exactly the window, in order.
    assert_eq!(trace.steps.detailed_len(), window.len());
    let indices: Vec<usize> = trace.iter_detailed_steps().map(|step| step.step_index).collect();
    assert_eq!(indices, window.clone().collect::<Vec<_>>());
    // Only detailed steps have ordering entries, and they index the overlay.
    assert_eq!(step_ordering_entries(&switching.tracer), window.len());
    for (k, step) in trace.iter_detailed_steps().enumerate() {
        assert_eq!(trace.detailed_step(k).unwrap().pc, step.pc);
        assert_eq!(trace.iter_steps().nth(step.step_index).unwrap().pc, step.pc);
        assert!(step.gas_cost > 0, "`step_end` filled the detail");
    }
    // Steps outside the window carry no detail.
    assert!(trace
        .iter_steps()
        .enumerate()
        .all(|(idx, step)| { step.detail().is_some() == window.contains(&idx) }));
}

const CALLEE_B: Address = Address::repeat_byte(0x43);
const CALLEE_C: Address = Address::repeat_byte(0x44);

/// `CALL(gas = 0x2710, to, 0, 0, 0, 0, 0)` then `POP`: nine steps.
fn call_and_pop(to: Address) -> Vec<u8> {
    let mut code = vec![
        opcode::PUSH1,
        0,
        opcode::PUSH1,
        0,
        opcode::PUSH1,
        0,
        opcode::PUSH1,
        0,
        opcode::PUSH1,
        0,
        opcode::PUSH20,
    ];
    code.extend_from_slice(to.as_slice());
    code.extend_from_slice(&[opcode::PUSH2, 0x27, 0x10, opcode::CALL, opcode::POP]);
    code
}

/// Full recording switched on inside the first callee: a call executed while only the base track
/// was recorded must still push its child frame, and the second call must be attributed to the
/// second child, not the first.
#[test]
fn child_frames_are_attributed_across_a_partial_window() {
    // A: call B, call C, stop — 19 steps.
    let mut caller = call_and_pop(CALLEE_B);
    caller.extend(call_and_pop(CALLEE_C));
    caller.push(opcode::STOP);
    let callee_b = vec![opcode::JUMPDEST, opcode::STOP]; // 2 steps
    let callee_c = vec![opcode::PC, opcode::POP, opcode::STOP]; // 3 steps

    // Steps are counted across frames: A's first eight steps (0..8, ending with `CALL`) run
    // before B's (8, 9); the window opens on B's first step.
    let window = 8..usize::MAX;
    let switching = run_code(
        Switching {
            tracer: TracingInspector::new(
                TracingInspectorConfig::none().set_step_recording(StepRecording::PcAndOp),
            ),
            seen: 0,
            window,
        },
        caller.into(),
        &[(CALLEE_B, callee_b.into()), (CALLEE_C, callee_c.into())],
    );
    let nodes = switching.tracer.traces().nodes();
    assert_eq!(nodes.len(), 3);
    assert_eq!(nodes[0].trace.step_count(), 19);
    // A's steps after the first `CALL`: `POP`, the second call's eight opcodes, `POP`, `STOP`.
    assert_eq!(nodes[0].trace.detailed_step_count(), 11);
    assert_eq!(nodes[1].trace.detailed_step_count(), 2);
    assert_eq!(nodes[2].trace.detailed_step_count(), 3);
    assert_eq!(nodes[1].trace.step_count(), 2);
    assert_eq!(nodes[2].trace.step_count(), 3);

    let frame = switching.tracer.geth_builder().geth_traces(
        0,
        Bytes::new(),
        GethDefaultTracingOptions::default(),
    );
    let logs: Vec<(u64, u64)> = frame.struct_logs.iter().map(|log| (log.depth, log.pc)).collect();
    // B's two steps (depth 2), A's POP and second call (depth 1), C's three steps (depth 2),
    // then A's POP and STOP.
    let mut expected = vec![(2, 0), (2, 1)];
    // A's `POP` at pc 35, then the second call's opcodes up to its `CALL` at pc 70.
    expected.extend([35, 36, 38, 40, 42, 44, 46, 67, 70].into_iter().map(|pc| (1, pc)));
    expected.extend([(2, 0), (2, 1), (2, 2), (1, 71), (1, 72)]);
    assert_eq!(logs, expected);
}

/// Raises the recording level to `Full` from inside `step_end`, i.e. after `step` already ran
/// for the same instruction under `PcAndOp`.
struct LateSwitch {
    tracer: TracingInspector,
    seen: usize,
    window: core::ops::Range<usize>,
    raise_at: usize,
}

impl<CTX: ContextTr<Journal: JournalExt>> Inspector<CTX, EthInterpreter> for LateSwitch {
    fn step(&mut self, interp: &mut Interpreter, context: &mut CTX) {
        if self.seen < self.raise_at {
            let level = if self.window.contains(&self.seen) {
                StepRecording::Full
            } else {
                StepRecording::PcAndOp
            };
            self.tracer.update_config(|config| config.set_step_recording(level));
        }
        self.seen += 1;
        self.tracer.step(interp, context);
    }

    fn step_end(&mut self, interp: &mut Interpreter, context: &mut CTX) {
        if self.seen == self.raise_at + 1 {
            self.tracer.update_config(|config| config.set_step_recording(StepRecording::Full));
        }
        self.tracer.step_end(interp, context);
    }

    fn call(&mut self, context: &mut CTX, inputs: &mut CallInputs) -> Option<CallOutcome> {
        self.tracer.call(context, inputs)
    }

    fn call_end(&mut self, context: &mut CTX, inputs: &CallInputs, outcome: &mut CallOutcome) {
        self.tracer.call_end(context, inputs, outcome);
    }
}

#[test]
fn full_recording_raised_between_step_and_step_end_leaves_older_details_alone() {
    let window = 100..250;
    let raise_at = 300;
    let switching = run(LateSwitch {
        tracer: TracingInspector::new(
            TracingInspectorConfig::none().set_step_recording(StepRecording::PcAndOp),
        ),
        seen: 0,
        window: window.clone(),
        raise_at,
    });
    let trace = &switching.tracer.traces().nodes()[0].trace;

    assert_eq!(trace.step_count(), LOOP_STEPS);
    // The window, then everything from the step after the late switch.
    let indices: Vec<usize> = trace.iter_detailed_steps().map(|step| step.step_index).collect();
    let expected: Vec<usize> = window.clone().chain(raise_at + 1..LOOP_STEPS).collect();
    assert_eq!(indices, expected);
    // The window's last detail was not refilled by the late switch's `step_end`.
    let last_of_window = trace.detailed_step(window.len() - 1).unwrap();
    assert_eq!(last_of_window.step_index, window.end - 1);
    assert!(last_of_window.gas_cost > 0);
}

/// Changes the recording level from the inspector hooks, as scripted per step index.
struct Scripted {
    tracer: TracingInspector,
    seen: usize,
    /// Level to switch to before the `n`-th step's `step`.
    on_step: fn(usize) -> Option<StepRecording>,
    /// Level to switch to before the `n`-th step's `step_end`.
    on_step_end: fn(usize) -> Option<StepRecording>,
    /// Level to switch to in the `call` hook.
    on_call: Option<StepRecording>,
}

impl Scripted {
    fn set(&mut self, level: Option<StepRecording>) {
        if let Some(level) = level {
            self.tracer.update_config(|config| config.set_step_recording(level));
        }
    }
}

impl<CTX: ContextTr<Journal: JournalExt>> Inspector<CTX, EthInterpreter> for Scripted {
    fn initialize_interp(&mut self, interp: &mut Interpreter, context: &mut CTX) {
        self.tracer.initialize_interp(interp, context);
    }

    fn step(&mut self, interp: &mut Interpreter, context: &mut CTX) {
        self.set((self.on_step)(self.seen));
        self.tracer.step(interp, context);
    }

    fn step_end(&mut self, interp: &mut Interpreter, context: &mut CTX) {
        self.set((self.on_step_end)(self.seen));
        self.seen += 1;
        self.tracer.step_end(interp, context);
    }

    fn call(&mut self, context: &mut CTX, inputs: &mut CallInputs) -> Option<CallOutcome> {
        self.set(self.on_call);
        self.tracer.call(context, inputs)
    }

    fn call_end(&mut self, context: &mut CTX, inputs: &CallInputs, outcome: &mut CallOutcome) {
        self.tracer.call_end(context, inputs, outcome);
    }
}

/// A detail pushed under `Full` is completed by `step_end` even if the level was lowered between
/// the two hooks.
#[test]
fn full_recording_lowered_between_step_and_step_end_completes_the_detail() {
    let scripted = run(Scripted {
        tracer: TracingInspector::new(
            TracingInspectorConfig::none().set_step_recording(StepRecording::PcAndOp),
        ),
        seen: 0,
        on_step: |n| (n == 200).then_some(StepRecording::Full),
        on_step_end: |n| (n == 200).then_some(StepRecording::PcAndOp),
        on_call: None,
    });
    let trace = &scripted.tracer.traces().nodes()[0].trace;

    assert_eq!(trace.step_count(), LOOP_STEPS);
    assert_eq!(trace.detailed_step_count(), 1);
    let step = trace.detailed_step(0).unwrap();
    assert_eq!(step.step_index, 200);
    // Step 200 is inside the loop, not a free `STOP`, so a completed record has a gas cost.
    assert!(step.gas_cost > 0);
}

/// Raising the level from `None` — in a frame hook, before any step of the new frame, or between
/// a step's two hooks — completes nothing: the previous detail record is left as it was, and an
/// empty frame does not trip the completion.
#[test]
fn raising_from_none_completes_no_stale_detail() {
    let mut caller = call_and_pop(CALLEE_B);
    caller.push(opcode::STOP);
    let scripted = run_code(
        Scripted {
            tracer: TracingInspector::new(
                TracingInspectorConfig::none().set_step_recording(StepRecording::Full),
            ),
            seen: 0,
            // Record the first step in full, then nothing, then raise between step 5's hooks and
            // again in the `call` hook that opens B's frame.
            on_step: |n| (n == 1).then_some(StepRecording::None),
            on_step_end: |n| (n == 5).then_some(StepRecording::Full),
            on_call: Some(StepRecording::Full),
        },
        Bytes::from(caller),
        &[(CALLEE_B, Bytes::from(vec![opcode::JUMPDEST, opcode::STOP]))],
    );
    let nodes = scripted.tracer.traces().nodes();

    // A's first step (`PUSH1`) keeps the gas cost of a single `PUSH1`, and the steps recorded
    // in full after the raise are complete.
    let first = nodes[0].trace.detailed_step(0).unwrap();
    assert_eq!(first.step_index, 0);
    assert_eq!(first.gas_cost, 3);
    for node in nodes {
        for step in node.trace.iter_detailed_steps() {
            assert!(step.gas_cost > 0 || step.op.get() == opcode::STOP, "{step:?}");
        }
    }
    assert_eq!(nodes[1].trace.step_count(), 2);
    assert_eq!(nodes[1].trace.detailed_step_count(), 2);
}

#[test]
fn clearing_steps_and_step_details() {
    let tracer = run(TracingInspector::new(
        TracingInspectorConfig::none().set_step_recording(StepRecording::Full),
    ));
    let mut node = tracer.traces().nodes()[0].clone();
    assert_eq!(node.trace.detailed_step_count(), LOOP_STEPS);

    node.clear_step_details();
    assert_eq!(node.trace.step_count(), LOOP_STEPS);
    assert_eq!(node.trace.detailed_step_count(), 0);
    assert!(!node.ordering.iter().any(|item| matches!(item, TraceMemberOrder::Step(_))));

    node.clear_steps();
    assert_eq!(node.trace.step_count(), 0);
}

#[cfg(feature = "serde")]
#[test]
fn deserialization_validates_the_store() {
    let tracer = run(Switching {
        tracer: TracingInspector::new(
            TracingInspectorConfig::none().set_step_recording(StepRecording::PcAndOp),
        ),
        seen: 0,
        window: 10..20,
    });
    let steps = &tracer.tracer.traces().nodes()[0].trace.steps;

    let json = serde_json::to_value(steps).unwrap();
    let roundtrip: revm_inspectors::tracing::types::StepStore =
        serde_json::from_value(json.clone()).unwrap();
    assert_eq!(&roundtrip, steps);

    let mut short_ops = json.clone();
    short_ops["ops"].as_array_mut().unwrap().pop();
    assert!(
        serde_json::from_value::<revm_inspectors::tracing::types::StepStore>(short_ops).is_err()
    );

    let mut unsorted = json.clone();
    unsorted["detail_steps"].as_array_mut().unwrap().swap(0, 1);
    assert!(serde_json::from_value::<revm_inspectors::tracing::types::StepStore>(unsorted).is_err());

    let mut out_of_track = json;
    out_of_track["detail_steps"][0] = serde_json::json!(LOOP_STEPS);
    assert!(
        serde_json::from_value::<revm_inspectors::tracing::types::StepStore>(out_of_track).is_err()
    );
}

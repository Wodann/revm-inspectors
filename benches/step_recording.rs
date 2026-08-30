//! Per-opcode cost of the step recording modes on a step-heavy contract.

use alloy_primitives::{hex, Address, Bytes, U256};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use revm::{
    context::TxEnv,
    context_interface::TransactTo,
    database::CacheDB,
    database_interface::EmptyDB,
    primitives::hardfork::SpecId,
    state::{AccountInfo, Bytecode},
    Context, ExecuteEvm, InspectEvm, MainBuilder, MainContext,
};
use revm_inspectors::tracing::{StepRecording, TracingInspector, TracingInspectorConfig};
use std::{cell::RefCell, hint::black_box, mem};

/// Counts down from 60 000, executing seven opcodes per iteration.
const LOOP_CODE: [u8; 13] = hex!("61ea605b600190038060035700");
/// `PUSH2`, then seven opcodes per iteration, then `STOP`.
const LOOP_STEPS: u64 = 1 + 7 * 60_000 + 1;
const CONTRACT: Address = Address::repeat_byte(0x42);
const CALLER: Address = Address::repeat_byte(0x01);

fn db() -> CacheDB<EmptyDB> {
    let mut db = CacheDB::new(EmptyDB::default());
    db.insert_account_info(
        CONTRACT,
        AccountInfo {
            code: Some(Bytecode::new_raw(Bytes::from_static(&LOOP_CODE))),
            ..Default::default()
        },
    );
    db.insert_account_info(
        CALLER,
        AccountInfo { balance: U256::from(u64::MAX), ..Default::default() },
    );
    db
}

fn tx() -> TxEnv {
    TxEnv {
        caller: CALLER,
        gas_limit: 100_000_000,
        kind: TransactTo::Call(CONTRACT),
        ..Default::default()
    }
}

fn bench_step_recording(c: &mut Criterion) {
    let mut group = c.benchmark_group("step_recording");
    group.throughput(Throughput::Elements(LOOP_STEPS));

    group.bench_function("no_inspector", |b| {
        b.iter_batched(
            || {
                Context::mainnet()
                    .with_db(db())
                    .modify_cfg_chained(|c| c.spec = SpecId::CANCUN)
                    .build_mainnet()
            },
            |mut evm| black_box(evm.transact(tx()).unwrap()),
            BatchSize::SmallInput,
        )
    });

    for (name, config) in [
        ("none", TracingInspectorConfig::none().set_step_recording(StepRecording::None)),
        ("pc_and_op", TracingInspectorConfig::none().set_step_recording(StepRecording::PcAndOp)),
        ("full", TracingInspectorConfig::none().set_step_recording(StepRecording::Full)),
        // Full recording with the snapshots a geth `debug_traceTransaction` takes.
        ("full_geth", TracingInspectorConfig::default_geth()),
    ] {
        group.bench_function(name, |b| {
            b.iter_batched(
                || {
                    Context::mainnet()
                        .with_db(db())
                        .modify_cfg_chained(|c| c.spec = SpecId::CANCUN)
                        .build_mainnet_with_inspector(TracingInspector::new(config))
                },
                |mut evm| black_box(evm.inspect_tx(tx()).unwrap()),
                BatchSize::SmallInput,
            )
        });
    }

    // Reusing one inspector across transactions pools the step stores.
    for (name, recording) in
        [("pc_and_op_fused", StepRecording::PcAndOp), ("full_fused", StepRecording::Full)]
    {
        let config = TracingInspectorConfig::none().set_step_recording(recording);
        let inspector = RefCell::new(Some(TracingInspector::new(config)));
        group.bench_function(name, |b| {
            b.iter_batched(
                || {
                    Context::mainnet()
                        .with_db(db())
                        .modify_cfg_chained(|c| c.spec = SpecId::CANCUN)
                        .build_mainnet_with_inspector(inspector.take().unwrap())
                },
                |mut evm| {
                    let res = black_box(evm.inspect_tx(tx()).unwrap());
                    let mut fused = mem::take(&mut evm.inspector);
                    fused.fuse();
                    inspector.replace(Some(fused));
                    res
                },
                // Setup and routine alternate, so the one inspector is always back in its slot.
                BatchSize::PerIteration,
            )
        });
    }
    group.finish();
}

criterion_group!(benches, bench_step_recording);
criterion_main!(benches);

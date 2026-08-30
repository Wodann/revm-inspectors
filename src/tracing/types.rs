//! Types for representing call trace items.

use crate::tracing::{config::TraceStyle, utils, utils::convert_memory};
use alloc::{
    boxed::Box,
    collections::VecDeque,
    format,
    string::{String, ToString},
    vec::Vec,
};
pub use alloy_primitives::Log;
use alloy_primitives::{Address, Bytes, FixedBytes, LogData, U256};
use alloy_rpc_types_trace::{
    geth::{CallFrame, CallLogFrame, GethDefaultTracingOptions, StructLog},
    parity::{
        Action, ActionType, CallAction, CallOutput, CallType, CreateAction, CreateOutput,
        CreationMethod, SelfdestructAction, TraceOutput, TransactionTrace,
    },
};
use revm::{
    bytecode::opcode::{self, OpCode},
    interpreter::{CallScheme, CreateScheme, InstructionResult},
};

/// Decoded call data.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DecodedCallData {
    /// The function signature.
    pub signature: String,
    /// The function arguments.
    pub args: Vec<String>,
}

/// Additional decoded data enhancing the [CallTrace].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DecodedCallTrace {
    /// Optional decoded label for the call.
    pub label: Option<String>,
    /// Optional decoded return data.
    pub return_data: Option<String>,
    /// Optional decoded call data.
    pub call_data: Option<DecodedCallData>,
}

/// A trace of a call with optional decoded data.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CallTrace {
    /// The depth of the call.
    ///
    /// Zero represents the root trace node.
    ///
    /// The depth of a call's steps is the depth of the call plus one.
    pub depth: usize,
    /// Whether the call was successful.
    pub success: bool,
    /// The caller address.
    pub caller: Address,
    /// The target address of this call.
    ///
    /// This is:
    /// - [`CallKind::Call`] and alike: the callee, the address of the contract being called
    /// - [`CallKind::Create`] and alike: the address of the created contract
    pub address: Address,
    /// Whether this is a call to a precompile.
    ///
    /// Note: This is optional because not all tracers make use of this.
    pub maybe_precompile: Option<bool>,
    /// The address of the selfdestructed contract.
    pub selfdestruct_address: Option<Address>,
    /// Holds the target for the selfdestruct refund target.
    ///
    /// This is only `Some` if a selfdestruct was executed and the call is executed before the
    /// Cancun hardfork.
    ///
    /// See [`is_selfdestruct`](Self::is_selfdestruct) for more information.
    pub selfdestruct_refund_target: Option<Address>,
    /// The value transferred on a selfdestruct.
    ///
    /// This is only `Some` if a selfdestruct was executed and the call is executed before the
    /// Cancun hardfork.
    ///
    /// See [`is_selfdestruct`](Self::is_selfdestruct) for more information.
    pub selfdestruct_transferred_value: Option<U256>,
    /// The kind of call.
    pub kind: CallKind,
    /// The value transferred in the call.
    pub value: U256,
    /// The calldata/input, or the init code for contract creations.
    pub data: Bytes,
    /// The return data, or the runtime bytecode of the created contract.
    pub output: Bytes,
    /// The total gas cost of the call.
    pub gas_used: u64,
    /// The gas limit of the call.
    pub gas_limit: u64,
    /// The cumulative refund counter for the entire transaction context at the end of this call.
    pub gas_refund_counter: u64,
    /// The final status of the call.
    pub status: Option<InstructionResult>,
    /// Opcode-level execution steps: the program counter and opcode of every executed step, and
    /// a detail record for the steps that were recorded in full (see [`StepStore`]).
    ///
    /// Prefer reading these through [`Self::iter_steps`], [`Self::iter_detailed_steps`] and
    /// [`Self::detailed_step`]: direct access couples the reader to how steps are stored. A trace
    /// serialized without steps deserializes with an empty store.
    #[cfg_attr(feature = "serde", serde(default))]
    pub steps: StepStore,
    /// Optional complementary decoded call data.
    pub decoded: Option<Box<DecodedCallTrace>>,
}

impl CallTrace {
    /// Returns how many steps the base track holds: every step executed while step recording was
    /// enabled. See [`Self::detailed_step_count`] for the steps recorded in full.
    #[inline]
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Returns how many steps were recorded in full, i.e. how many [`Self::iter_detailed_steps`]
    /// yields.
    #[inline]
    pub fn detailed_step_count(&self) -> usize {
        self.steps.detailed_len()
    }

    /// Returns a view of every step in the base track, in execution order.
    #[inline]
    pub fn iter_steps(
        &self,
    ) -> impl ExactSizeIterator<Item = StepRef<'_>> + DoubleEndedIterator + Clone {
        self.steps.iter()
    }

    /// Returns a view of the steps that were recorded in full, in execution order.
    #[inline]
    pub fn iter_detailed_steps(
        &self,
    ) -> impl ExactSizeIterator<Item = DetailedStepRef<'_>> + DoubleEndedIterator + Clone {
        self.steps.iter_detailed()
    }

    /// Returns the step recorded in full at `idx` in the detail overlay.
    ///
    /// Indices come from the enclosing [`CallTraceNode::ordering`] (a [`TraceMemberOrder::Step`]
    /// entry) and from [`DecodedTraceStep::InternalCall`].
    #[inline]
    pub fn detailed_step(&self, idx: usize) -> Option<DetailedStepRef<'_>> {
        self.steps.detailed(idx)
    }

    /// Returns the last step recorded in full, if any.
    #[inline]
    pub fn last_detailed_step(&self) -> Option<DetailedStepRef<'_>> {
        self.steps.last_detailed()
    }

    /// Returns true if the status code is an error or revert, See [InstructionResult::Revert]
    #[inline]
    pub const fn is_error(&self) -> bool {
        let Some(status) = self.status else {
            return false;
        };
        !status.is_ok()
    }

    /// Returns true if the status code is a revert.
    #[inline]
    pub fn is_revert(&self) -> bool {
        self.status.is_some_and(|status| status == InstructionResult::Revert)
    }

    /// Returns `true` if this trace was a selfdestruct.
    ///
    /// See also `TracingInspector::selfdestruct`.
    ///
    /// We can't rely entirely on [`Self::status`] being [`InstructionResult::SelfDestruct`]
    /// because there's an edge case where a new created contract (CREATE) is immediately
    /// selfdestructed.
    ///
    /// We also can't rely entirely on `selfdestruct_refund_target` being `Some` as the
    /// `selfdestruct` inspector function will not be called after the Cancun hardfork.
    #[inline]
    pub const fn is_selfdestruct(&self) -> bool {
        matches!(self.status, Some(InstructionResult::SelfDestruct))
            || self.selfdestruct_refund_target.is_some()
    }

    /// Returns the error message if it is an erroneous result.
    pub(crate) fn as_error_msg(&self, kind: TraceStyle) -> Option<String> {
        self.status.and_then(|status| utils::fmt_error_msg(status, kind))
    }

    /// Gets the decoded call trace.
    ///
    /// Initializes with the default value if not yet set.
    pub fn decoded(&mut self) -> &mut DecodedCallTrace {
        self.decoded.get_or_insert_with(Default::default)
    }

    #[allow(dead_code)]
    pub(crate) fn decoded_label<'a>(&'a self, fallback: &'a str) -> &'a str {
        self.decoded.as_ref().and_then(|d| d.label.as_deref()).unwrap_or(fallback)
    }

    #[allow(dead_code)]
    pub(crate) fn decoded_call_data(&self) -> Option<&DecodedCallData> {
        self.decoded.as_ref()?.call_data.as_ref()
    }

    #[allow(dead_code)]
    pub(crate) fn decoded_return_data(&self) -> Option<&str> {
        self.decoded.as_ref()?.return_data.as_deref()
    }
}

/// Additional decoded data enhancing the [CallLog].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DecodedCallLog {
    /// The decoded event name.
    pub name: Option<String>,
    /// The decoded log parameters, a vector of the parameter name (e.g. foo) and the parameter
    /// value (e.g. 0x9d3...45ca).
    pub params: Option<Vec<(String, String)>>,
}

/// A log with optional decoded data.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CallLog {
    /// The address of the log emitter.
    pub address: Address,
    /// The raw log data.
    pub raw_log: LogData,
    /// Optional complementary decoded log data.
    pub decoded: Option<Box<DecodedCallLog>>,
    /// The position of the log relative to subcalls within the same trace.
    pub position: u64,
    /// The position of the log relative to subcalls within the same trace.
    pub index: u64,
}

impl From<Log> for CallLog {
    /// Converts a [`Log`] into a [`CallLog`].
    fn from(log: Log) -> Self {
        Self { address: log.address, raw_log: log.data, decoded: None, position: 0, index: 0 }
    }
}

impl CallLog {
    /// Sets the position of the log.
    #[inline]
    pub const fn with_position(mut self, position: u64) -> Self {
        self.position = position;
        self
    }

    /// Sets index of the log in the transaction.
    #[inline]
    pub const fn with_index(mut self, index: u64) -> Self {
        self.index = index;
        self
    }

    /// Gets the decoded call log.
    ///
    /// Initializes with the default value if not yet set.
    pub fn decoded(&mut self) -> &mut DecodedCallLog {
        self.decoded.get_or_insert_with(Default::default)
    }

    #[allow(dead_code)]
    pub(crate) fn decoded_name(&self) -> Option<&str> {
        self.decoded.as_deref()?.name.as_deref()
    }

    #[allow(dead_code)]
    pub(crate) fn decoded_params(&self) -> Option<&[(String, String)]> {
        self.decoded.as_deref()?.params.as_deref()
    }
}

/// A node in the arena
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CallTraceNode {
    /// Parent node index in the arena
    pub parent: Option<usize>,
    /// Children node indexes in the arena
    pub children: Vec<usize>,
    /// This node's index in the arena
    pub idx: usize,
    /// The call trace
    pub trace: CallTrace,
    /// Recorded logs, if enabled
    pub logs: Vec<CallLog>,
    /// Ordering of child calls and logs
    pub ordering: Vec<TraceMemberOrder>,
}

impl CallTraceNode {
    /// Discards the recorded EVM steps — base track and detail overlay — and their entries in the
    /// ordering, keeping the calls and logs.
    pub fn clear_steps(&mut self) {
        self.trace.steps = StepStore::default();
        self.retain_non_step_ordering();
    }

    /// Discards the detail overlay and its entries in the ordering, keeping the base track, the
    /// calls and the logs.
    pub fn clear_step_details(&mut self) {
        self.trace.steps.clear_details();
        self.retain_non_step_ordering();
    }

    fn retain_non_step_ordering(&mut self) {
        self.ordering.retain(|item| !matches!(item, TraceMemberOrder::Step(_)));
        // `retain` keeps the capacity, which grew with one entry per detailed step.
        self.ordering.shrink_to_fit();
    }

    /// Returns the call context's execution address
    ///
    /// See `Inspector::call` impl of [TracingInspector](crate::tracing::TracingInspector)
    pub const fn execution_address(&self) -> Address {
        if self.trace.kind.is_delegate() {
            self.trace.caller
        } else {
            self.trace.address
        }
    }

    /// Pushes all steps onto the stack in reverse order
    /// so that the first step is on top of the stack
    pub(crate) fn push_steps_on_stack<'a>(
        &'a self,
        stack: &mut VecDeque<CallTraceStepStackItem<'a>>,
    ) {
        // Children are attributed to call-like steps in execution order, counted over the whole
        // base track so that a call executed outside a full-recording window still advances to
        // the next child. A call-like step without a child is one that reverted or ran out of gas
        // before a frame was opened: <https://github.com/paradigmxyz/reth/issues/3915>. Walking
        // the steps backwards, the children are therefore handed out from the last attributed one
        // down.
        let call_like_steps = self.trace.iter_steps().filter(|step| step.is_call_like_op()).count();
        let mut child_id = call_like_steps.min(self.children.len());
        let mut unattributed = call_like_steps - child_id;

        stack.reserve(self.trace.steps.detailed_len() + child_id);
        for step in self.trace.iter_steps().rev() {
            let mut call_child_id = None;
            if step.is_call_like_op() {
                if unattributed > 0 {
                    unattributed -= 1;
                } else {
                    child_id -= 1;
                    call_child_id = Some(self.children[child_id]);
                }
            }
            // A step recorded in full becomes a struct log; a call-like step that was not still
            // has to push its child frame, whose steps may have been recorded in full.
            let step = step.detailed();
            if step.is_some() || call_child_id.is_some() {
                stack.push_back(CallTraceStepStackItem { trace_node: self, step, call_child_id });
            }
        }
    }

    /// Returns how many logs this trace already has.
    #[inline]
    pub(crate) fn log_count(&self) -> usize {
        self.logs.len()
    }

    /// Returns true if this is a call to a precompile
    #[inline]
    pub fn is_precompile(&self) -> bool {
        self.trace.maybe_precompile.unwrap_or(false)
    }

    /// Returns the kind of call the trace belongs to
    #[inline]
    pub const fn kind(&self) -> CallKind {
        self.trace.kind
    }

    /// Returns the status of the call
    #[inline]
    pub const fn status(&self) -> Option<InstructionResult> {
        self.trace.status
    }

    /// Returns the call context's 4 byte selector
    pub fn selector(&self) -> Option<FixedBytes<4>> {
        (self.trace.data.len() >= 4).then(|| FixedBytes::from_slice(&self.trace.data[..4]))
    }

    /// Returns `true` if this trace was a selfdestruct.
    ///
    /// See [`CallTrace::is_selfdestruct`] for more details.
    #[inline]
    pub const fn is_selfdestruct(&self) -> bool {
        self.trace.is_selfdestruct()
    }

    /// Converts this node into a parity `TransactionTrace`
    pub fn parity_transaction_trace(&self, trace_address: Vec<usize>) -> TransactionTrace {
        let action = self.parity_action();
        let result = if self.trace.is_error() && !self.trace.is_revert() {
            // if the trace is a selfdestruct or an error that is not a revert, the result is None
            None
        } else {
            Some(self.parity_trace_output())
        };
        let error = self.trace.as_error_msg(TraceStyle::Parity);
        TransactionTrace { action, error, result, trace_address, subtraces: self.children.len() }
    }

    /// Returns the `Output` for a parity trace
    pub fn parity_trace_output(&self) -> TraceOutput {
        match self.kind() {
            CallKind::Call
            | CallKind::StaticCall
            | CallKind::CallCode
            | CallKind::DelegateCall
            | CallKind::AuthCall => TraceOutput::Call(CallOutput {
                gas_used: self.trace.gas_used,
                output: self.trace.output.clone(),
            }),
            CallKind::Create | CallKind::Create2 => TraceOutput::Create(CreateOutput {
                gas_used: self.trace.gas_used,
                code: self.trace.output.clone(),
                address: self.trace.address,
            }),
        }
    }

    /// If the trace is a selfdestruct, returns the `Action` for a parity trace.
    pub fn parity_selfdestruct_action(&self) -> Option<Action> {
        self.is_selfdestruct().then(|| {
            Action::Selfdestruct(SelfdestructAction {
                address: self.trace.selfdestruct_address.unwrap_or_default(),
                refund_address: self.trace.selfdestruct_refund_target.unwrap_or_default(),
                balance: self.trace.selfdestruct_transferred_value.unwrap_or_default(),
            })
        })
    }

    /// If the trace is a selfdestruct, returns the `CallFrame` for a geth call trace
    pub fn geth_selfdestruct_call_trace(&self) -> Option<CallFrame> {
        self.is_selfdestruct().then(|| CallFrame {
            typ: "SELFDESTRUCT".to_string(),
            from: self.trace.selfdestruct_address.unwrap_or_default(),
            to: self.trace.selfdestruct_refund_target,
            value: self.trace.selfdestruct_transferred_value,
            ..Default::default()
        })
    }

    /// If the trace is a selfdestruct, returns the `TransactionTrace` for a parity trace.
    pub fn parity_selfdestruct_trace(&self, trace_address: Vec<usize>) -> Option<TransactionTrace> {
        let trace = self.parity_selfdestruct_action()?;
        Some(TransactionTrace {
            action: trace,
            error: None,
            result: None,
            trace_address,
            subtraces: 0,
        })
    }

    /// Returns the `Action` for a parity trace.
    ///
    /// Caution: This does not include the selfdestruct action, if the trace is a selfdestruct,
    /// since those are handled in addition to the call action.
    pub fn parity_action(&self) -> Action {
        match self.kind() {
            CallKind::Call
            | CallKind::StaticCall
            | CallKind::CallCode
            | CallKind::DelegateCall
            | CallKind::AuthCall => Action::Call(CallAction {
                from: self.trace.caller,
                to: self.trace.address,
                value: self.trace.value,
                gas: self.trace.gas_limit,
                input: self.trace.data.clone(),
                call_type: self.kind().into(),
            }),
            CallKind::Create | CallKind::Create2 => Action::Create(CreateAction {
                from: self.trace.caller,
                value: self.trace.value,
                gas: self.trace.gas_limit,
                init: self.trace.data.clone(),
                creation_method: self.kind().into(),
            }),
        }
    }

    /// Converts this call trace into an _empty_ geth [CallFrame]
    pub fn geth_empty_call_frame(&self, include_logs: bool) -> CallFrame {
        #[allow(clippy::needless_update)]
        let mut call_frame = CallFrame {
            typ: self.trace.kind.to_string(),
            from: self.trace.caller,
            to: Some(self.trace.address),
            value: Some(self.trace.value),
            gas: U256::from(self.trace.gas_limit),
            gas_used: U256::from(self.trace.gas_used),
            input: self.trace.data.clone(),
            output: (!self.trace.output.is_empty()).then(|| self.trace.output.clone()),
            error: None,
            revert_reason: None,
            calls: Default::default(),
            logs: Default::default(),
            ..Default::default()
        };

        if self.trace.kind.is_static_call() {
            // STATICCALL frames don't have a value
            call_frame.value = None;
        }

        // we need to populate error and revert reason
        if !self.trace.success {
            if self.kind().is_any_create() {
                call_frame.to = None;
            }

            if !self.status().is_some_and(|status| status.is_revert()) {
                call_frame.gas_used = U256::from(self.trace.gas_limit);
                call_frame.output = None;
            }

            call_frame.revert_reason = utils::maybe_revert_reason(self.trace.output.as_ref());

            // Note: regular calltracer uses geth errors, only flatCallTracer uses parity errors: <https://github.com/ethereum/go-ethereum/blob/a9523b6428238a762e1a1e55e46ead47630c3a23/eth/tracers/native/call_flat.go#L226>
            call_frame.error = self.trace.as_error_msg(TraceStyle::Geth);
        }

        if include_logs && !self.logs.is_empty() {
            call_frame.logs = self
                .logs
                .iter()
                .map(|log| CallLogFrame {
                    address: Some(log.address),
                    topics: Some(log.raw_log.topics().to_vec()),
                    data: Some(log.raw_log.data.clone()),
                    position: Some(log.position),
                    index: Some(log.index),
                })
                .collect();
        }

        call_frame
    }
}

/// A unified representation of a call.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "UPPERCASE"))]
pub enum CallKind {
    /// Represents a regular call.
    #[default]
    Call,
    /// Represents a static call.
    StaticCall,
    /// Represents a call code operation.
    CallCode,
    /// Represents a delegate call.
    DelegateCall,
    /// Represents an authorized call.
    AuthCall,
    /// Represents a contract creation operation.
    Create,
    /// Represents a contract creation operation using the CREATE2 opcode.
    Create2,
}

impl CallKind {
    /// Returns the string representation of the call kind.
    pub const fn to_str(self) -> &'static str {
        match self {
            Self::Call => "CALL",
            Self::StaticCall => "STATICCALL",
            Self::CallCode => "CALLCODE",
            Self::DelegateCall => "DELEGATECALL",
            Self::AuthCall => "AUTHCALL",
            Self::Create => "CREATE",
            Self::Create2 => "CREATE2",
        }
    }

    /// Returns true if the call is a create
    #[inline]
    pub const fn is_any_create(&self) -> bool {
        matches!(self, Self::Create | Self::Create2)
    }

    /// Returns true if the call is a delegate of some sorts
    #[inline]
    pub const fn is_delegate(&self) -> bool {
        matches!(self, Self::DelegateCall | Self::CallCode)
    }

    /// Returns true if the call is [CallKind::StaticCall].
    #[inline]
    pub const fn is_static_call(&self) -> bool {
        matches!(self, Self::StaticCall)
    }

    /// Returns true if the call is [CallKind::AuthCall].
    #[inline]
    pub const fn is_auth_call(&self) -> bool {
        matches!(self, Self::AuthCall)
    }
}

impl From<CallKind> for CreationMethod {
    fn from(kind: CallKind) -> CreationMethod {
        match kind {
            CallKind::Create => CreationMethod::Create,
            CallKind::Create2 => CreationMethod::Create2,
            _ => CreationMethod::None,
        }
    }
}

impl core::fmt::Display for CallKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.to_str())
    }
}

impl From<CallScheme> for CallKind {
    fn from(scheme: CallScheme) -> Self {
        match scheme {
            CallScheme::Call => Self::Call,
            CallScheme::StaticCall => Self::StaticCall,
            CallScheme::DelegateCall => Self::DelegateCall,
            CallScheme::CallCode => Self::CallCode,
        }
    }
}

impl From<CreateScheme> for CallKind {
    fn from(create: CreateScheme) -> Self {
        match create {
            CreateScheme::Create => Self::Create,
            CreateScheme::Create2 { .. } => Self::Create2,
            CreateScheme::Custom { .. } => Self::Create,
        }
    }
}

impl From<CallKind> for ActionType {
    fn from(kind: CallKind) -> Self {
        match kind {
            CallKind::Call
            | CallKind::StaticCall
            | CallKind::DelegateCall
            | CallKind::CallCode
            | CallKind::AuthCall => Self::Call,
            CallKind::Create | CallKind::Create2 => Self::Create,
        }
    }
}

impl From<CallKind> for CallType {
    fn from(ty: CallKind) -> Self {
        match ty {
            CallKind::Call => Self::Call,
            CallKind::StaticCall => Self::StaticCall,
            CallKind::CallCode => Self::CallCode,
            CallKind::DelegateCall => Self::DelegateCall,
            CallKind::Create | CallKind::Create2 => Self::None,
            CallKind::AuthCall => Self::AuthCall,
        }
    }
}

pub(crate) struct CallTraceStepStackItem<'a> {
    /// The trace node that contains this step
    pub(crate) trace_node: &'a CallTraceNode,
    /// The step that this stack item represents
    ///
    /// `None` for a call-like step that was not recorded in full but has a child frame to push.
    pub(crate) step: Option<DetailedStepRef<'a>>,
    /// The index of the child call in the CallArena if this step's opcode is a call
    pub(crate) call_child_id: Option<usize>,
}

/// Ordering enum for calls, logs and steps
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TraceMemberOrder {
    /// Contains the index of the corresponding log
    Log(usize),
    /// Contains the index of the corresponding trace node
    Call(usize),
    /// Contains the index of the corresponding step in the call's detail overlay; only steps
    /// recorded in full get an entry. See [`StepStore::detailed`].
    Step(usize),
}

/// Represents a decoded internal function call.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DecodedInternalCall {
    /// Name of the internal function.
    pub func_name: String,
    /// Input arguments of the internal function.
    pub args: Option<Vec<String>>,
    /// Optional decoded return data.
    pub return_data: Option<Vec<String>>,
}

/// Represents a decoded trace step. Currently two formats are supported.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DecodedTraceStep {
    /// Decoded internal function call. Displayed similarly to external calls.
    ///
    /// Keeps decoded internal call data and the index of the detailed step where the internal call
    /// execution ends — an index into the call's detail overlay, as a [`TraceMemberOrder::Step`]
    /// entry carries and [`CallTrace::detailed_step`] resolves.
    InternalCall(DecodedInternalCall, usize),
    /// Arbitrary line representing the step. Might be used for displaying individual opcodes.
    Line(String),
}

/// The detail record of a step that was recorded in full: everything about the step besides its
/// program counter and opcode, which every step keeps in [`StepStore`]'s base track.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StepDetail {
    // Fields filled in `step`
    /// Stack before step execution
    pub stack: Option<Box<[U256]>>,
    /// The new stack items placed by this step if any
    pub push_stack: Option<Box<[U256]>>,
    /// Memory before step execution.
    ///
    /// This will be `None` only if memory capture is disabled.
    pub memory: Option<RecordedMemory>,
    /// Returndata before step execution
    pub returndata: Bytes,
    /// Remaining gas before step execution
    pub gas_remaining: u64,
    /// Gas refund counter before step execution
    pub gas_refund_counter: u64,
    /// Total gas used before step execution
    pub gas_used: u64,
    // Fields filled in `step_end`
    /// Gas cost of step execution
    pub gas_cost: u64,
    /// Change of the contract state after step execution (effect of the SLOAD/SSTORE instructions)
    pub storage_change: Option<Box<StorageChange>>,
    /// Final status of the step
    ///
    /// This is set after the step was executed.
    pub status: Option<InstructionResult>,
    /// Immediate bytes of the step
    pub immediate_bytes: Option<Bytes>,
    /// Optional complementary decoded step data.
    pub decoded: Option<Box<DecodedTraceStep>>,
}

impl StepDetail {
    // Returns true if the status code is an error or revert, See [InstructionResult::Revert]
    #[inline]
    pub(crate) const fn is_error(&self) -> bool {
        let Some(status) = self.status else {
            return false;
        };
        status.is_halt()
    }

    /// Returns the error message if it is an erroneous result.
    #[inline]
    pub(crate) fn as_error(&self) -> Option<String> {
        self.is_error().then(|| format!("{:?}", self.status))
    }

    /// Returns `DecodedTraceStep` from `StepDetail`.
    pub fn decoded_mut(&mut self) -> &mut DecodedTraceStep {
        self.decoded.get_or_insert_with(|| Box::new(DecodedTraceStep::Line(String::new())))
    }

    /// Returns the heap bytes the record owns: the snapshots and the boxed fields.
    fn owned_bytes(&self) -> usize {
        self.stack.as_ref().map_or(0, |stack| stack.len() * core::mem::size_of::<U256>())
            + self.push_stack.as_ref().map_or(0, |stack| stack.len() * core::mem::size_of::<U256>())
            + self.memory.as_ref().map_or(0, RecordedMemory::len)
            + self.returndata.len()
            + self.immediate_bytes.as_ref().map_or(0, |bytes| bytes.len())
            + self.storage_change.as_ref().map_or(0, |_| core::mem::size_of::<StorageChange>())
            + self.decoded.as_ref().map_or(0, |_| core::mem::size_of::<DecodedTraceStep>())
    }
}

/// An owned copy of one recorded step: its program counter and opcode, plus the detail record when
/// the step was recorded in full. Produced by [`StepRef::into_owned`]; the tracer stores steps in a
/// [`StepStore`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallTraceStep {
    /// Program counter before step execution
    pub pc: usize,
    /// Opcode to be executed
    pub op: OpCode,
    /// The detail record, if the step was recorded in full.
    pub detail: Option<StepDetail>,
}

/// The steps a call executed.
///
/// Every step executed while step recording was enabled contributes its program counter and
/// opcode to a dense *base track*. The steps that were *recorded in full* — all of them under
/// [`StepRecording::Full`](crate::tracing::StepRecording), or those inside a window in which full
/// recording was switched on — additionally have a [`StepDetail`] in a sparse *detail overlay*
/// ordered by step. [`TraceMemberOrder::Step`] entries in a node's ordering index the overlay: only
/// detailed steps are interleaved with calls and logs.
///
/// # Invariants
///
/// Every step has a program counter and an opcode ([`Self::len`] counts them); every detail record
/// belongs to a distinct step of the base track ([`Self::detailed_len`] is at most `len`), and
/// [`Self::iter_detailed`] yields them in strictly increasing `step_index` order. Deserialization
/// rejects input that violates them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct StepStore {
    /// Program counter of each executed step.
    pcs: Vec<u32>,
    /// Opcode of each executed step.
    ops: Vec<u8>,
    /// Base-track index of each detail record, strictly increasing.
    detail_steps: Vec<u32>,
    /// Detail records of the steps recorded in full, in step order.
    details: Vec<StepDetail>,
}

impl StepStore {
    /// Returns how many steps the base track holds: every step executed while step recording was
    /// enabled.
    #[inline]
    pub fn len(&self) -> usize {
        self.pcs.len()
    }

    /// Returns true if the base track is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pcs.is_empty()
    }

    /// Returns how many steps were recorded in full.
    #[inline]
    pub fn detailed_len(&self) -> usize {
        self.details.len()
    }

    /// Returns the opcode of the step at `idx` in the base track.
    ///
    /// # Panics
    ///
    /// If `idx` is out of bounds.
    #[inline]
    pub(crate) fn op_at(&self, idx: usize) -> u8 {
        self.ops[idx]
    }

    /// Returns a view of the step at `idx` in the base track.
    ///
    /// The detail record, if any, is located by a binary search of the overlay; iterate with
    /// [`Self::iter`] to walk both tracks in one pass.
    #[inline]
    pub fn get(&self, idx: usize) -> Option<StepRef<'_>> {
        let pc = *self.pcs.get(idx)?;
        let op = OpCode::new_or_unknown(self.ops[idx]);
        let detail = self.detail_steps.binary_search(&(idx as u32)).ok().map(|k| &self.details[k]);
        Some(StepRef { pc: pc as usize, op, step_index: idx, detail })
    }

    /// Returns a view of every step in the base track, in execution order.
    #[inline]
    pub fn iter(&self) -> impl ExactSizeIterator<Item = StepRef<'_>> + DoubleEndedIterator + Clone {
        StepIter {
            store: self,
            front: 0,
            back: self.pcs.len(),
            detail_front: 0,
            detail_back: self.details.len(),
        }
    }

    /// Returns a view of the steps that were recorded in full, in execution order.
    #[inline]
    pub fn iter_detailed(
        &self,
    ) -> impl ExactSizeIterator<Item = DetailedStepRef<'_>> + DoubleEndedIterator + Clone {
        self.detail_steps
            .iter()
            .zip(&self.details)
            .map(|(&step_index, detail)| self.detailed_ref(step_index as usize, detail))
    }

    /// Returns the `k`-th detailed step, `k` being the index a [`TraceMemberOrder::Step`] entry
    /// carries: the position of the step's detail record in the overlay, not its position in the
    /// base track.
    #[inline]
    pub fn detailed(&self, k: usize) -> Option<DetailedStepRef<'_>> {
        let detail = self.details.get(k)?;
        Some(self.detailed_ref(self.detail_steps[k] as usize, detail))
    }

    /// Returns the last step recorded in full, if any.
    #[inline]
    pub fn last_detailed(&self) -> Option<DetailedStepRef<'_>> {
        self.detailed(self.details.len().checked_sub(1)?)
    }

    /// Returns the `k`-th detail record; see [`Self::detailed`] for the index.
    #[inline]
    pub fn detail(&self, k: usize) -> Option<&StepDetail> {
        self.details.get(k)
    }

    /// Returns the `k`-th detail record mutably — for decoders that fill in
    /// [`StepDetail::decoded`]; see [`Self::detailed`] for the index.
    #[inline]
    pub fn detail_mut(&mut self, k: usize) -> Option<&mut StepDetail> {
        self.details.get_mut(k)
    }

    /// Returns the last detail record together with the base-track index of its step, mutably.
    #[inline]
    pub(crate) fn last_detail_mut(&mut self) -> Option<(usize, &mut StepDetail)> {
        let step_index = *self.detail_steps.last()? as usize;
        Some((step_index, self.details.last_mut()?))
    }

    /// Appends a step to the base track and returns its index.
    ///
    /// The inspector records steps itself; this is for building traces by hand, e.g. in tests.
    #[inline]
    pub fn push(&mut self, pc: usize, op: OpCode) -> usize {
        // Code size is bounded by the gas cost of supplying it, far below `u32::MAX`, so the cast
        // cannot truncate.
        debug_assert!(u32::try_from(pc).is_ok(), "program counter {pc} exceeds u32");
        self.pcs.push(pc as u32);
        self.ops.push(op.get());
        self.pcs.len() - 1
    }

    /// Appends a detail record for the step at `step_index`, which must be the last step of the
    /// base track, and returns its index, i.e. the value a [`TraceMemberOrder::Step`] entry for it
    /// should carry.
    ///
    /// # Panics
    ///
    /// If `step_index` is not the last step of the base track, or that step already has a detail
    /// record.
    #[inline]
    pub fn push_detail(&mut self, step_index: usize, detail: StepDetail) -> usize {
        assert_eq!(step_index + 1, self.pcs.len(), "a detail record belongs to the last step");
        assert_ne!(
            self.detail_steps.last().copied(),
            Some(step_index as u32),
            "a step has at most one detail record"
        );
        self.detail_steps.push(step_index as u32);
        self.details.push(detail);
        self.details.len() - 1
    }

    /// Discards the detail records, keeping the base track.
    pub(crate) fn clear_details(&mut self) {
        self.detail_steps = Vec::new();
        self.details = Vec::new();
    }

    /// Clears the store for reuse, keeping the allocated capacity.
    pub(crate) fn reset(&mut self) {
        self.pcs.clear();
        self.ops.clear();
        self.detail_steps.clear();
        self.details.clear();
    }

    /// Returns an estimate of the heap bytes the store keeps alive: its columns and the snapshots
    /// and boxed fields the detail records own. Shared [`Bytes`] buffers are counted once per
    /// record that references them, and the strings inside [`DecodedTraceStep`] are not counted.
    pub fn bytes(&self) -> usize {
        self.capacity_bytes() + self.details.iter().map(StepDetail::owned_bytes).sum::<usize>()
    }

    /// Returns the heap bytes the store's columns occupy, including unused capacity.
    pub(crate) fn capacity_bytes(&self) -> usize {
        self.pcs.capacity() * core::mem::size_of::<u32>()
            + self.ops.capacity()
            + self.detail_steps.capacity() * core::mem::size_of::<u32>()
            + self.details.capacity() * core::mem::size_of::<StepDetail>()
    }

    #[inline]
    fn detailed_ref<'a>(
        &'a self,
        step_index: usize,
        detail: &'a StepDetail,
    ) -> DetailedStepRef<'a> {
        DetailedStepRef {
            pc: self.pcs[step_index] as usize,
            op: OpCode::new_or_unknown(self.ops[step_index]),
            step_index,
            detail,
        }
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for StepStore {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;

        #[derive(serde::Deserialize)]
        struct Repr {
            pcs: Vec<u32>,
            ops: Vec<u8>,
            detail_steps: Vec<u32>,
            details: Vec<StepDetail>,
        }

        let Repr { pcs, ops, detail_steps, details } = Repr::deserialize(deserializer)?;
        if pcs.len() != ops.len() {
            return Err(D::Error::custom("`pcs` and `ops` differ in length"));
        }
        if detail_steps.len() != details.len() {
            return Err(D::Error::custom("`detail_steps` and `details` differ in length"));
        }
        let in_range = detail_steps.iter().all(|&step| (step as usize) < pcs.len());
        let increasing = detail_steps.windows(2).all(|pair| pair[0] < pair[1]);
        if !in_range || !increasing {
            return Err(D::Error::custom(
                "`detail_steps` must be strictly increasing indices into the base track",
            ));
        }
        Ok(Self { pcs, ops, detail_steps, details })
    }
}

/// Iterator over every step of a [`StepStore`], walking the base track and the detail overlay in
/// one pass.
#[derive(Clone)]
struct StepIter<'a> {
    store: &'a StepStore,
    front: usize,
    back: usize,
    /// Detail records not yet yielded from the front.
    detail_front: usize,
    /// Detail records not yet yielded from the back.
    detail_back: usize,
}

impl<'a> StepIter<'a> {
    #[inline]
    fn step(&self, idx: usize, detail: Option<usize>) -> StepRef<'a> {
        let store = self.store;
        StepRef {
            pc: store.pcs[idx] as usize,
            op: OpCode::new_or_unknown(store.ops[idx]),
            step_index: idx,
            detail: detail.map(|k| &store.details[k]),
        }
    }
}

impl<'a> Iterator for StepIter<'a> {
    type Item = StepRef<'a>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.front == self.back {
            return None;
        }
        let idx = self.front;
        self.front += 1;
        let detail = (self.detail_front < self.detail_back
            && self.store.detail_steps[self.detail_front] as usize == idx)
            .then(|| {
                self.detail_front += 1;
                self.detail_front - 1
            });
        Some(self.step(idx, detail))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.back - self.front;
        (len, Some(len))
    }
}

impl DoubleEndedIterator for StepIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front == self.back {
            return None;
        }
        self.back -= 1;
        let idx = self.back;
        let detail = (self.detail_front < self.detail_back
            && self.store.detail_steps[self.detail_back - 1] as usize == idx)
            .then(|| {
                self.detail_back -= 1;
                self.detail_back
            });
        Some(self.step(idx, detail))
    }
}

impl ExactSizeIterator for StepIter<'_> {}

/// A borrowed view of one recorded step: its program counter and opcode, plus the detail record
/// when the step was recorded in full.
///
/// Consumers read steps through [`CallTrace::iter_steps`] and [`CallTrace::iter_detailed_steps`]
/// rather than through the storage in [`CallTrace::steps`].
#[derive(Clone, Copy, Debug)]
pub struct StepRef<'a> {
    /// Program counter before step execution
    pub pc: usize,
    /// Opcode to be executed
    pub op: OpCode,
    /// Index of the step in the call's base track.
    pub step_index: usize,
    detail: Option<&'a StepDetail>,
}

impl<'a> StepRef<'a> {
    /// Returns the step's detail record, if it was recorded in full, borrowed from the trace rather
    /// than from this view.
    #[inline]
    pub const fn detail(self) -> Option<&'a StepDetail> {
        self.detail
    }

    /// Returns the step as a detailed step, if it was recorded in full.
    #[inline]
    pub const fn detailed(self) -> Option<DetailedStepRef<'a>> {
        match self.detail {
            Some(detail) => Some(DetailedStepRef {
                pc: self.pc,
                op: self.op,
                step_index: self.step_index,
                detail,
            }),
            None => None,
        }
    }

    /// Returns true if the step is a call-like operation: `CALL`, `CALLCODE`, `DELEGATECALL`,
    /// `STATICCALL`, `CREATE` or `CREATE2`.
    #[inline]
    pub const fn is_call_like_op(self) -> bool {
        is_call_like_op(self.op)
    }

    /// Copies the step out of the store, cloning its detail record and the snapshots the record
    /// owns.
    pub fn into_owned(self) -> CallTraceStep {
        CallTraceStep { pc: self.pc, op: self.op, detail: self.detail.cloned() }
    }
}

/// A borrowed view of one step that was recorded in full. Dereferences to its [`StepDetail`], so
/// the record's fields can be read directly; [`Self::pc`], [`Self::op`] and [`Self::step_index`]
/// are the view's own copies.
///
/// Field access through the dereference borrows the view, not the trace. When a borrow must
/// outlive the view — for instance to collect references from several steps — use the accessors
/// that return `'a` references, such as [`Self::storage_change`].
#[derive(Clone, Copy, Debug)]
pub struct DetailedStepRef<'a> {
    /// Program counter before step execution
    pub pc: usize,
    /// Opcode to be executed
    pub op: OpCode,
    /// Index of the step in the call's base track.
    pub step_index: usize,
    detail: &'a StepDetail,
}

impl<'a> DetailedStepRef<'a> {
    /// Returns the step's detail record, borrowed from the trace rather than from this view.
    #[inline]
    pub const fn detail(self) -> &'a StepDetail {
        self.detail
    }

    /// Returns the step's storage change, if any, borrowed from the trace rather than from this
    /// view.
    #[inline]
    pub fn storage_change(self) -> Option<&'a StorageChange> {
        self.detail.storage_change.as_deref()
    }

    /// Returns the step's decoded data, if any, borrowed from the trace rather than from this
    /// view.
    #[inline]
    pub fn decoded(self) -> Option<&'a DecodedTraceStep> {
        self.detail.decoded.as_deref()
    }

    /// Returns true if the step is a STOP opcode
    #[inline]
    pub(crate) const fn is_stop(self) -> bool {
        matches!(self.op.get(), opcode::STOP)
    }

    /// Returns true if the step is a call-like operation: `CALL`, `CALLCODE`, `DELEGATECALL`,
    /// `STATICCALL`, `CREATE` or `CREATE2`.
    #[inline]
    pub const fn is_call_like_op(self) -> bool {
        is_call_like_op(self.op)
    }

    /// Converts this step into a geth [StructLog]
    ///
    /// This sets memory and stack capture based on the `opts` parameter.
    pub(crate) fn convert_to_geth_struct_log(
        &self,
        opts: &GethDefaultTracingOptions,
        depth: u64,
    ) -> StructLog {
        #[allow(clippy::needless_update)]
        StructLog {
            depth,
            error: self.as_error(),
            gas: self.gas_remaining,
            gas_cost: self.gas_cost,
            op: if self.op.is_valid() { self.op.as_str().into() } else { "Unknown".into() },
            pc: self.pc as u64,
            refund_counter: Some(self.gas_refund_counter),
            stack: if opts.is_stack_enabled() {
                self.stack.as_ref().map(|stack| stack.to_vec())
            } else {
                None
            },
            memory: if opts.is_memory_enabled() {
                self.memory.as_ref().map(RecordedMemory::memory_chunks)
            } else {
                None
            },

            // Filled from external storage.
            storage: None,
            // Filled from `CallTraceNode`.
            return_data: None,

            // This is always `None` in the RPC response.
            memory_size: None,
            ..Default::default()
        }
    }
}

impl core::ops::Deref for DetailedStepRef<'_> {
    type Target = StepDetail;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.detail
    }
}

/// Returns true if `op` is a call-like operation: `CALL`, `CALLCODE`, `DELEGATECALL`,
/// `STATICCALL`, `CREATE` or `CREATE2`.
#[inline]
pub(crate) const fn is_call_like_op(op: OpCode) -> bool {
    matches!(
        op.get(),
        opcode::CALL
            | opcode::DELEGATECALL
            | opcode::STATICCALL
            | opcode::CREATE
            | opcode::CALLCODE
            | opcode::CREATE2
    )
}

/// Represents the source of a storage change - e.g., whether it came
/// from an SSTORE or SLOAD instruction.
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StorageChangeReason {
    /// SLOAD opcode
    SLOAD,
    /// SSTORE opcode
    SSTORE,
}

/// Represents a storage change during execution.
///
/// This maps to evm internals:
/// [JournalEntry::StorageChanged](revm::JournalEntry::StorageChanged)
///
/// It is used to track both storage change and warm load of a storage slot. For warm load in regard
/// to EIP-2929 AccessList had_value will be None.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StorageChange {
    /// key of the storage slot
    pub key: U256,
    /// Current value of the storage slot
    pub value: U256,
    /// The previous value of the storage slot, if any
    pub had_value: Option<U256>,
    /// How this storage was accessed
    pub reason: StorageChangeReason,
}

/// Represents the memory captured during execution
///
/// This is a wrapper around the [SharedMemory](revm::interpreter::SharedMemory) context memory.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RecordedMemory(pub(crate) Bytes);

impl RecordedMemory {
    #[inline]
    pub(crate) fn new(mem: &[u8]) -> Self {
        if mem.is_empty() {
            return Self(Bytes::new());
        }

        Self(Bytes::copy_from_slice(mem))
    }

    /// Returns the memory as a byte slice
    #[inline]
    pub fn as_bytes(&self) -> &Bytes {
        &self.0
    }

    /// Returns the memory as a byte vector
    #[inline]
    pub fn into_bytes(self) -> Bytes {
        self.0
    }

    /// Returns the size of the memory.
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns whether the memory is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Formats memory data into a list of 32-byte hex-encoded chunks.
    ///
    /// See: <https://github.com/ethereum/go-ethereum/blob/366d2169fbc0e0f803b68c042b77b6b480836dbc/eth/tracers/logger/logger.go#L450-L452>
    #[inline]
    pub fn memory_chunks(&self) -> Vec<String> {
        convert_memory(self.as_bytes())
    }
}

impl AsRef<[u8]> for RecordedMemory {
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

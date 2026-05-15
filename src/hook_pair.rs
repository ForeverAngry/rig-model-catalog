//! [`HookPair`] — sequentially compose two [`PromptHook`] implementations.
//!
//! `rig-core` ships `PromptHook` with no built-in composition: a single
//! `.with_hook(h)` call replaces whichever hook was in scope. `HookPair`
//! is the minimum adapter needed to attach more than one hook to a
//! single prompt request, which is the normal case once you want both
//! ambient telemetry (`MetaHook`) and behavioural hooks (e.g.
//! `MemvidPersistHook`).
//!
//! Semantics: the `first` hook runs before the `second` for every
//! callback. If the `first` hook returns a non-`Continue` action
//! (`Terminate` for [`HookAction`], or `Skip` / `Terminate` for
//! [`ToolCallHookAction`]), the `second` hook is **not** called and the
//! `first`'s action is returned verbatim. This matches the short-circuit
//! semantics rig-core already applies between hook calls and the
//! agentic loop.
//!
//! ```no_run
//! # #[cfg(feature = "rig-hook")]
//! # fn doc() {
//! use rig_model_meta::{HookPair, MetaHook};
//!
//! // Pair an ambient telemetry hook with whatever behavioural hook you
//! // already had attached.
//! # let telemetry = MetaHook::unresolved("ollama", "llama3.2:3b");
//! # let persist = (); // any other `PromptHook` impl
//! let combined = HookPair::new(telemetry, persist);
//! // `combined` itself implements `PromptHook<M>` and can be passed to
//! // `.with_hook(combined)`.
//! # let _ = combined;
//! # }
//! ```

use rig_core::agent::{HookAction, PromptHook, ToolCallHookAction};
use rig_core::completion::CompletionModel;
use rig_core::message::Message;

/// Run two [`PromptHook`]s in sequence (`first` then `second`),
/// short-circuiting on any non-`Continue` action from `first`.
#[derive(Debug, Clone)]
pub struct HookPair<A, B> {
    first: A,
    second: B,
}

impl<A, B> HookPair<A, B> {
    /// Build a pair that runs `first` before `second` on every callback.
    pub fn new(first: A, second: B) -> Self {
        Self { first, second }
    }

    /// Chain a third hook on the end. `HookPair::new(a, b).then(c)` runs
    /// `a -> b -> c`.
    pub fn then<C>(self, third: C) -> HookPair<Self, C> {
        HookPair::new(self, third)
    }

    /// Borrow the first hook.
    pub fn first(&self) -> &A {
        &self.first
    }

    /// Borrow the second hook.
    pub fn second(&self) -> &B {
        &self.second
    }
}

impl<A, B, M> PromptHook<M> for HookPair<A, B>
where
    M: CompletionModel,
    A: PromptHook<M>,
    B: PromptHook<M>,
{
    async fn on_completion_call(&self, prompt: &Message, history: &[Message]) -> HookAction {
        match self.first.on_completion_call(prompt, history).await {
            HookAction::Continue => self.second.on_completion_call(prompt, history).await,
            terminate => terminate,
        }
    }

    async fn on_completion_response(
        &self,
        prompt: &Message,
        response: &rig_core::completion::CompletionResponse<M::Response>,
    ) -> HookAction {
        match self.first.on_completion_response(prompt, response).await {
            HookAction::Continue => self.second.on_completion_response(prompt, response).await,
            terminate => terminate,
        }
    }

    async fn on_tool_call(
        &self,
        tool_name: &str,
        tool_call_id: Option<String>,
        internal_call_id: &str,
        args: &str,
    ) -> ToolCallHookAction {
        match self
            .first
            .on_tool_call(tool_name, tool_call_id.clone(), internal_call_id, args)
            .await
        {
            ToolCallHookAction::Continue => {
                self.second
                    .on_tool_call(tool_name, tool_call_id, internal_call_id, args)
                    .await
            }
            other => other,
        }
    }

    async fn on_tool_result(
        &self,
        tool_name: &str,
        tool_call_id: Option<String>,
        internal_call_id: &str,
        args: &str,
        result: &str,
    ) -> HookAction {
        match self
            .first
            .on_tool_result(
                tool_name,
                tool_call_id.clone(),
                internal_call_id,
                args,
                result,
            )
            .await
        {
            HookAction::Continue => {
                self.second
                    .on_tool_result(tool_name, tool_call_id, internal_call_id, args, result)
                    .await
            }
            terminate => terminate,
        }
    }

    async fn on_text_delta(&self, text_delta: &str, aggregated_text: &str) -> HookAction {
        match self.first.on_text_delta(text_delta, aggregated_text).await {
            HookAction::Continue => self.second.on_text_delta(text_delta, aggregated_text).await,
            terminate => terminate,
        }
    }

    async fn on_tool_call_delta(
        &self,
        tool_call_id: &str,
        internal_call_id: &str,
        tool_name: Option<&str>,
        tool_call_delta: &str,
    ) -> HookAction {
        match self
            .first
            .on_tool_call_delta(tool_call_id, internal_call_id, tool_name, tool_call_delta)
            .await
        {
            HookAction::Continue => {
                self.second
                    .on_tool_call_delta(tool_call_id, internal_call_id, tool_name, tool_call_delta)
                    .await
            }
            terminate => terminate,
        }
    }

    async fn on_stream_completion_response_finish(
        &self,
        prompt: &Message,
        response: &<M as CompletionModel>::StreamingResponse,
    ) -> HookAction {
        match self
            .first
            .on_stream_completion_response_finish(prompt, response)
            .await
        {
            HookAction::Continue => {
                self.second
                    .on_stream_completion_response_finish(prompt, response)
                    .await
            }
            terminate => terminate,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn hook_pair_constructs_and_chains() {
        let pair = HookPair::new((), ());
        let chained = pair.then(());
        // Smoke check: borrows return the wrapped halves without
        // panicking.
        let _ = chained.first();
        let _ = chained.second();
    }
}

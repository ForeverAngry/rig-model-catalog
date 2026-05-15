//! [`MetaHook`] — a [`rig_core::agent::PromptHook`] that stamps model
//! metadata and token-usage telemetry onto `tracing` spans.
//!
//! `MetaHook` is the "ambient telemetry" answer for any Rig agent:
//!
//! - On `on_completion_call`, it emits a `tracing` event carrying the
//!   model's provider, id, and `context_window` (when known).
//! - On `on_completion_response`, it emits an event with the per-turn
//!   `Usage` (`input_tokens`, `output_tokens`, `total_tokens`) plus a
//!   computed `gen_ai.usage.context_used_pct` that joins the response's
//!   `input_tokens` against the resolved context window.
//!
//! The hook is intentionally **observation-only**: it always returns
//! `HookAction::cont()`. Pair with [`crate::Cache`] if you want a single
//! upstream probe call amortised across many `MetaHook` instances.
//!
//! ```no_run
//! # #[cfg(all(feature = "rig-hook", feature = "ollama"))]
//! # async fn run() -> anyhow::Result<()> {
//! use rig_model_meta::{MetaHook, OllamaProbe};
//!
//! let probe = OllamaProbe::live("http://localhost:11434");
//! let hook = MetaHook::resolve(&probe, "ollama", "llama3.2:3b").await?;
//! // `hook` now implements `rig_core::agent::PromptHook<M>` for any
//! // `CompletionModel M`. Pass it to `agent.prompt(...).with_hook(hook)`.
//! # let _ = hook;
//! # Ok(())
//! # }
//! ```

use std::future::Future;

use rig_core::agent::{HookAction, PromptHook};
use rig_core::completion::{CompletionModel, CompletionResponse};
use rig_core::message::Message;
use rig_core::wasm_compat::WasmCompatSend;

use crate::{ModelDescriptor, ModelMetaProbe, ProbeError, ProviderId};

/// Observation-only [`PromptHook`] that stamps model metadata + token
/// usage on `tracing` spans.
///
/// Construct eagerly with [`MetaHook::resolve`] (probes once at build
/// time) or lazily with [`MetaHook::unresolved`] (skips probing
/// entirely; useful when descriptor data isn't required yet).
#[derive(Debug, Clone)]
pub struct MetaHook {
    provider: ProviderId,
    model: String,
    descriptor: Option<ModelDescriptor>,
}

impl MetaHook {
    /// Probe `probe` for `(provider, model)` and store the resolved
    /// descriptor inside the hook. Returns `Ok` even if the probe
    /// returned `None` — telemetry will simply omit context-window data
    /// for that case.
    pub async fn resolve<P>(
        probe: &P,
        provider: impl Into<ProviderId>,
        model: impl Into<String>,
    ) -> Result<Self, ProbeError>
    where
        P: ModelMetaProbe + ?Sized,
    {
        let model = model.into();
        let descriptor = probe.describe(&model).await?;
        Ok(Self {
            provider: provider.into(),
            model,
            descriptor,
        })
    }

    /// Construct a hook without probing anything. Telemetry events will
    /// still fire but the `context_window` and `context_used_pct` fields
    /// will be absent.
    pub fn unresolved(provider: impl Into<ProviderId>, model: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            descriptor: None,
        }
    }

    /// Construct a hook from an already-resolved descriptor — useful in
    /// tests or when descriptors come from a non-probe source.
    pub fn from_descriptor(
        provider: impl Into<ProviderId>,
        model: impl Into<String>,
        descriptor: Option<ModelDescriptor>,
    ) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            descriptor,
        }
    }

    /// Borrow the cached descriptor, if any.
    pub fn descriptor(&self) -> Option<&ModelDescriptor> {
        self.descriptor.as_ref()
    }

    /// Provider id this hook was constructed for.
    pub fn provider(&self) -> &ProviderId {
        &self.provider
    }

    /// Model id this hook was constructed for.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Compute `input_tokens / context_window * 100`, when the window is
    /// known and non-zero.
    fn context_used_pct(&self, input_tokens: u64) -> Option<f64> {
        self.descriptor
            .as_ref()
            .and_then(|d| d.context_window)
            .and_then(|w| {
                if w == 0 {
                    None
                } else {
                    Some(input_tokens as f64 / w as f64 * 100.0)
                }
            })
    }
}

impl<M> PromptHook<M> for MetaHook
where
    M: CompletionModel,
{
    fn on_completion_call(
        &self,
        _prompt: &Message,
        _history: &[Message],
    ) -> impl Future<Output = HookAction> + WasmCompatSend {
        let window = self.descriptor.as_ref().and_then(|d| d.context_window);
        tracing::info!(
            target: "rig_model_meta::hook",
            gen_ai_system = %self.provider,
            gen_ai_request_model = %self.model,
            gen_ai_model_context_window = window,
            "completion call start",
        );
        async { HookAction::cont() }
    }

    fn on_completion_response(
        &self,
        _prompt: &Message,
        response: &CompletionResponse<M::Response>,
    ) -> impl Future<Output = HookAction> + WasmCompatSend {
        let usage = response.usage;
        let window = self.descriptor.as_ref().and_then(|d| d.context_window);
        let pct = self.context_used_pct(usage.input_tokens);
        tracing::info!(
            target: "rig_model_meta::hook",
            gen_ai_system = %self.provider,
            gen_ai_response_model = %self.model,
            gen_ai_usage_input_tokens = usage.input_tokens,
            gen_ai_usage_output_tokens = usage.output_tokens,
            gen_ai_usage_total_tokens = usage.total_tokens,
            gen_ai_model_context_window = window,
            gen_ai_usage_context_used_pct = pct,
            "completion call complete",
        );
        async { HookAction::cont() }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::{ModelDescriptor, StubProbe};

    fn descriptor() -> ModelDescriptor {
        ModelDescriptor::builder("ollama", "llama3.2:3b")
            .context_window(131_072)
            .build()
    }

    #[tokio::test]
    async fn resolve_caches_descriptor_from_probe() {
        let probe = StubProbe::new([("llama3.2:3b", descriptor())]);
        let hook = MetaHook::resolve(&probe, "ollama", "llama3.2:3b")
            .await
            .unwrap();
        assert_eq!(hook.provider().as_str(), "ollama");
        assert_eq!(hook.model(), "llama3.2:3b");
        assert_eq!(hook.descriptor().unwrap().context_window, Some(131_072));
    }

    #[tokio::test]
    async fn resolve_tolerates_unknown_model() {
        let probe = StubProbe::default();
        let hook = MetaHook::resolve(&probe, "ollama", "unknown")
            .await
            .unwrap();
        assert!(hook.descriptor().is_none());
    }

    #[test]
    fn context_used_pct_computes_against_window() {
        let hook = MetaHook::from_descriptor("ollama", "llama3.2:3b", Some(descriptor()));
        let pct = hook.context_used_pct(65_536).unwrap();
        assert!((pct - 50.0).abs() < 1e-9);
    }

    #[test]
    fn context_used_pct_is_none_when_window_unknown() {
        let hook = MetaHook::unresolved("openai", "gpt-4o");
        assert!(hook.context_used_pct(1000).is_none());
    }

    #[test]
    fn context_used_pct_handles_zero_window() {
        let desc = ModelDescriptor::builder("p", "m").context_window(0).build();
        let hook = MetaHook::from_descriptor("p", "m", Some(desc));
        assert!(hook.context_used_pct(100).is_none());
    }
}

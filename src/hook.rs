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
//! let hook = MetaHook::resolve(&probe, "ollama", "qwen3.5:9b").await?;
//! // `hook` now implements `rig_core::agent::PromptHook<M>` for any
//! // `CompletionModel M`. Pass it to `agent.prompt(...).with_hook(hook)`.
//! # let _ = hook;
//! # Ok(())
//! # }
//! ```

use std::future::Future;
#[cfg(feature = "observe")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "observe")]
use std::time::{SystemTime, UNIX_EPOCH};

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
    #[cfg(feature = "observe")]
    observe_conversation_id: String,
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
            #[cfg(feature = "observe")]
            observe_conversation_id: "default".into(),
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
            #[cfg(feature = "observe")]
            observe_conversation_id: "default".into(),
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
            #[cfg(feature = "observe")]
            observe_conversation_id: "default".into(),
        }
    }

    /// Set the `conversation_id` stamped on `rig_tap` prompt events
    /// emitted when the `observe` feature is enabled.
    ///
    /// The underlying Rig `PromptHook` does not carry request context, so the
    /// default value is `"default"`. Construct one hook per conversation or
    /// agent instance when consumers need stronger correlation.
    #[cfg(feature = "observe")]
    #[must_use]
    pub fn with_observe_conversation_id(mut self, conversation_id: impl Into<String>) -> Self {
        self.observe_conversation_id = conversation_id.into();
        self
    }

    /// Borrow the `conversation_id` stamped on `rig_tap` prompt events
    /// when the `observe` feature is enabled.
    #[cfg(feature = "observe")]
    pub fn observe_conversation_id(&self) -> &str {
        &self.observe_conversation_id
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
        #[cfg(feature = "observe")]
        emit_observe_prompt_started(
            &self.observe_conversation_id,
            &self.model,
            _history.len().saturating_add(1),
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
        #[cfg(feature = "observe")]
        emit_observe_prompt_completed(
            &self.observe_conversation_id,
            &self.model,
            positive(usage.input_tokens),
            positive(usage.output_tokens),
        );
        async { HookAction::cont() }
    }
}

#[cfg(feature = "observe")]
static OBSERVE_TICK: AtomicU64 = AtomicU64::new(1);

#[cfg(feature = "observe")]
fn emit_observe_prompt_started(conversation_id: &str, model: &str, messages_in: usize) {
    let mut event = observe_envelope(conversation_id, "prompt.started");
    event.insert("model".into(), serde_json::Value::String(model.into()));
    event.insert("messages_in".into(), serde_json::json!(messages_in));
    emit_observe_event(event);
}

#[cfg(feature = "observe")]
fn emit_observe_prompt_completed(
    conversation_id: &str,
    model: &str,
    tokens_in: Option<u64>,
    tokens_out: Option<u64>,
) {
    let mut event = observe_envelope(conversation_id, "prompt.completed");
    event.insert("model".into(), serde_json::Value::String(model.into()));
    if let Some(tokens_in) = tokens_in {
        event.insert("tokens_in".into(), serde_json::json!(tokens_in));
    }
    if let Some(tokens_out) = tokens_out {
        event.insert("tokens_out".into(), serde_json::json!(tokens_out));
    }
    emit_observe_event(event);
}

#[cfg(feature = "observe")]
fn observe_envelope(
    conversation_id: &str,
    kind: &str,
) -> serde_json::Map<String, serde_json::Value> {
    let mut event = serde_json::Map::new();
    event.insert("version".into(), serde_json::json!(1));
    event.insert("occurred_at_millis".into(), serde_json::json!(now_millis()));
    event.insert(
        "tick".into(),
        serde_json::json!(OBSERVE_TICK.fetch_add(1, Ordering::Relaxed)),
    );
    event.insert(
        "conversation_id".into(),
        serde_json::Value::String(conversation_id.into()),
    );
    event.insert("kind".into(), serde_json::Value::String(kind.into()));
    event
}

#[cfg(feature = "observe")]
fn emit_observe_event(event: serde_json::Map<String, serde_json::Value>) {
    if let Ok(json) = serde_json::to_string(&event) {
        tracing::info!(target: "rig_tap", event = %json);
    }
}

#[cfg(feature = "observe")]
fn now_millis() -> u64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => u64::try_from(duration.as_millis()).unwrap_or(u64::MAX),
        Err(_) => 0,
    }
}

#[cfg(feature = "observe")]
fn positive(value: u64) -> Option<u64> {
    if value == 0 { None } else { Some(value) }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::{ModelDescriptor, StubProbe};

    fn descriptor() -> ModelDescriptor {
        ModelDescriptor::builder("ollama", "qwen3.5:9b")
            .context_window(131_072)
            .build()
    }

    #[tokio::test]
    async fn resolve_caches_descriptor_from_probe() {
        let probe = StubProbe::new([("qwen3.5:9b", descriptor())]);
        let hook = MetaHook::resolve(&probe, "ollama", "qwen3.5:9b")
            .await
            .unwrap();
        assert_eq!(hook.provider().as_str(), "ollama");
        assert_eq!(hook.model(), "qwen3.5:9b");
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
        let hook = MetaHook::from_descriptor("ollama", "qwen3.5:9b", Some(descriptor()));
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

    #[cfg(feature = "observe")]
    #[test]
    fn observe_envelope_matches_rig_tap_shape() {
        let event = observe_envelope("thread-1", "prompt.started");
        assert_eq!(event.get("version").unwrap(), &serde_json::json!(1));
        assert_eq!(
            event.get("conversation_id").unwrap(),
            &serde_json::json!("thread-1")
        );
        assert_eq!(
            event.get("kind").unwrap(),
            &serde_json::json!("prompt.started")
        );
        assert!(event.get("occurred_at_millis").is_some());
        assert!(event.get("tick").is_some());
    }

    #[cfg(feature = "observe")]
    #[test]
    fn observe_conversation_id_is_configurable() {
        let hook =
            MetaHook::unresolved("ollama", "qwen3.5:9b").with_observe_conversation_id("thread-42");
        assert_eq!(hook.observe_conversation_id(), "thread-42");
    }
}

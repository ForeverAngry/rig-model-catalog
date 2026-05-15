//! Runtime model state — what a provider has actually loaded *right now*,
//! as opposed to the declared metadata returned by
//! [`crate::ModelMetaProbe::describe`].
//!
//! The motivating footgun: Ollama's `/api/show` reports a manifest
//! `context_length` of e.g. 128k, but the daemon will silently load the
//! model with whatever `num_ctx` the running configuration has set —
//! 2048 by default. Agents then truncate input at 2048 tokens with no
//! visible warning. [`RuntimeDescriptor`] surfaces the *currently loaded*
//! context window alongside the manifest one so callers can compare.
//!
//! Probes that can answer "is this model loaded, and with what
//! settings?" return [`RuntimeDescriptor`] from a dedicated method
//! (e.g. [`crate::OllamaProbe::runtime`]); probes against providers
//! without a runtime-introspection endpoint simply don't grow that
//! method.

use serde::{Deserialize, Serialize};

use crate::ProviderId;

/// Snapshot of a provider's currently loaded model state.
///
/// `#[non_exhaustive]` so adjacent runtime fields (GPU memory split,
/// keep-alive deadline, currently-pinned LoRAs, etc.) can be added
/// without a breaking change.
///
/// `expires_at` is kept as a free-form `String` (provider-supplied,
/// usually an RFC 3339 timestamp) so this crate stays clear of a date /
/// time-crate choice — callers parse on demand.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RuntimeDescriptor {
    /// Provider hosting the model.
    pub provider: ProviderId,
    /// Provider-specific model identifier.
    pub model: String,
    /// Currently loaded context window in tokens. May differ from the
    /// manifest value reported by `describe()`.
    pub effective_context_window: Option<u64>,
    /// VRAM occupied by the loaded weights, in bytes (when reported).
    pub size_vram_bytes: Option<u64>,
    /// Provider-supplied keep-alive deadline, raw string.
    pub expires_at: Option<String>,
    /// Provider-specific passthrough for fields we don't model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<serde_json::Value>,
}

impl RuntimeDescriptor {
    /// Minimal constructor; every detail field defaults to `None`.
    pub fn new(provider: impl Into<ProviderId>, model: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            effective_context_window: None,
            size_vram_bytes: None,
            expires_at: None,
            raw: None,
        }
    }
}

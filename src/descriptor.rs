//! Plain-data types describing a model: who serves it, how big its context
//! window is, what it can do, and how it was built.
//!
//! [`ModelDescriptor`] is the single payload every [`crate::ModelMetaProbe`]
//! returns. Every field is `Option`-wrapped or set-typed because providers
//! disagree about what is knowable: Ollama exposes quantization and
//! capabilities; OpenAI exposes neither. The descriptor is
//! `#[non_exhaustive]` so adjacent features (pricing, deprecation,
//! knowledge-cutoff dates) can extend it without a breaking change.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

/// Stable identifier for a model provider.
///
/// Stored as a `String` newtype rather than an enum so adding a new provider
/// is never a breaking change. Conventional values match the lowercase Rig
/// provider crate suffix: `ollama`, `openai`, `anthropic`, `gemini`,
/// `bedrock`, `llama_cpp`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ProviderId(pub String);

impl ProviderId {
    /// Construct a provider id from any string-like value.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Borrow the inner identifier.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for ProviderId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl From<String> for ProviderId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl std::fmt::Display for ProviderId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Coarse capability flags advertised by a model.
///
/// These map 1-to-1 against Ollama's `capabilities` array and against the
/// columns in OpenAI's and Anthropic's public model tables. Probes set the
/// flags they can verify; absent flags mean "unknown", not "unsupported".
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Capability {
    /// Standard text completion / chat.
    Completion,
    /// Function / tool calling.
    Tools,
    /// Multimodal image input.
    Vision,
    /// Embedding generation.
    Embedding,
    /// Structured / JSON-schema-constrained output.
    StructuredOutput,
    /// Explicit step-by-step "thinking" mode (e.g. o-series, qwen3-thinking).
    Thinking,
    /// Image generation output.
    ImageGen,
}

/// Quantization scheme for a local model. `None` means unquantized
/// (typically `fp16` / `bf16`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
#[non_exhaustive]
pub enum Quantization {
    /// 4-bit K-quant medium (most common).
    Q4KM,
    /// 4-bit K-quant small.
    Q4KS,
    /// 5-bit K-quant medium.
    Q5KM,
    /// 8-bit quantization.
    Q8_0,
    /// IEEE-754 binary16.
    Fp16,
    /// bfloat16.
    Bf16,
    /// Anything else — the provider's raw label is preserved.
    Other(String),
}

impl Quantization {
    /// Best-effort parse of a provider's quantization label (e.g. Ollama's
    /// `details.quantization_level` field).
    pub fn parse(label: &str) -> Self {
        match label.trim().to_ascii_uppercase().as_str() {
            "Q4_K_M" | "Q4KM" => Self::Q4KM,
            "Q4_K_S" | "Q4KS" => Self::Q4KS,
            "Q5_K_M" | "Q5KM" => Self::Q5KM,
            "Q8_0" | "Q8" => Self::Q8_0,
            "F16" | "FP16" => Self::Fp16,
            "BF16" => Self::Bf16,
            _ => Self::Other(label.to_string()),
        }
    }
}

/// Everything a probe can say about a model.
///
/// `#[non_exhaustive]` so adjacent crates can add fields (pricing,
/// knowledge-cutoff date, deprecation info) without a breaking change.
/// Construct one with [`ModelDescriptor::new`] or
/// [`ModelDescriptor::builder`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ModelDescriptor {
    /// Provider that serves this model.
    pub provider: ProviderId,
    /// Provider-specific model identifier (e.g. `gpt-4o`, `llama3.2:3b`).
    pub model: String,
    /// Maximum input context window in tokens, if known.
    pub context_window: Option<u64>,
    /// Maximum output / completion tokens the provider will emit, if known.
    pub max_output_tokens: Option<u64>,
    /// Coarse capability flags advertised by the model.
    pub capabilities: BTreeSet<Capability>,
    /// Architecture family (`llama`, `qwen2`, `mistral`, `claude`, ...).
    pub family: Option<String>,
    /// Parameter count in absolute units (so a 7B model is `7_000_000_000`).
    pub parameter_count: Option<u64>,
    /// Quantization scheme, if the model is locally quantized.
    pub quantization: Option<Quantization>,
    /// Provider-specific passthrough. Lets callers reach fields we don't
    /// model without forcing the descriptor to grow forever.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<serde_json::Value>,
}

impl ModelDescriptor {
    /// Construct a minimal descriptor for the given provider + model id.
    /// Every other field defaults to `None` / empty.
    pub fn new(provider: impl Into<ProviderId>, model: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            context_window: None,
            max_output_tokens: None,
            capabilities: BTreeSet::new(),
            family: None,
            parameter_count: None,
            quantization: None,
            raw: None,
        }
    }

    /// Start a builder for chained construction.
    pub fn builder(
        provider: impl Into<ProviderId>,
        model: impl Into<String>,
    ) -> ModelDescriptorBuilder {
        ModelDescriptorBuilder {
            inner: Self::new(provider, model),
        }
    }

    /// Fraction of the input context window consumed by `input_tokens`, if
    /// the window is known and non-zero.
    pub fn context_used_fraction(&self, input_tokens: u64) -> Option<f64> {
        match self.context_window {
            Some(window) if window > 0 => Some(input_tokens as f64 / window as f64),
            _ => None,
        }
    }

    /// Convenience: `true` if `cap` is set on this descriptor.
    pub fn has_capability(&self, cap: Capability) -> bool {
        self.capabilities.contains(&cap)
    }
}

/// Chained builder for [`ModelDescriptor`].
#[derive(Debug, Clone)]
pub struct ModelDescriptorBuilder {
    inner: ModelDescriptor,
}

impl ModelDescriptorBuilder {
    /// Set the context window in tokens.
    pub fn context_window(mut self, tokens: u64) -> Self {
        self.inner.context_window = Some(tokens);
        self
    }

    /// Set the max output tokens the provider will emit.
    pub fn max_output_tokens(mut self, tokens: u64) -> Self {
        self.inner.max_output_tokens = Some(tokens);
        self
    }

    /// Insert a single capability.
    pub fn capability(mut self, cap: Capability) -> Self {
        self.inner.capabilities.insert(cap);
        self
    }

    /// Replace the capability set wholesale.
    pub fn capabilities(mut self, caps: impl IntoIterator<Item = Capability>) -> Self {
        self.inner.capabilities = caps.into_iter().collect();
        self
    }

    /// Set the architecture family.
    pub fn family(mut self, family: impl Into<String>) -> Self {
        self.inner.family = Some(family.into());
        self
    }

    /// Set the parameter count in absolute units (`7B` → `7_000_000_000`).
    pub fn parameter_count(mut self, count: u64) -> Self {
        self.inner.parameter_count = Some(count);
        self
    }

    /// Set the quantization scheme.
    pub fn quantization(mut self, q: Quantization) -> Self {
        self.inner.quantization = Some(q);
        self
    }

    /// Attach provider-specific passthrough JSON.
    pub fn raw(mut self, raw: serde_json::Value) -> Self {
        self.inner.raw = Some(raw);
        self
    }

    /// Finalise the descriptor.
    pub fn build(self) -> ModelDescriptor {
        self.inner
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn quantization_parses_common_labels() {
        assert_eq!(Quantization::parse("Q4_K_M"), Quantization::Q4KM);
        assert_eq!(Quantization::parse("q4_k_s"), Quantization::Q4KS);
        assert_eq!(Quantization::parse("F16"), Quantization::Fp16);
        match Quantization::parse("Q3_K_S") {
            Quantization::Other(label) => assert_eq!(label, "Q3_K_S"),
            other => panic!("expected Other, got {other:?}"),
        }
    }

    #[test]
    fn builder_roundtrip() {
        let d = ModelDescriptor::builder("ollama", "llama3.2:3b")
            .context_window(128_000)
            .capability(Capability::Completion)
            .capability(Capability::Tools)
            .family("llama")
            .quantization(Quantization::Q4KM)
            .build();
        assert_eq!(d.context_window, Some(128_000));
        assert!(d.has_capability(Capability::Completion));
        assert!(d.has_capability(Capability::Tools));
        assert!(!d.has_capability(Capability::Vision));
        assert_eq!(d.context_used_fraction(64_000), Some(0.5));
    }

    #[test]
    fn context_used_fraction_handles_zero_and_unknown() {
        let mut d = ModelDescriptor::new("ollama", "x");
        assert_eq!(d.context_used_fraction(100), None);
        d.context_window = Some(0);
        assert_eq!(d.context_used_fraction(100), None);
        d.context_window = Some(1000);
        assert_eq!(d.context_used_fraction(250), Some(0.25));
    }

    #[test]
    fn descriptor_serde_roundtrip() {
        let d = ModelDescriptor::builder("openai", "gpt-4o")
            .context_window(128_000)
            .max_output_tokens(16_384)
            .capability(Capability::Completion)
            .capability(Capability::Tools)
            .capability(Capability::Vision)
            .build();
        let json = serde_json::to_string(&d).unwrap();
        let back: ModelDescriptor = serde_json::from_str(&json).unwrap();
        assert_eq!(d, back);
    }
}

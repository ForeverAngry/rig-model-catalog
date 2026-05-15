//! [`StaticProbe`] — curated in-memory catalog for providers that don't
//! expose a live metadata endpoint (OpenAI, Anthropic).
//!
//! The bundled tables are deliberately conservative: they cover headline
//! models with public, stable context-window numbers. Callers needing fresh
//! data should layer a live probe in front of [`StaticProbe`] via
//! [`crate::ChainedProbe`].

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::{Capability, ModelDescriptor, ModelMetaProbe, ProbeError, ProviderId};

const OPENAI_JSON: &str = include_str!("../data/openai.json");
const ANTHROPIC_JSON: &str = include_str!("../data/anthropic.json");

/// Catalog probe keyed on `(provider, model)`. Lookup is by model id alone
/// (the first matching provider wins) so callers don't need to know the
/// origin of a model to look it up.
#[derive(Debug, Clone, Default)]
pub struct StaticProbe {
    by_model: BTreeMap<String, ModelDescriptor>,
}

impl StaticProbe {
    /// Load the bundled OpenAI + Anthropic catalogs.
    pub fn builtin() -> Self {
        let mut probe = Self::default();
        probe.extend_from_json("openai", OPENAI_JSON);
        probe.extend_from_json("anthropic", ANTHROPIC_JSON);
        probe
    }

    /// Load only OpenAI's bundled catalog.
    pub fn builtin_openai() -> Self {
        let mut probe = Self::default();
        probe.extend_from_json("openai", OPENAI_JSON);
        probe
    }

    /// Load only Anthropic's bundled catalog.
    pub fn builtin_anthropic() -> Self {
        let mut probe = Self::default();
        probe.extend_from_json("anthropic", ANTHROPIC_JSON);
        probe
    }

    /// Merge a custom JSON catalog (same shape as the bundled data files)
    /// into this probe.
    pub fn extend_from_json_str(
        &mut self,
        provider: impl Into<String>,
        json: &str,
    ) -> Result<(), ProbeError> {
        let provider = provider.into();
        let entries: Vec<CatalogEntry> =
            serde_json::from_str(json).map_err(|e| ProbeError::Parse(e.to_string()))?;
        for entry in entries {
            let desc = entry.into_descriptor(ProviderId::new(provider.clone()));
            self.by_model.insert(desc.model.clone(), desc);
        }
        Ok(())
    }

    fn extend_from_json(&mut self, provider: &str, json: &str) {
        // Bundled JSON is validated by tests; failure here would be a
        // crate-build bug. We still avoid panicking — drop bad entries
        // silently and let the consumer notice via test failures.
        if let Ok(entries) = serde_json::from_str::<Vec<CatalogEntry>>(json) {
            for entry in entries {
                let desc = entry.into_descriptor(ProviderId::new(provider.to_string()));
                self.by_model.insert(desc.model.clone(), desc);
            }
        }
    }

    /// Insert a single descriptor.
    pub fn insert(&mut self, descriptor: ModelDescriptor) {
        self.by_model.insert(descriptor.model.clone(), descriptor);
    }

    /// Number of catalog entries.
    pub fn len(&self) -> usize {
        self.by_model.len()
    }

    /// `true` when the catalog is empty.
    pub fn is_empty(&self) -> bool {
        self.by_model.is_empty()
    }
}

impl ModelMetaProbe for StaticProbe {
    async fn describe(&self, model: &str) -> Result<Option<ModelDescriptor>, ProbeError> {
        Ok(self.by_model.get(model).cloned())
    }
}

#[derive(Debug, Deserialize)]
struct CatalogEntry {
    model: String,
    #[serde(default)]
    context_window: Option<u64>,
    #[serde(default)]
    max_output_tokens: Option<u64>,
    #[serde(default)]
    capabilities: Vec<Capability>,
    #[serde(default)]
    family: Option<String>,
}

impl CatalogEntry {
    fn into_descriptor(self, provider: ProviderId) -> ModelDescriptor {
        ModelDescriptor {
            provider,
            model: self.model,
            context_window: self.context_window,
            max_output_tokens: self.max_output_tokens,
            capabilities: self.capabilities.into_iter().collect(),
            family: self.family,
            parameter_count: None,
            quantization: None,
            raw: None,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn bundled_openai_parses() {
        let probe = StaticProbe::builtin_openai();
        assert!(!probe.is_empty(), "openai catalog should not be empty");
    }

    #[test]
    fn bundled_anthropic_parses() {
        let probe = StaticProbe::builtin_anthropic();
        assert!(!probe.is_empty(), "anthropic catalog should not be empty");
    }

    #[tokio::test]
    async fn looks_up_gpt_4o() {
        let probe = StaticProbe::builtin();
        let desc = probe.describe("gpt-4o").await.unwrap().unwrap();
        assert_eq!(desc.provider, ProviderId::new("openai"));
        assert_eq!(desc.context_window, Some(128_000));
    }

    #[tokio::test]
    async fn looks_up_claude_sonnet() {
        let probe = StaticProbe::builtin();
        let desc = probe
            .describe("claude-3-5-sonnet-latest")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(desc.provider, ProviderId::new("anthropic"));
        assert_eq!(desc.context_window, Some(200_000));
    }

    #[tokio::test]
    async fn unknown_model_is_none() {
        let probe = StaticProbe::builtin();
        assert!(probe.describe("not-a-model").await.unwrap().is_none());
    }
}

//! In-memory stub probe for tests, fixtures, and offline development.
//!
//! Construct from any iterator of `(model_id, descriptor)` pairs:
//!
//! ```
//! use rig_model_catalog::{ModelDescriptor, ModelMetaProbe, StubProbe};
//!
//! # async fn run() -> anyhow::Result<()> {
//! let probe = StubProbe::new([(
//!     "gpt-4o".to_string(),
//!     ModelDescriptor::builder("openai", "gpt-4o")
//!         .context_window(128_000)
//!         .build(),
//! )]);
//! let desc = probe.describe("gpt-4o").await?.expect("known model");
//! assert_eq!(desc.context_window, Some(128_000));
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;

use crate::{ModelDescriptor, ModelMetaProbe, ProbeError};

/// Deterministic in-memory probe. Useful for tests and as the canonical
/// example of how to implement [`ModelMetaProbe`].
#[derive(Debug, Clone, Default)]
pub struct StubProbe {
    table: HashMap<String, ModelDescriptor>,
}

impl StubProbe {
    /// Construct a stub from any iterable of `(model_id, descriptor)` pairs.
    pub fn new<I, K>(entries: I) -> Self
    where
        I: IntoIterator<Item = (K, ModelDescriptor)>,
        K: Into<String>,
    {
        Self {
            table: entries.into_iter().map(|(k, v)| (k.into(), v)).collect(),
        }
    }

    /// Add or replace an entry.
    pub fn insert(&mut self, model: impl Into<String>, descriptor: ModelDescriptor) {
        self.table.insert(model.into(), descriptor);
    }

    /// Number of known models.
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// `true` when the table is empty.
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }
}

impl ModelMetaProbe for StubProbe {
    async fn describe(&self, model: &str) -> Result<Option<ModelDescriptor>, ProbeError> {
        Ok(self.table.get(model).cloned())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::{Capability, ModelMetaProbe};

    fn fixture() -> StubProbe {
        StubProbe::new([(
            "gpt-4o",
            ModelDescriptor::builder("openai", "gpt-4o")
                .context_window(128_000)
                .capability(Capability::Completion)
                .build(),
        )])
    }

    #[tokio::test]
    async fn known_model_returns_descriptor() {
        let probe = fixture();
        let desc = probe.describe("gpt-4o").await.unwrap().unwrap();
        assert_eq!(desc.context_window, Some(128_000));
    }

    #[tokio::test]
    async fn unknown_model_returns_none_not_error() {
        let probe = fixture();
        let out = probe.describe("nope").await.unwrap();
        assert!(out.is_none());
    }
}

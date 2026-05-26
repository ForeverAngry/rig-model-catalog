//! [`ChainedProbe`] — try probes in order, return the first match.
//!
//! Typical use: a live Ollama probe for local models, falling back to a
//! static OpenAI / Anthropic catalog for cloud models the daemon doesn't
//! know about.
//!
//! ```
//! use rig_model_catalog::{
//!     ChainedProbe, ModelDescriptor, ModelMetaProbe, StubProbe,
//! };
//!
//! # async fn run() -> anyhow::Result<()> {
//! let primary = StubProbe::new([(
//!     "local-model",
//!     ModelDescriptor::builder("ollama", "local-model")
//!         .context_window(8192)
//!         .build(),
//! )]);
//! let fallback = StubProbe::new([(
//!     "gpt-4o",
//!     ModelDescriptor::builder("openai", "gpt-4o")
//!         .context_window(128_000)
//!         .build(),
//! )]);
//! let chained = ChainedProbe::new(primary, fallback);
//! assert_eq!(
//!     chained.describe("gpt-4o").await?.unwrap().context_window,
//!     Some(128_000),
//! );
//! # Ok(())
//! # }
//! ```

use crate::{ModelDescriptor, ModelMetaProbe, ProbeError};

/// Probe composer: ask `primary` first, then `fallback`.
///
/// Errors from `primary` are **not** suppressed — if the primary probe
/// reports a hard failure, the chain returns it rather than masking real
/// outages behind a happy fallback. Use [`ChainedProbe::tolerant`] to opt
/// into "swallow primary errors and try the fallback" behaviour.
#[derive(Debug, Clone)]
pub struct ChainedProbe<P, F> {
    primary: P,
    fallback: F,
    tolerant: bool,
}

impl<P, F> ChainedProbe<P, F> {
    /// Strict chain: primary errors propagate.
    pub fn new(primary: P, fallback: F) -> Self {
        Self {
            primary,
            fallback,
            tolerant: false,
        }
    }

    /// Tolerant chain: primary errors are logged via `tracing::warn` and
    /// the fallback is consulted anyway.
    pub fn tolerant(primary: P, fallback: F) -> Self {
        Self {
            primary,
            fallback,
            tolerant: true,
        }
    }
}

impl<P, F> ModelMetaProbe for ChainedProbe<P, F>
where
    P: ModelMetaProbe,
    F: ModelMetaProbe,
{
    async fn describe(&self, model: &str) -> Result<Option<ModelDescriptor>, ProbeError> {
        match self.primary.describe(model).await {
            Ok(Some(desc)) => Ok(Some(desc)),
            Ok(None) => self.fallback.describe(model).await,
            Err(err) if self.tolerant => {
                tracing::warn!(
                    target: "rig_model_catalog::chained",
                    error = %err,
                    model = model,
                    "primary probe failed; consulting fallback"
                );
                self.fallback.describe(model).await
            }
            Err(err) => Err(err),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::{ModelDescriptor, StubProbe};

    fn primary() -> StubProbe {
        StubProbe::new([(
            "local",
            ModelDescriptor::builder("ollama", "local")
                .context_window(8192)
                .build(),
        )])
    }

    fn fallback() -> StubProbe {
        StubProbe::new([(
            "cloud",
            ModelDescriptor::builder("openai", "cloud")
                .context_window(128_000)
                .build(),
        )])
    }

    #[tokio::test]
    async fn primary_match_wins() {
        let chain = ChainedProbe::new(primary(), fallback());
        let desc = chain.describe("local").await.unwrap().unwrap();
        assert_eq!(desc.context_window, Some(8192));
    }

    #[tokio::test]
    async fn falls_back_when_primary_returns_none() {
        let chain = ChainedProbe::new(primary(), fallback());
        let desc = chain.describe("cloud").await.unwrap().unwrap();
        assert_eq!(desc.context_window, Some(128_000));
    }

    #[tokio::test]
    async fn both_miss_returns_none() {
        let chain = ChainedProbe::new(primary(), fallback());
        assert!(chain.describe("nope").await.unwrap().is_none());
    }

    #[derive(Clone)]
    struct AlwaysFails;

    impl ModelMetaProbe for AlwaysFails {
        async fn describe(&self, _model: &str) -> Result<Option<ModelDescriptor>, ProbeError> {
            Err(ProbeError::Transport("nope".to_string()))
        }
    }

    #[tokio::test]
    async fn strict_chain_propagates_primary_error() {
        let chain = ChainedProbe::new(AlwaysFails, fallback());
        let err = chain.describe("cloud").await.unwrap_err();
        assert!(matches!(err, ProbeError::Transport(_)));
    }

    #[tokio::test]
    async fn tolerant_chain_consults_fallback_after_primary_error() {
        let chain = ChainedProbe::tolerant(AlwaysFails, fallback());
        let desc = chain.describe("cloud").await.unwrap().unwrap();
        assert_eq!(desc.context_window, Some(128_000));
    }
}

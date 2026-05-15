//! Object-safe adapter for [`ModelMetaProbe`].
//!
//! The base trait uses return-position-impl-trait (`async fn` in trait), which
//! is ergonomic but **not** `dyn`-compatible. [`ModelMetaProbeDyn`] mirrors
//! the same contract with a boxed future so callers can store probes in
//! `Box<dyn ModelMetaProbeDyn>` or `Arc<dyn ModelMetaProbeDyn>`.
//!
//! A blanket impl wires every static [`ModelMetaProbe`] into the dyn-flavour
//! automatically, and [`DynProbe`] adapts the other direction so a
//! `dyn ModelMetaProbeDyn` can stand in wherever a `ModelMetaProbe` bound is
//! required.
//!
//! ```
//! use std::sync::Arc;
//!
//! use rig_model_meta::{
//!     DynProbe, ModelDescriptor, ModelMetaProbe, ModelMetaProbeDyn, StubProbe,
//! };
//!
//! # async fn run() -> anyhow::Result<()> {
//! let probes: Vec<Arc<dyn ModelMetaProbeDyn>> = vec![Arc::new(StubProbe::new([(
//!     "gpt-4o",
//!     ModelDescriptor::builder("openai", "gpt-4o")
//!         .context_window(128_000)
//!         .build(),
//! )]))];
//! // Re-promote to the static trait for downstream consumers.
//! let adapter = DynProbe::new(Arc::clone(&probes[0]));
//! assert_eq!(
//!     adapter.describe("gpt-4o").await?.unwrap().context_window,
//!     Some(128_000),
//! );
//! # Ok(())
//! # }
//! ```

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::{ModelDescriptor, ModelMetaProbe, ProbeError};

/// Boxed-future type alias used throughout the dyn-trait surface.
pub type ProbeFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Option<ModelDescriptor>, ProbeError>> + Send + 'a>>;

/// Object-safe mirror of [`ModelMetaProbe`].
///
/// Every static [`ModelMetaProbe`] automatically implements this trait via
/// a blanket impl, so callers who need `Box<dyn ...>` storage pay nothing
/// extra. See [`DynProbe`] for the inverse adapter.
pub trait ModelMetaProbeDyn: Send + Sync {
    /// Look up metadata for `model`. Contract matches [`ModelMetaProbe`].
    fn describe_boxed<'a>(&'a self, model: &'a str) -> ProbeFuture<'a>;
}

impl<P> ModelMetaProbeDyn for P
where
    P: ModelMetaProbe + ?Sized,
{
    fn describe_boxed<'a>(&'a self, model: &'a str) -> ProbeFuture<'a> {
        Box::pin(self.describe(model))
    }
}

/// Adapter that lifts an erased [`ModelMetaProbeDyn`] back into a static
/// [`ModelMetaProbe`].
///
/// Useful when a downstream API requires the impl-trait flavour (e.g. for
/// [`crate::ChainedProbe`]) but the caller is holding the erased dyn form.
#[derive(Clone)]
pub struct DynProbe {
    inner: Arc<dyn ModelMetaProbeDyn>,
}

impl DynProbe {
    /// Wrap an `Arc<dyn ModelMetaProbeDyn>` so it can satisfy a
    /// `ModelMetaProbe` bound.
    pub fn new(inner: Arc<dyn ModelMetaProbeDyn>) -> Self {
        Self { inner }
    }
}

impl std::fmt::Debug for DynProbe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynProbe").finish_non_exhaustive()
    }
}

impl ModelMetaProbe for DynProbe {
    async fn describe(&self, model: &str) -> Result<Option<ModelDescriptor>, ProbeError> {
        self.inner.describe_boxed(model).await
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::{ChainedProbe, ModelDescriptor, StubProbe};

    fn make_probe() -> StubProbe {
        StubProbe::new([(
            "gpt-4o",
            ModelDescriptor::builder("openai", "gpt-4o")
                .context_window(128_000)
                .build(),
        )])
    }

    #[tokio::test]
    async fn blanket_impl_satisfies_dyn() {
        let probes: Vec<Box<dyn ModelMetaProbeDyn>> = vec![Box::new(make_probe())];
        let desc = probes[0].describe_boxed("gpt-4o").await.unwrap().unwrap();
        assert_eq!(desc.context_window, Some(128_000));
    }

    #[tokio::test]
    async fn dyn_probe_round_trip_through_static_trait() {
        let erased: Arc<dyn ModelMetaProbeDyn> = Arc::new(make_probe());
        let lifted = DynProbe::new(erased);
        let desc = lifted.describe("gpt-4o").await.unwrap().unwrap();
        assert_eq!(desc.context_window, Some(128_000));
    }

    #[tokio::test]
    async fn dyn_probe_composes_with_chained() {
        let primary: Arc<dyn ModelMetaProbeDyn> = Arc::new(StubProbe::default());
        let fallback: Arc<dyn ModelMetaProbeDyn> = Arc::new(make_probe());
        let chained = ChainedProbe::new(DynProbe::new(primary), DynProbe::new(fallback));
        let desc = chained.describe("gpt-4o").await.unwrap().unwrap();
        assert_eq!(desc.context_window, Some(128_000));
    }
}

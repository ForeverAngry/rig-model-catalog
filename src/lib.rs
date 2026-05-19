//! # rig-model-meta
//!
//! Provider-agnostic model metadata for the [Rig](https://crates.io/crates/rig-core)
//! ecosystem. This crate exposes a single trait — [`ModelMetaProbe`] — and a
//! plain-data [`ModelDescriptor`] type, so any consumer (eval harnesses,
//! evolution loops, kernel coordinators, demo apps) can ask "what is this
//! model's max context window? what capabilities does it advertise?" without
//! reinventing a per-provider probe each time.
//!
//! The crate is intentionally narrow:
//!
//! - The default build is **dependency-light** (no HTTP client, no provider
//!   SDKs). You get the trait, the descriptor, error types, [`StubProbe`],
//!   and [`ChainedProbe`].
//! - Live probes are **feature-gated**. Enable `ollama` to pull
//!   [`OllamaProbe`]; enable `static` for the bundled catalog of OpenAI /
//!   Anthropic models that have no live metadata endpoint.
//!
//! ## Quick start
//!
//! ```no_run
//! # #[cfg(feature = "ollama")]
//! # async fn run() -> anyhow::Result<()> {
//! use rig_model_meta::{ModelMetaProbe, OllamaProbe};
//!
//! let probe = OllamaProbe::live("http://localhost:11434");
//! if let Some(desc) = probe.describe("qwen3.5:9b").await? {
//!     println!("context window: {:?}", desc.context_window);
//!     println!("capabilities:   {:?}", desc.capabilities);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Composition
//!
//! Probes compose. [`ChainedProbe`] tries probes in order and returns the
//! first one to recognise the model — useful for blending a live Ollama
//! probe with a static OpenAI / Anthropic catalog:
//!
//! ```no_run
//! # #[cfg(all(feature = "ollama", feature = "static"))]
//! # async fn run() -> anyhow::Result<()> {
//! use rig_model_meta::{
//!     ChainedProbe, ModelMetaProbe, OllamaProbe, StaticProbe,
//! };
//!
//! let chained = ChainedProbe::new(
//!     OllamaProbe::live("http://localhost:11434"),
//!     StaticProbe::builtin(),
//! );
//! let _ = chained.describe("gpt-4o").await?;
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

mod cache;
mod chained;
mod descriptor;
mod dyn_probe;
mod error;
mod probe;
mod runtime;
mod stub;

#[cfg(feature = "ollama")]
mod ollama;

#[cfg(feature = "static")]
mod static_catalog;

#[cfg(feature = "rig-hook")]
mod hook;

#[cfg(feature = "rig-hook")]
mod hook_pair;

#[cfg(feature = "pricing")]
mod pricing;

pub use crate::cache::Cache;
pub use crate::chained::ChainedProbe;
pub use crate::descriptor::{Capability, ModelDescriptor, ProviderId, Quantization};
pub use crate::dyn_probe::{DynProbe, ModelMetaProbeDyn, ProbeFuture};
pub use crate::error::ProbeError;
pub use crate::probe::ModelMetaProbe;
pub use crate::runtime::RuntimeDescriptor;
pub use crate::stub::StubProbe;

#[cfg(feature = "ollama")]
pub use crate::ollama::OllamaProbe;

#[cfg(feature = "static")]
pub use crate::static_catalog::StaticProbe;

#[cfg(feature = "rig-hook")]
pub use crate::hook::MetaHook;

#[cfg(feature = "rig-hook")]
pub use crate::hook_pair::HookPair;

#[cfg(feature = "pricing")]
pub use crate::pricing::{ModelPrice, PricingTable};

/// Convenience re-exports for typical consumers.
pub mod prelude {
    pub use crate::{
        Cache, Capability, ChainedProbe, DynProbe, ModelDescriptor, ModelMetaProbe,
        ModelMetaProbeDyn, ProbeError, ProviderId, Quantization, RuntimeDescriptor, StubProbe,
    };

    #[cfg(feature = "ollama")]
    pub use crate::OllamaProbe;

    #[cfg(feature = "static")]
    pub use crate::StaticProbe;

    #[cfg(feature = "rig-hook")]
    pub use crate::MetaHook;

    #[cfg(feature = "rig-hook")]
    pub use crate::HookPair;

    #[cfg(feature = "pricing")]
    pub use crate::{ModelPrice, PricingTable};
}

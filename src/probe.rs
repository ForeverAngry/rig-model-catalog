//! The [`ModelMetaProbe`] trait — a one-method contract every backend
//! implements. Returns `Ok(None)` for unknown models so callers can chain
//! probes without paying error-handling tax.

use std::future::Future;

use crate::{ModelDescriptor, ProbeError};

/// One-method trait implemented by every metadata source (live HTTP probes,
/// static catalogs, fixtures).
///
/// ## Contract
///
/// - **`Ok(Some(desc))`** — the probe recognises `model` and returns what it
///   knows. Unknown sub-fields stay `None`; absent capabilities mean "I
///   can't confirm this," not "the model lacks it."
/// - **`Ok(None)`** — the probe does not recognise `model`. Callers using
///   [`crate::ChainedProbe`] continue to the next probe.
/// - **`Err(ProbeError)`** — the probe could not even attempt the lookup
///   (transport down, malformed response, auth rejection). Hard failures
///   only — never use `Err` to mean "unknown."
///
/// Implementations must be cheap to clone (typically `Arc`-wrap any shared
/// state internally) so probes can be composed into chains.
///
/// ## Example
///
/// ```
/// use rig_model_catalog::{ModelDescriptor, ModelMetaProbe, ProbeError, StubProbe};
///
/// # async fn run() -> Result<(), ProbeError> {
/// let probe = StubProbe::new([(
///     "gpt-4o".to_string(),
///     ModelDescriptor::builder("openai", "gpt-4o")
///         .context_window(128_000)
///         .build(),
/// )]);
/// assert_eq!(probe.describe("gpt-4o").await?.unwrap().context_window, Some(128_000));
/// assert!(probe.describe("unknown-model").await?.is_none());
/// # Ok(())
/// # }
/// ```
pub trait ModelMetaProbe: Send + Sync {
    /// Look up metadata for `model`. See trait docs for the `Result`
    /// contract.
    fn describe(
        &self,
        model: &str,
    ) -> impl Future<Output = Result<Option<ModelDescriptor>, ProbeError>> + Send;
}

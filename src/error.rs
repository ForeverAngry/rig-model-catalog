//! Typed error variants for probe failures.
//!
//! The probe trait returns `Result<Option<ModelDescriptor>, ProbeError>`:
//! `Ok(None)` means "this probe does not know that model" (the common case),
//! while `Err(ProbeError::...)` is reserved for hard failures (transport
//! down, malformed response, auth rejection).

use thiserror::Error;

/// Errors surfaced by [`crate::ModelMetaProbe`] implementations.
///
/// "Unknown model" is **not** an error — probes return `Ok(None)` for that.
/// `ProbeError` exists for cases where the probe could not even attempt the
/// lookup (network failure, malformed manifest, etc.).
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ProbeError {
    /// The underlying transport (HTTP, IPC, file) failed before a response
    /// could be parsed.
    #[error("transport error: {0}")]
    Transport(String),

    /// A response was received but could not be parsed into a known shape.
    #[error("parse error: {0}")]
    Parse(String),

    /// The provider rejected the request (e.g. missing credentials, 401, 403).
    #[error("unauthorized")]
    Unauthorized,

    /// Catch-all for provider-specific failures that don't map to the above.
    #[error("{0}")]
    Other(String),
}

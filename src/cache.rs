//! [`Cache`] — TTL-bounded wrapper that memoises probe results.
//!
//! Provider manifests rarely change inside a single agent run, so the
//! second `describe("gpt-4o")` call should not pay the HTTP cost of the
//! first. [`Cache`] wraps any [`ModelMetaProbe`] and stores the most recent
//! result (success **and** "unknown") for a caller-supplied TTL.
//!
//! ```
//! use std::time::Duration;
//!
//! use rig_model_meta::{
//!     Cache, ModelDescriptor, ModelMetaProbe, StubProbe,
//! };
//!
//! # async fn run() -> anyhow::Result<()> {
//! let backing = StubProbe::new([(
//!     "gpt-4o",
//!     ModelDescriptor::builder("openai", "gpt-4o")
//!         .context_window(128_000)
//!         .build(),
//! )]);
//! let cache = Cache::new(backing, Duration::from_secs(60));
//! let _ = cache.describe("gpt-4o").await?;
//! let _ = cache.describe("gpt-4o").await?; // served from cache
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;

use crate::{ModelDescriptor, ModelMetaProbe, ProbeError};

/// TTL-bounded memoiser for [`ModelMetaProbe`] implementations.
///
/// Stores both `Some(...)` and `None` results so repeated lookups of an
/// unknown model do not re-hit the upstream backend. Errors are **not**
/// cached — a transient failure shouldn't poison subsequent lookups.
///
/// Clones share the same backing store via [`Arc`], so passing a `Cache`
/// into multiple chains is cheap.
#[derive(Debug, Clone)]
pub struct Cache<P> {
    inner: P,
    ttl: Duration,
    entries: Arc<RwLock<HashMap<String, Entry>>>,
}

#[derive(Debug, Clone)]
struct Entry {
    value: Option<ModelDescriptor>,
    expires_at: Instant,
}

impl<P> Cache<P> {
    /// Wrap `inner` with a `ttl`-bounded memoiser.
    pub fn new(inner: P, ttl: Duration) -> Self {
        Self {
            inner,
            ttl,
            entries: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Drop every cached entry. Useful when a provider config has changed
    /// mid-run and you want subsequent `describe` calls to re-probe.
    pub fn invalidate(&self) {
        self.entries.write().clear();
    }

    /// Drop a single cached entry.
    pub fn invalidate_model(&self, model: &str) {
        self.entries.write().remove(model);
    }

    /// Number of live (non-expired) entries currently in the cache.
    pub fn len(&self) -> usize {
        let now = Instant::now();
        self.entries
            .read()
            .values()
            .filter(|e| e.expires_at > now)
            .count()
    }

    /// `true` when no live entries are cached.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrow the wrapped probe. Useful for reaching probe-specific
    /// methods that aren't part of the [`ModelMetaProbe`] trait
    /// (e.g. [`crate::OllamaProbe::runtime`]) without dropping the
    /// cache wrapper.
    pub fn inner(&self) -> &P {
        &self.inner
    }
}

impl<P: ModelMetaProbe> ModelMetaProbe for Cache<P> {
    async fn describe(&self, model: &str) -> Result<Option<ModelDescriptor>, ProbeError> {
        // Fast path: shared-read lookup. Lock is released before .await.
        let now = Instant::now();
        let hit = {
            let guard = self.entries.read();
            guard
                .get(model)
                .filter(|e| e.expires_at > now)
                .map(|e| e.value.clone())
        };
        if let Some(value) = hit {
            return Ok(value);
        }

        // Slow path: probe the inner backend without holding any lock.
        let value = self.inner.describe(model).await?;

        // Re-acquire the write lock briefly to insert.
        let expires_at = Instant::now() + self.ttl;
        self.entries.write().insert(
            model.to_string(),
            Entry {
                value: value.clone(),
                expires_at,
            },
        );
        Ok(value)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::{ModelDescriptor, ProviderId};

    #[derive(Clone, Default)]
    struct Counting {
        calls: Arc<AtomicUsize>,
        known: Option<ModelDescriptor>,
    }

    impl Counting {
        fn known() -> Self {
            Self {
                calls: Arc::new(AtomicUsize::new(0)),
                known: Some(
                    ModelDescriptor::builder("test", "x")
                        .context_window(2048)
                        .build(),
                ),
            }
        }

        fn unknown() -> Self {
            Self {
                calls: Arc::new(AtomicUsize::new(0)),
                known: None,
            }
        }

        fn call_count(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    impl ModelMetaProbe for Counting {
        async fn describe(&self, _model: &str) -> Result<Option<ModelDescriptor>, ProbeError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(self.known.clone())
        }
    }

    #[tokio::test]
    async fn known_model_is_cached() {
        let probe = Counting::known();
        let cache = Cache::new(probe.clone(), Duration::from_secs(60));
        let a = cache.describe("x").await.unwrap().unwrap();
        let b = cache.describe("x").await.unwrap().unwrap();
        assert_eq!(a, b);
        assert_eq!(
            probe.call_count(),
            1,
            "second call should be served from cache"
        );
        assert_eq!(cache.len(), 1);
    }

    #[tokio::test]
    async fn unknown_model_is_also_cached() {
        let probe = Counting::unknown();
        let cache = Cache::new(probe.clone(), Duration::from_secs(60));
        assert!(cache.describe("x").await.unwrap().is_none());
        assert!(cache.describe("x").await.unwrap().is_none());
        assert_eq!(
            probe.call_count(),
            1,
            "unknown result must still be memoised"
        );
    }

    #[tokio::test]
    async fn expired_entries_force_reprobe() {
        let probe = Counting::known();
        let cache = Cache::new(probe.clone(), Duration::from_millis(0));
        cache.describe("x").await.unwrap();
        // TTL = 0 → every read is expired.
        cache.describe("x").await.unwrap();
        cache.describe("x").await.unwrap();
        assert_eq!(probe.call_count(), 3);
        assert_eq!(cache.len(), 0, "expired entries do not count");
    }

    #[tokio::test]
    async fn invalidate_clears_entries() {
        let probe = Counting::known();
        let cache = Cache::new(probe.clone(), Duration::from_secs(60));
        cache.describe("x").await.unwrap();
        cache.invalidate();
        cache.describe("x").await.unwrap();
        assert_eq!(probe.call_count(), 2);
    }

    #[tokio::test]
    async fn invalidate_model_targets_one_entry() {
        let probe = Counting::known();
        let cache = Cache::new(probe.clone(), Duration::from_secs(60));
        cache.describe("a").await.unwrap();
        cache.describe("b").await.unwrap();
        cache.invalidate_model("a");
        cache.describe("a").await.unwrap();
        cache.describe("b").await.unwrap();
        assert_eq!(probe.call_count(), 3, "only 'a' should be re-probed");
    }

    #[tokio::test]
    async fn errors_are_not_cached() {
        #[derive(Clone)]
        struct Flaky(Arc<AtomicUsize>);
        impl ModelMetaProbe for Flaky {
            async fn describe(&self, _m: &str) -> Result<Option<ModelDescriptor>, ProbeError> {
                let n = self.0.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    Err(ProbeError::Transport("first call fails".into()))
                } else {
                    Ok(Some(ModelDescriptor::new(ProviderId::new("t"), "x")))
                }
            }
        }

        let calls = Arc::new(AtomicUsize::new(0));
        let cache = Cache::new(Flaky(calls.clone()), Duration::from_secs(60));
        assert!(cache.describe("x").await.is_err());
        let desc = cache.describe("x").await.unwrap().unwrap();
        assert_eq!(desc.model, "x");
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }
}

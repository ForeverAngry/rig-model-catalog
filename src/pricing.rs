//! USD pricing tables for hosted model providers.
//!
//! Pricing is provider-quoted in **USD per million tokens** for four
//! distinct token categories:
//!
//! - `input_per_million` — uncached prompt tokens.
//! - `output_per_million` — completion tokens.
//! - `cached_input_per_million` — prompt tokens read from a
//!   provider-managed cache (OpenAI prompt-cache, Anthropic prompt-cache
//!   reads). Optional; absent means the provider does not bill cache hits
//!   separately, in which case cached tokens are charged at the
//!   `input_per_million` rate.
//! - `cache_write_per_million` — prompt tokens *written* into the
//!   provider's cache (Anthropic only at time of writing). Optional;
//!   absent means cache writes are billed at the `input_per_million`
//!   rate.
//!
//! These four buckets map 1-to-1 against the five `Usage` fields exposed
//! by `rig-core` (`input_tokens`, `output_tokens`, `cached_input_tokens`,
//! `cache_creation_input_tokens`; `total_tokens` is derived). The
//! [`ModelPrice::cost_for`] helper takes those four scalars and returns
//! the bill in USD; an opt-in `cost_for_usage` bridge against
//! `rig_core::completion::Usage` is exposed under the `rig-hook` feature.
//!
//! # Quick start
//!
//! ```
//! use rig_model_catalog::{ModelPrice, PricingTable};
//!
//! let table = PricingTable::builtin();
//! let price = table
//!     .lookup("openai", "gpt-4o-mini")
//!     .expect("gpt-4o-mini is in the seed catalog");
//!
//! // 10,000 prompt tokens, 2,000 completion tokens, no cache activity.
//! let usd = price.cost_for(10_000, 2_000, 0, 0);
//! assert!((usd - (10_000.0 * 0.15 + 2_000.0 * 0.60) / 1_000_000.0).abs() < 1e-12);
//! # let _: &ModelPrice = price;
//! ```
//!
//! # Freshness
//!
//! The bundled `data/pricing.json` is a **point-in-time snapshot**
//! curated against public provider pricing pages. Pricing changes
//! out-of-band; treat the builtin as a starting point and override with
//! [`PricingTable::from_json`] or programmatic [`PricingTable::with`]
//! calls when accuracy matters (cost dashboards, billing-tier gates).
//! [`PricingTable::snapshot_date`] and
//! [`PricingTable::snapshot_provenance`] expose the bundled snapshot
//! metadata when the JSON source provides it.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::ProviderId;

/// USD per-million-token rates for one `(provider, model)` pair.
///
/// `#[non_exhaustive]` so adjacent fields (e.g. image-token rates, tool-
/// call rates) can be added without a breaking change.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ModelPrice {
    /// USD per million uncached prompt / input tokens.
    pub input_per_million: f64,
    /// USD per million completion / output tokens.
    pub output_per_million: f64,
    /// USD per million prompt tokens served from the provider's cache.
    /// `None` means cache reads bill at `input_per_million`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_input_per_million: Option<f64>,
    /// USD per million prompt tokens written into the provider's cache.
    /// `None` means cache writes bill at `input_per_million`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_per_million: Option<f64>,
}

impl ModelPrice {
    /// Construct a minimal price quote (input + output rates only).
    pub const fn new(input_per_million: f64, output_per_million: f64) -> Self {
        Self {
            input_per_million,
            output_per_million,
            cached_input_per_million: None,
            cache_write_per_million: None,
        }
    }

    /// Builder: declare the cached-input rate.
    pub const fn with_cached_input(mut self, rate_per_million: f64) -> Self {
        self.cached_input_per_million = Some(rate_per_million);
        self
    }

    /// Builder: declare the cache-write rate.
    pub const fn with_cache_write(mut self, rate_per_million: f64) -> Self {
        self.cache_write_per_million = Some(rate_per_million);
        self
    }

    /// Compute the USD cost of a single turn given the four token
    /// buckets reported by the provider.
    ///
    /// Tokens that fall in the cache buckets are billed at their
    /// dedicated rate when present, and at `input_per_million` when not.
    /// Tokens are not double-counted: callers must pass the *exclusive*
    /// uncached input count in `input_tokens` (i.e. `Usage.input_tokens`
    /// is already net of `cached_input_tokens` for every provider rig
    /// supports today).
    pub fn cost_for(
        &self,
        input_tokens: u64,
        output_tokens: u64,
        cached_input_tokens: u64,
        cache_write_tokens: u64,
    ) -> f64 {
        const PER_MILLION: f64 = 1_000_000.0;
        let input = input_tokens as f64 * self.input_per_million;
        let output = output_tokens as f64 * self.output_per_million;
        let cached_rate = self
            .cached_input_per_million
            .unwrap_or(self.input_per_million);
        let cache_write_rate = self
            .cache_write_per_million
            .unwrap_or(self.input_per_million);
        let cached = cached_input_tokens as f64 * cached_rate;
        let cache_write = cache_write_tokens as f64 * cache_write_rate;
        (input + output + cached + cache_write) / PER_MILLION
    }

    /// Compute the USD cost of a turn from a `rig_core::completion::Usage`.
    ///
    /// Available under the `rig-hook` feature so the bridge lives next to
    /// the rest of the rig-core integration surface.
    #[cfg(feature = "rig-hook")]
    pub fn cost_for_usage(&self, usage: &rig_core::completion::Usage) -> f64 {
        self.cost_for(
            usage.input_tokens,
            usage.output_tokens,
            usage.cached_input_tokens,
            usage.cache_creation_input_tokens,
        )
    }
}

/// Owned representation of one JSON row in `data/pricing.json`.
#[derive(Debug, Clone, Deserialize)]
struct PricingEntry {
    provider: ProviderId,
    model: String,
    input_per_million: f64,
    output_per_million: f64,
    #[serde(default)]
    cached_input_per_million: Option<f64>,
    #[serde(default)]
    cache_write_per_million: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct PricingSnapshot {
    #[serde(default)]
    snapshot_date: Option<String>,
    #[serde(default)]
    snapshot_provenance: Option<String>,
    entries: Vec<PricingEntry>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum PricingJson {
    Snapshot(PricingSnapshot),
    Entries(Vec<PricingEntry>),
}

/// Lookup table mapping `(provider, model)` pairs to [`ModelPrice`].
///
/// The builtin catalog is curated and point-in-time; see the module-
/// level docs for freshness caveats.
#[derive(Debug, Clone, Default)]
pub struct PricingTable {
    entries: BTreeMap<(ProviderId, String), ModelPrice>,
    snapshot_date: Option<String>,
    snapshot_provenance: Option<String>,
}

impl PricingTable {
    /// Empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: insert / overwrite a single quote.
    pub fn with(
        mut self,
        provider: impl Into<ProviderId>,
        model: impl Into<String>,
        price: ModelPrice,
    ) -> Self {
        self.entries.insert((provider.into(), model.into()), price);
        self
    }

    /// Builder: attach machine-readable snapshot metadata.
    #[must_use]
    pub fn with_snapshot_metadata(
        mut self,
        snapshot_date: impl Into<String>,
        snapshot_provenance: impl Into<String>,
    ) -> Self {
        self.snapshot_date = Some(snapshot_date.into());
        self.snapshot_provenance = Some(snapshot_provenance.into());
        self
    }

    /// Snapshot date for the pricing source, when provided.
    #[must_use]
    pub fn snapshot_date(&self) -> Option<&str> {
        self.snapshot_date.as_deref()
    }

    /// Human-readable provenance for the pricing source, when provided.
    #[must_use]
    pub fn snapshot_provenance(&self) -> Option<&str> {
        self.snapshot_provenance.as_deref()
    }

    /// Look up a price quote. Returns `None` for unknown models.
    pub fn lookup(&self, provider: impl Into<ProviderId>, model: &str) -> Option<&ModelPrice> {
        self.entries.get(&(provider.into(), model.to_string()))
    }

    /// Number of `(provider, model)` rows in the table.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if the table has no rows.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over every quote, in `(provider, model)` order.
    pub fn iter(&self) -> impl Iterator<Item = (&ProviderId, &str, &ModelPrice)> {
        self.entries
            .iter()
            .map(|((provider, model), price)| (provider, model.as_str(), price))
    }

    /// Parse pricing JSON into a table.
    ///
    /// Accepts the original JSON array of `{provider, model,
    /// input_per_million, ...}` rows and the newer snapshot object shape:
    /// `{snapshot_date, snapshot_provenance, entries}`.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let parsed: PricingJson = serde_json::from_str(json)?;
        Ok(match parsed {
            PricingJson::Snapshot(snapshot) => Self::from_entries(snapshot.entries)
                .with_optional_snapshot_metadata(
                    snapshot.snapshot_date,
                    snapshot.snapshot_provenance,
                ),
            PricingJson::Entries(entries) => Self::from_entries(entries),
        })
    }

    fn with_optional_snapshot_metadata(
        mut self,
        snapshot_date: Option<String>,
        snapshot_provenance: Option<String>,
    ) -> Self {
        self.snapshot_date = snapshot_date;
        self.snapshot_provenance = snapshot_provenance;
        self
    }

    fn from_entries(entries: Vec<PricingEntry>) -> Self {
        let mut table = Self::new();
        for entry in entries {
            let price = ModelPrice {
                input_per_million: entry.input_per_million,
                output_per_million: entry.output_per_million,
                cached_input_per_million: entry.cached_input_per_million,
                cache_write_per_million: entry.cache_write_per_million,
            };
            table.entries.insert((entry.provider, entry.model), price);
        }
        table
    }

    /// Curated OpenAI + Anthropic price snapshot baked into the crate.
    ///
    /// See `data/pricing.json` for the source rows and the CHANGELOG for
    /// the snapshot date.
    pub fn builtin() -> Self {
        // Parsing a baked-in &'static str against a struct we authored:
        // a malformed bundle is a build-time invariant violation, not a
        // runtime error. We assert it parses with an explicit message
        // and surface the empty table on the impossible failure path.
        match Self::from_json(BUILTIN_JSON) {
            Ok(table) => table,
            Err(err) => {
                tracing::error!(
                    error = %err,
                    "rig-model-catalog: bundled pricing.json failed to parse; \
                     returning empty table",
                );
                Self::new()
            }
        }
    }
}

const BUILTIN_JSON: &str = include_str!("../data/pricing.json");

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
mod tests {
    use super::*;

    #[test]
    fn cost_for_input_output_only() {
        let price = ModelPrice::new(2.50, 10.00);
        // 1M input + 1M output => $2.50 + $10.00 = $12.50.
        let cost = price.cost_for(1_000_000, 1_000_000, 0, 0);
        assert!((cost - 12.50).abs() < 1e-9);
    }

    #[test]
    fn cost_for_with_cached_input_uses_dedicated_rate() {
        let price = ModelPrice::new(2.50, 10.00).with_cached_input(1.25);
        // 1M uncached + 1M cached + 0 output =>
        // $2.50 + $1.25 = $3.75.
        let cost = price.cost_for(1_000_000, 0, 1_000_000, 0);
        assert!((cost - 3.75).abs() < 1e-9);
    }

    #[test]
    fn cost_for_with_cache_write_uses_dedicated_rate() {
        let price = ModelPrice::new(3.00, 15.00)
            .with_cached_input(0.30)
            .with_cache_write(3.75);
        // 1M cache-write input => $3.75.
        let cost = price.cost_for(0, 0, 0, 1_000_000);
        assert!((cost - 3.75).abs() < 1e-9);
    }

    #[test]
    fn cost_falls_back_to_input_rate_when_cache_rates_absent() {
        let price = ModelPrice::new(2.50, 10.00);
        // No cached_input_per_million => bill at input rate.
        let cost = price.cost_for(0, 0, 1_000_000, 0);
        assert!((cost - 2.50).abs() < 1e-9);
        // No cache_write_per_million either.
        let cost = price.cost_for(0, 0, 0, 1_000_000);
        assert!((cost - 2.50).abs() < 1e-9);
    }

    #[test]
    fn pricing_table_with_then_lookup() {
        let table = PricingTable::new().with("openai", "gpt-4o", ModelPrice::new(2.50, 10.00));
        let price = table.lookup("openai", "gpt-4o").expect("inserted");
        assert!((price.input_per_million - 2.50).abs() < 1e-9);
        assert!(table.lookup("openai", "missing").is_none());
        assert_eq!(table.len(), 1);
        assert!(!table.is_empty());
    }

    #[test]
    fn pricing_table_from_json_round_trip() {
        let json = r#"[
            {
                "provider": "openai",
                "model": "gpt-4o",
                "input_per_million": 2.5,
                "output_per_million": 10.0,
                "cached_input_per_million": 1.25
            }
        ]"#;
        let table = PricingTable::from_json(json).expect("parses");
        let price = table.lookup("openai", "gpt-4o").expect("present");
        assert_eq!(price.cached_input_per_million, Some(1.25));
        assert_eq!(table.snapshot_date(), None);
        assert_eq!(table.snapshot_provenance(), None);
    }

    #[test]
    fn pricing_table_parses_snapshot_metadata() {
        let json = r#"{
            "snapshot_date": "2026-05-28",
            "snapshot_provenance": "manual public pricing snapshot",
            "entries": [
                {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "input_per_million": 0.15,
                    "output_per_million": 0.60
                }
            ]
        }"#;

        let table = PricingTable::from_json(json).expect("parses");

        assert_eq!(table.snapshot_date(), Some("2026-05-28"));
        assert_eq!(
            table.snapshot_provenance(),
            Some("manual public pricing snapshot")
        );
        assert!(table.lookup("openai", "gpt-4o-mini").is_some());
    }

    #[test]
    fn builtin_catalog_seeds_known_models() {
        let table = PricingTable::builtin();
        assert_eq!(table.snapshot_date(), Some("2026-05-28"));
        assert!(
            table
                .snapshot_provenance()
                .is_some_and(|value| value.contains("Manual public pricing"))
        );
        assert!(
            table.lookup("openai", "gpt-4o-mini").is_some(),
            "seed must include gpt-4o-mini",
        );
        assert!(
            table
                .lookup("anthropic", "claude-3-5-sonnet-20241022")
                .is_some(),
            "seed must include claude-3-5-sonnet-20241022",
        );
        // Spot-check non-zero rates.
        for (_, _, price) in table.iter() {
            assert!(price.input_per_million > 0.0);
            assert!(price.output_per_million > 0.0);
        }
    }

    #[test]
    fn pricing_table_iter_is_sorted() {
        let table = PricingTable::builtin();
        let keys: Vec<(String, String)> = table
            .iter()
            .map(|(p, m, _)| (p.as_str().to_string(), m.to_string()))
            .collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted, "iter() must yield rows in sorted order");
    }
}

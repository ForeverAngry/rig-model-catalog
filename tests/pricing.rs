//! Integration tests for the `pricing` feature: builtin catalog
//! invariants and round-tripping arithmetic against known rates.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use rig_model_catalog::{ModelPrice, PricingTable};

#[test]
fn builtin_table_costs_match_published_rates() {
    let table = PricingTable::builtin();

    // gpt-4o-mini: $0.15 in / $0.60 out per million.
    // 1M input + 1M output => $0.75.
    let mini = table.lookup("openai", "gpt-4o-mini").expect("seeded");
    let cost = mini.cost_for(1_000_000, 1_000_000, 0, 0);
    assert!(
        (cost - 0.75).abs() < 1e-9,
        "gpt-4o-mini 1M+1M expected $0.75, got ${cost}",
    );

    // claude-3-5-sonnet-20241022: $3 in / $15 out, $0.30 cached read,
    // $3.75 cache write per million.
    // 100k uncached + 100k cached + 100k cache_write + 50k output =>
    // 0.30 + 0.03 + 0.375 + 0.75 = $1.455.
    let sonnet = table
        .lookup("anthropic", "claude-3-5-sonnet-20241022")
        .expect("seeded");
    let cost = sonnet.cost_for(100_000, 50_000, 100_000, 100_000);
    assert!(
        (cost - 1.455).abs() < 1e-9,
        "sonnet mixed-bucket cost expected $1.455, got ${cost}",
    );
}

#[test]
fn user_supplied_table_overrides_via_builder() {
    let table = PricingTable::new()
        .with("custom", "frontier-1", ModelPrice::new(5.00, 25.00))
        .with(
            "custom",
            "cached-1",
            ModelPrice::new(2.00, 10.00).with_cached_input(0.20),
        );

    assert_eq!(table.len(), 2);
    let cached = table.lookup("custom", "cached-1").expect("inserted");
    assert_eq!(cached.cached_input_per_million, Some(0.20));
    // 1M cached input only => $0.20.
    let cost = cached.cost_for(0, 0, 1_000_000, 0);
    assert!((cost - 0.20).abs() < 1e-9);
}

#[test]
fn from_json_rejects_malformed_input() {
    let err = PricingTable::from_json("not json").unwrap_err();
    // Just confirm we surface a serde_json::Error rather than panicking.
    assert!(!err.to_string().is_empty());
}

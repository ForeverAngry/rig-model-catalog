# Roadmap

`rig-model-catalog` is the provider-agnostic model-catalogdata layer for the
Rig ecosystem. This roadmap tracks what has shipped, what is queued, and
what is deliberately out of scope. For day-to-day conventions see
[AGENTS.md](AGENTS.md).

## Landed

- `ModelMetaProbe` async trait with `describe(model) -> Result<Option<ModelDescriptor>, ProbeError>`.
- `ModelDescriptor` (`#[non_exhaustive]`) carrying provider, context window,
  capability set, quantization hint, and pricing-table key.
- `Capability` enum including `Thinking`, with host-contract documentation
  for the Ollama `additional_params({"think": true})` pattern and the
  live-assertion guard against raw `<think>` tags reaching users.
- `StubProbe` and `ChainedProbe` for tests and fallback chains.
- `OllamaProbe` (feature `ollama`) backed by `reqwest`, including
  `RuntimeDescriptor` + `OllamaProbe::runtime(model)` via `GET /api/ps`
  for the `num_ctx` footgun (manifest says 128k, loaded `num_ctx=2048`).
- `StaticProbe` (feature `static`) backed by bundled
  `data/openai.json` + `data/anthropic.json` with no extra runtime deps.
- `MetaHook` prompt-hook (feature `rig-hook`) emitting model metadata
  alongside Rig calls; `observe` extends it to emit the `rig-tap`
  `ObservabilityEvent` wire shape without depending on `rig-tap`.
- `PricingTable` + `ModelPrice` (feature `pricing`) with
  `cost_for(input, output, cached_input, cache_write)` and
  `cost_for_usage(&rig_core::completion::Usage)` under `rig-hook`.

## Next Work

1. **Pricing snapshot freshness** — wire a documented refresh cadence and
   provenance comment in `data/pricing.json`; expose
   `PricingTable::snapshot_date()` so downstream cost reports can flag
   stale rates instead of silently overcharging or undercharging.
2. **Provider coverage expansion** — add `StaticProbe` catalogs for Gemini
   and Mistral, keyed by the same `(provider, model)` pricing table so
   `cost_for_usage` covers them on the same dispatch path. Treat new
   providers as additive `data/*.json` plus an enum variant; no new
   feature flags.
3. **Capability surface** — fold structured-output and audio-in/out
   capability variants into `Capability` once the host providers
   advertise them stably. Additive only — `Capability` stays
   `#[non_exhaustive]`.
4. **Probe cache adapter** — provide a tiny in-process cache wrapper
   `CachedProbe<P>` so callers can avoid hammering `/api/tags`,
   `/api/show`, and `/api/ps` on every dispatch. Keep zero default deps;
   guard cache keys behind the probe's existing input.
5. **Runtime drift telemetry** — when both `describe()` and `runtime()`
   are queried, emit a structured `tracing` warning (no new event kind)
   if `effective_context_window < context_window`, so hosts can surface
   the `num_ctx` footgun in dashboards without bespoke comparison code.

## Prototype Grade

- The bundled OpenAI / Anthropic catalogs reflect a manual snapshot; date
  provenance is documented in `CHANGELOG.md` but not yet machine-readable
  in `data/*.json`. Treat anything older than 90 days as stale.
- `OllamaProbe` assumes the Ollama HTTP daemon is reachable at the
  configured base URL. Hosts behind reverse proxies should chain a
  `StaticProbe` ahead of it.
- `MetaHook` telemetry uses the `rig-tap` JSON wire shape but does not
  depend on `rig-tap`; if downstream collectors expect a hard schema
  contract, pin both crates together.

## Out of Scope

- Bundling a full pricing oracle. `PricingTable` is a static catalog plus
  override hooks; live billing belongs in the provider client.
- Forking or vendoring `rig-core`. The optional `rig-hook` /
  `observe` features pull `rig-core` with `default-features = false`
  precisely so this crate remains consumable by `rig-memvid`,
  `rig-compose`, `rig-resources`, `rig-mcp`, `rig-retrieval-evals`, and
  `rig-tokudo` without coupling them to each other.
- Live capability probing (function-calling smoke tests, vision-input
  probes, etc.). That belongs in `rig-retrieval-evals` or a downstream eval
  harness; this crate reports metadata, not behavior.

## Reopen Triggers

- A provider publishes a stable model-catalogdata endpoint and we can
  retire the matching `StaticProbe` JSON.
- `rig-core` exposes a `Usage` extension that adds fields beyond
  `cached_input` / `cache_write`; `ModelPrice::cost_for_usage` must
  follow additively.
- `Capability::Thinking` semantics change upstream (e.g. Ollama renames
  `think` or moves it out of `additional_params`); update the doc
  contract on `Capability::Thinking` and the live assertion guidance in
  the README.

# Changelog

All notable changes to `rig-model-meta` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this crate adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `RuntimeDescriptor` + `OllamaProbe::runtime(model)` — `GET /api/ps`
  surface that returns the model's *currently loaded* state if any:
  `effective_context_window` (Ollama's runtime `num_ctx`), `size_vram_bytes`,
  and `expires_at`. Compare against
  `ModelDescriptor::context_window` from `describe()` to detect the
  `num_ctx` footgun where a 128k-manifest model silently loads with
  `num_ctx=2048`. Returns `Ok(None)` when the model isn't currently
  loaded. `RuntimeDescriptor` is `#[non_exhaustive]` and re-exported
  unconditionally (plain data); the lookup method itself only exists
  on `OllamaProbe`, behind the `ollama` feature.
- `PricingTable` + `ModelPrice` (feature `pricing`) — USD-per-million-
  token rates keyed by `(provider, model)`, with optional dedicated
  `cached_input_per_million` and `cache_write_per_million` rates that
  map 1-to-1 against rig-core's `Usage` cache fields. `ModelPrice::cost_for(
  input, output, cached_input, cache_write)` returns USD; the
  `rig-hook`-gated `cost_for_usage(&rig_core::completion::Usage)` is a
  one-call bridge for hook consumers. Bundled `data/pricing.json` seeds
  OpenAI (`gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo`) and
  Anthropic (`claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022`,
  `claude-3-opus-20240229`) at their published per-million rates;
  override programmatically via `PricingTable::with(...)` or load a
  fresher snapshot via `PricingTable::from_json(...)`. Snapshot taken
  2025-Q1 against the providers' public pricing pages — treat as a
  starting point, not a billing source of truth.
- `MetaHook` (feature `rig-hook`) — implements
  `rig_core::agent::PromptHook` and stamps `gen_ai_*` telemetry on every
  completion call: provider, model, context window, per-turn
  `input_tokens` / `output_tokens` / `total_tokens`, and a derived
  `context_used_pct`. Construct eagerly with `MetaHook::resolve` (one
  probe call up-front) or lazily with `MetaHook::unresolved`. Observation
  only — always returns `HookAction::cont()`.
- `HookPair<A, B>` (feature `rig-hook`) — sequentially compose two
  `PromptHook`s into a single one (`first` runs before `second`, with
  short-circuit on any non-`Continue` action). Bridges the gap that
  `rig-core`'s `.with_hook(...)` only accepts a single hook. Chains
  further with `.then(third)`.
- `Cache<P>` — TTL-bounded memoiser for any [`ModelMetaProbe`]. Caches
  both `Some(...)` and `None` results so repeat lookups of unknown
  models are also short-circuited. Errors are not cached. Exposes
  `invalidate()` and `invalidate_model(...)` for manual eviction.
- `ModelMetaProbeDyn` — object-safe mirror of `ModelMetaProbe` with a
  blanket impl over every static probe. Lets callers store probes in
  `Box<dyn ...>` / `Arc<dyn ...>`.
- `DynProbe` — adapter that lifts an `Arc<dyn ModelMetaProbeDyn>` back
  into the static `ModelMetaProbe` trait for downstream composition
  (`ChainedProbe`, `Cache`, etc.).
- `ProbeFuture<'a>` type alias for the boxed-future return shape.
- `ModelMetaProbe` trait — provider-agnostic async lookup returning
  `Result<Option<ModelDescriptor>, ProbeError>`.
- `ModelDescriptor` (non-exhaustive) with `context_window`,
  `max_output_tokens`, `capabilities`, `family`, `parameter_count`,
  `quantization`, and a `raw` JSON passthrough.
- `Capability` and `Quantization` enums covering the cases Ollama and the
  major cloud providers actually report.
- `StubProbe` — in-memory probe for tests and fixtures.
- `ChainedProbe` with strict (`new`) and tolerant (`tolerant`) composition
  modes.
- `OllamaProbe` (feature `ollama`) — live probe against
  `POST /api/show`, extracting context window, capabilities, family,
  parameter size, and quantization.
- `StaticProbe` (feature `static`) — curated in-memory catalog with
  bundled OpenAI (`gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo`,
  `o1`, `o1-mini`, `text-embedding-3-{small,large}`) and Anthropic
  (`claude-3-5-{sonnet,haiku}`, `claude-3-{opus,haiku}`) entries.
- Examples: `probe_ollama`, `chained`.

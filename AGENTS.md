# AGENTS.md

Guidance for AI coding agents working in `rig-model-meta`. Mirrors
[.github/copilot-instructions.md](.github/copilot-instructions.md).

## Project

`rig-model-meta` is the provider-agnostic model-metadata layer for the
Rig ecosystem. Public primitives:

- `ModelMetaProbe` trait — async `describe(model) -> Result<Option<...>>`
  ([src/probe.rs](src/probe.rs)).
- `ModelDescriptor` / `Capability` / `Quantization` / `ProviderId`
  ([src/descriptor.rs](src/descriptor.rs)).
- `StubProbe`, `ChainedProbe` ([src/stub.rs](src/stub.rs), [src/chained.rs](src/chained.rs)).
- `OllamaProbe` (feature `ollama`, [src/ollama.rs](src/ollama.rs)).
- `StaticProbe` (feature `static`, [src/static_catalog.rs](src/static_catalog.rs))
  backed by `data/openai.json` + `data/anthropic.json`.

## Rules

- Rust 2024, MSRV 1.89. Library is runtime-agnostic; do not add `tokio`
  to `[dependencies]`.
- Errors: typed `thiserror` enum in [src/error.rs](src/error.rs); return
  `Result<_, ProbeError>`. Use `Option` for "unknown is normal" cases —
  do not use `Err` to mean "model not in catalog."
- Never `.await` while holding a `Mutex`/`RwLock` guard. Scope-drop
  first (`clippy::await_holding_lock = deny`).
- No `unwrap`, `expect`, `panic!`, `todo!`, `unimplemented!`, `dbg!`,
  indexing/slicing, or `unreachable!` in library code — clippy
  `deny`/`forbid`. Use `?`, `ok_or(ProbeError::...)`, `get(..)`, `match`,
  `strip_suffix`. Allowed in `tests/`, `examples/`, `#[cfg(test)]`
  blocks (gate with
  `#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]`).
- Use `tracing` for logs; no `println!` in library code.
- Document new `pub` items with `///` rustdoc; provide a `no_run` /
  doctested example for new traits or probe types.
- Re-export new public items from [src/lib.rs](src/lib.rs).
- `ModelDescriptor` is `#[non_exhaustive]` — adding a field is additive.
  Renaming or removing a field is a **breaking** change.

## Feature flags

Default = none. Optional: `ollama` (pulls `reqwest`), `static` (bundled
JSON via `include_str!`), `rig-hook` (pulls `rig-core` for `PromptHook`
telemetry), and `pricing` (bundled pricing catalog). Gate optional code with
`#[cfg(feature = "...")]`. CI matrix covers default, individual feature
combos, combined provider/catalog features, and all-features.

## Validation

```sh
just check
# fmt --check + clippy (× feature combos) + tests + examples
```

Integration tests live in [tests/](tests/). Examples must keep building:
`cargo build --examples --all-features`.

## Scope

Do not vendor `rig-core`. The crate must not depend on `rig-memvid`,
`rig-compose`, `rig-resources`, `rig-mcp`, or `rig-evals-rag` — it must
remain consumable by **all** of them. Update [README.md](README.md) and
[CHANGELOG.md](CHANGELOG.md) for user-visible changes.

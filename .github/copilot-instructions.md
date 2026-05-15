# rig-model-meta — Copilot Instructions

See [AGENTS.md](../AGENTS.md) for the authoritative copy. Summary:

- Rust 2024, MSRV `1.89`. Library is runtime-agnostic — do not add
  `tokio` to `[dependencies]`.
- Typed `thiserror::Error` in [src/error.rs](../src/error.rs). Return
  `Result<_, ProbeError>`. Use `Option<ModelDescriptor>` for "unknown is
  normal"; reserve `Err` for hard failures (transport, parse, auth).
- Never `.await` while holding a `Mutex`/`RwLock` guard
  (`clippy::await_holding_lock` is `deny`).
- No `unwrap`/`expect`/`panic!`/`todo!`/`unimplemented!`/`dbg!`/indexing
  in library code (clippy `deny`/`forbid`). Use `?`,
  `ok_or(ProbeError::…)`, `get(..)`, `strip_suffix`, pattern matching.
- `unwrap`/`expect` allowed in `tests/`, `examples/`, and `#[cfg(test)]`
  blocks (gate the test module with
  `#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]`).
- Use `tracing` for logs; no `println!` in library code.
- Document new `pub` items with `///` rustdoc. Re-export from
  [src/lib.rs](../src/lib.rs).
- `ModelDescriptor` is `#[non_exhaustive]`. Adding a field is additive;
  renaming / removing is breaking.

## Validation

```sh
just check
# fmt --check + clippy (× feature combos) + cargo test --all-features + examples
```

## Feature flags

- _default_: trait + descriptor + `StubProbe` + `ChainedProbe`.
- `ollama`: pulls `reqwest`. Adds `OllamaProbe`.
- `static`: zero extra runtime deps. Adds `StaticProbe` backed by
  `data/openai.json` + `data/anthropic.json`.
- `rig-hook`: pulls `rig-core` with default features disabled. Adds `MetaHook`
  prompt-hook telemetry.
- `pricing`: zero extra runtime deps. Adds `PricingTable` / `ModelPrice` backed
  by `data/pricing.json`.

## Scope

Do not depend on `rig-compose`, `rig-memvid`, `rig-resources`, `rig-mcp`,
`rig-veh`, or `rig-evals-rag`. Keep the optional `rig-core` dependency confined
to the `rig-hook` feature with default features disabled so this crate remains
consumable by all of them.

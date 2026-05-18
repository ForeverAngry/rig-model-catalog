# rig-model-meta

Provider-agnostic model metadata for the [Rig](https://crates.io/crates/rig-core)
ecosystem.

`rig-model-meta` answers one question consistently across providers:

> _Given a model id, what is its context window, what capabilities does it
> advertise, and what do I need to know to size prompts and budget tokens?_

It ships a single trait — `ModelMetaProbe` — plus a plain-data
`ModelDescriptor` and a handful of backends (live Ollama, static catalogs
for OpenAI / Anthropic, and a stub for tests). The default build pulls
**no** HTTP client or provider SDKs; backends are opt-in via Cargo
features.

## Status

`v0.1.0` — initial release. API is `#[non_exhaustive]` where it counts;
additive changes (new descriptor fields, new probes, new capability
variants) are not breaking.

## Quick start

```toml
[dependencies]
rig-model-meta = { version = "0.1", features = ["ollama", "static"] }
```

```rust,no_run
use rig_model_meta::{ChainedProbe, ModelMetaProbe, OllamaProbe, StaticProbe};

# async fn run() -> anyhow::Result<()> {
let probe = ChainedProbe::new(
    OllamaProbe::live("http://localhost:11434"),
    StaticProbe::builtin(),
);

if let Some(desc) = probe.describe("gpt-4o").await? {
    println!("context window: {:?}", desc.context_window);
    println!("capabilities:   {:?}", desc.capabilities);
}
# Ok(()) }
```

## Thinking-capable models

`Capability::Thinking` means the model advertises a native reasoning channel.
For Ollama models used through Rig, callers should opt into that native
separation with `think: true` and still assert that user-visible output is free
of raw `<think>` / `</think>` tags.

```rust,no_run
use rig_model_meta::{Capability, ModelMetaProbe, OllamaProbe};

let base_url = "http://localhost:11434";
let model_name = "qwen3.5:9b";
let model = ollama_client.completion_model(model_name);

let probe = OllamaProbe::live(base_url);
let descriptor = probe.describe(model_name).await?;
let supports_thinking = descriptor
  .as_ref()
  .is_some_and(|d| d.capabilities.contains(&Capability::Thinking));

let mut builder = rig::agent::AgentBuilder::new(model);
if supports_thinking {
  builder = builder.additional_params(serde_json::json!({ "think": true }));
}
let agent = builder.build();
```

The capability is not a sanitizer. It tells the host which provider contract to
activate. Applications that render model output directly should keep a live
smoke check equivalent to `!output.contains("<think>") &&
!output.contains("</think>")` so provider or model regressions are caught
before users see reasoning tags.

## Features

| Feature   | Pulls          | What you get                                      |
|-----------|----------------|---------------------------------------------------|
| _default_ | —              | `ModelMetaProbe`, `ModelDescriptor`, `StubProbe`, `ChainedProbe` |
| `ollama`  | `reqwest`      | `OllamaProbe::live(...)` against `POST /api/show` |
| `static`  | —              | `StaticProbe::builtin()` with bundled OpenAI + Anthropic tables |
| `rig-hook`| `rig-core`     | `MetaHook` + `HookPair` (PromptHook telemetry composition) |
| `observe` | `rig-hook`     | Re-emits `MetaHook` prompt lifecycle events on the `rig_tap` target |
| `pricing` | —              | `PricingTable` + `ModelPrice` with bundled OpenAI + Anthropic USD/M rates |

## Design rules

- **Unknown is normal.** Probes return `Ok(None)` for models they don't
  recognise. `Err` is reserved for hard failures (transport down, auth
  rejection, malformed response).
- **No `tokio` in `[dependencies]`.** The library is runtime-agnostic;
  tests and examples use `tokio` via dev-deps.
- **No panics in library code.** Every fallible operation is a `Result`
  or an `Option`. Clippy `unwrap_used`, `expect_used`, `panic`,
  `indexing_slicing`, `todo`, `unimplemented`, `unreachable`, `dbg_macro`,
  and `await_holding_lock` are `deny`/`forbid`.
- **`#[non_exhaustive]` on the public payload.** Adding fields to
  `ModelDescriptor` or variants to `Capability` is not a breaking change.

## Roadmap

Released in `v0.1.0`. Unreleased on `main` and exercised in the CI matrix:
`MetaHook` + `HookPair` (`rig-hook` feature), `MetaHook` `rig_tap`
emission (`observe` feature), `PricingTable` + `ModelPrice` (`pricing` feature),
`RuntimeDescriptor` / `OllamaProbe::runtime`,
`ModelMetaProbeDyn` + `DynProbe`, and a TTL `Cache<P>` memoiser. See
[`CHANGELOG.md`](CHANGELOG.md) for the full list.

Still planned:

- Knowledge-cutoff + deprecation fields (additive, data files).
- Bedrock and llama.cpp probes.
- CLI: `rig-model-meta show <provider>:<model>`.

## License

Dual-licensed under [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE) at your option.

//! Compose a (stub-)Ollama probe with the bundled static catalog.
//!
//! ```sh
//! cargo run --example chained --features "static"
//! ```

use rig_model_meta::{
    Capability, ChainedProbe, ModelDescriptor, ModelMetaProbe, StaticProbe, StubProbe,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Stand in for a live OllamaProbe so the example doesn't need a daemon.
    let ollama_stub = StubProbe::new([(
        "qwen3.5:9b",
        ModelDescriptor::builder("ollama", "qwen3.5:9b")
            .context_window(131_072)
            .family("qwen")
            .capability(Capability::Completion)
            .capability(Capability::Tools)
            .build(),
    )]);

    let chained = ChainedProbe::new(ollama_stub, StaticProbe::builtin());

    for model in [
        "qwen3.5:9b",
        "gpt-4o",
        "claude-3-5-sonnet-latest",
        "unknown",
    ] {
        match chained.describe(model).await? {
            Some(d) => println!(
                "{:<30} provider={:<10} ctx={:>7?} caps={:?}",
                d.model, d.provider, d.context_window, d.capabilities
            ),
            None => println!("{model:<30} <unknown>"),
        }
    }

    Ok(())
}

//! Live probe against a running Ollama daemon.
//!
//! Run with:
//!
//! ```sh
//! OLLAMA_URL=http://localhost:11434 \
//! OLLAMA_MODEL=qwen3.5:9b \
//!     cargo run --example probe_ollama --features ollama
//! ```
//!
//! If the model isn't pulled, the example prints `unknown model` and exits 0.

use std::env;

use rig_model_catalog::{ModelMetaProbe, OllamaProbe};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let base_url = env::var("OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());
    let model = env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen3.5:9b".to_string());

    let probe = OllamaProbe::live(&base_url);
    match probe.describe(&model).await? {
        Some(desc) => {
            println!("provider:         {}", desc.provider);
            println!("model:            {}", desc.model);
            println!("context_window:   {:?}", desc.context_window);
            println!("max_output_tokens:{:?}", desc.max_output_tokens);
            println!("family:           {:?}", desc.family);
            println!("parameter_count:  {:?}", desc.parameter_count);
            println!("quantization:     {:?}", desc.quantization);
            println!("capabilities:     {:?}", desc.capabilities);
        }
        None => {
            println!("unknown model: {model} (try `ollama pull {model}`)");
        }
    }

    Ok(())
}

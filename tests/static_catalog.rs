//! Integration coverage for the bundled static catalog.

#![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]

use rig_model_catalog::{Capability, ModelMetaProbe, StaticProbe};

#[tokio::test]
async fn openai_catalog_has_gpt_family() {
    let probe = StaticProbe::builtin();
    for model in [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "o1",
        "o1-mini",
    ] {
        let desc = probe
            .describe(model)
            .await
            .unwrap()
            .unwrap_or_else(|| panic!("missing {model} from openai catalog"));
        assert!(
            desc.context_window.is_some(),
            "{model} missing context window"
        );
        assert!(
            desc.has_capability(Capability::Completion)
                || desc.has_capability(Capability::Embedding)
        );
    }
}

#[tokio::test]
async fn anthropic_catalog_has_claude_family() {
    let probe = StaticProbe::builtin();
    for model in [
        "claude-3-5-sonnet-latest",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-latest",
        "claude-3-opus-latest",
        "claude-3-haiku-20240307",
    ] {
        let desc = probe
            .describe(model)
            .await
            .unwrap()
            .unwrap_or_else(|| panic!("missing {model} from anthropic catalog"));
        assert_eq!(desc.context_window, Some(200_000));
        assert!(desc.has_capability(Capability::Vision));
    }
}

#[tokio::test]
async fn extend_from_custom_json() {
    let mut probe = StaticProbe::default();
    let json = r#"[{"model":"custom-7b","context_window":32768,"capabilities":["completion"]}]"#;
    probe.extend_from_json_str("custom-provider", json).unwrap();
    let desc = probe.describe("custom-7b").await.unwrap().unwrap();
    assert_eq!(desc.context_window, Some(32_768));
    assert_eq!(desc.provider.as_str(), "custom-provider");
}

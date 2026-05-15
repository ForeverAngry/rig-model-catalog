//! [`OllamaProbe`] — live metadata for any model pulled by an Ollama daemon.
//!
//! Calls `POST /api/show` and parses the response into a [`ModelDescriptor`].
//! The interesting fields exposed by Ollama:
//!
//! - `capabilities: ["completion", "tools", "vision", ...]` → [`Capability`]
//! - `details.family` → architecture family (`llama`, `qwen2`, ...)
//! - `details.parameter_size` (e.g. `"7.2B"`) → [`ModelDescriptor::parameter_count`]
//! - `details.quantization_level` (e.g. `"Q4_K_M"`) → [`Quantization`]
//! - `model_info["<arch>.context_length"]` → context window
//!
//! 404 / "model not found" maps to `Ok(None)` per the [`ModelMetaProbe`]
//! contract.

use std::collections::BTreeSet;
use std::sync::Arc;

use reqwest::StatusCode;
use serde::Deserialize;

use crate::{
    Capability, ModelDescriptor, ModelMetaProbe, ProbeError, ProviderId, Quantization,
    RuntimeDescriptor,
};

/// Live Ollama metadata probe.
///
/// Clones share the same [`reqwest::Client`] and base URL via [`Arc`], so
/// passing a probe into multiple chains is cheap.
#[derive(Debug, Clone)]
pub struct OllamaProbe {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    client: reqwest::Client,
    base_url: String,
}

impl OllamaProbe {
    /// Construct a probe pointing at a live Ollama daemon.
    ///
    /// `base_url` should be the daemon root, e.g. `http://localhost:11434`.
    /// Trailing slashes are tolerated.
    pub fn live(base_url: impl Into<String>) -> Self {
        Self::with_client(reqwest::Client::new(), base_url)
    }

    /// Variant for callers who want to share a pre-configured
    /// [`reqwest::Client`] (custom timeouts, proxies, instrumentation).
    pub fn with_client(client: reqwest::Client, base_url: impl Into<String>) -> Self {
        let base_url = base_url.into().trim_end_matches('/').to_string();
        Self {
            inner: Arc::new(Inner { client, base_url }),
        }
    }

    /// Look up the *runtime* state of a model — i.e. whether the daemon
    /// has it loaded right now and with what context window.
    ///
    /// Returns `Ok(None)` when the model is not currently loaded
    /// (cold-start, evicted, never pulled). Compare
    /// [`RuntimeDescriptor::effective_context_window`] against
    /// [`ModelDescriptor::context_window`] from
    /// [`crate::ModelMetaProbe::describe`] to detect the
    /// `OLLAMA_NUM_CTX` / `num_ctx` footgun where a 128k-manifest model
    /// silently loads with `num_ctx=2048`.
    ///
    /// Calls `GET /api/ps` and matches by exact `model` string. Unknown
    /// fields in the response are preserved on
    /// [`RuntimeDescriptor::raw`].
    pub async fn runtime(&self, model: &str) -> Result<Option<RuntimeDescriptor>, ProbeError> {
        let url = format!("{}/api/ps", self.inner.base_url);
        let resp = self
            .inner
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ProbeError::Transport(e.to_string()))?;

        match resp.status() {
            StatusCode::OK => {}
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
                return Err(ProbeError::Unauthorized);
            }
            status => {
                let body = resp.text().await.unwrap_or_default();
                return Err(ProbeError::Other(format!(
                    "ollama /api/ps returned {status}: {body}"
                )));
            }
        }

        let body: PsResponse = resp
            .json()
            .await
            .map_err(|e| ProbeError::Parse(e.to_string()))?;
        Ok(body.find_runtime(model))
    }
}

impl ModelMetaProbe for OllamaProbe {
    async fn describe(&self, model: &str) -> Result<Option<ModelDescriptor>, ProbeError> {
        let url = format!("{}/api/show", self.inner.base_url);
        let resp = self
            .inner
            .client
            .post(&url)
            .json(&serde_json::json!({ "model": model }))
            .send()
            .await
            .map_err(|e| ProbeError::Transport(e.to_string()))?;

        match resp.status() {
            StatusCode::OK => {}
            StatusCode::NOT_FOUND => return Ok(None),
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
                return Err(ProbeError::Unauthorized);
            }
            status => {
                // Some Ollama builds return 500 with a body like
                // `{"error":"model 'foo' not found"}` for missing models.
                let body = resp.text().await.unwrap_or_default();
                if body.to_lowercase().contains("not found") {
                    return Ok(None);
                }
                return Err(ProbeError::Other(format!(
                    "ollama /api/show returned {status}: {body}"
                )));
            }
        }

        let body: ShowResponse = resp
            .json()
            .await
            .map_err(|e| ProbeError::Parse(e.to_string()))?;
        Ok(Some(body.into_descriptor(model)))
    }
}

/// Subset of Ollama's `/api/show` response we care about. Unknown fields
/// are ignored.
#[derive(Debug, Deserialize)]
struct ShowResponse {
    #[serde(default)]
    capabilities: Vec<String>,
    #[serde(default)]
    details: Option<Details>,
    #[serde(default)]
    model_info: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct Details {
    #[serde(default)]
    family: Option<String>,
    #[serde(default)]
    parameter_size: Option<String>,
    #[serde(default)]
    quantization_level: Option<String>,
}

impl ShowResponse {
    fn into_descriptor(self, model: &str) -> ModelDescriptor {
        let capabilities: BTreeSet<Capability> = self
            .capabilities
            .iter()
            .filter_map(|s| parse_capability(s))
            .collect();

        let (family, parameter_count, quantization) = match self.details {
            Some(d) => (
                d.family,
                d.parameter_size.as_deref().and_then(parse_parameter_size),
                d.quantization_level.as_deref().map(Quantization::parse),
            ),
            None => (None, None, None),
        };

        let context_window = extract_context_window(&self.model_info);

        let raw = serde_json::Value::Object(self.model_info);

        ModelDescriptor {
            provider: ProviderId::new("ollama"),
            model: model.to_string(),
            context_window,
            max_output_tokens: None,
            capabilities,
            family,
            parameter_count,
            quantization,
            raw: if raw.as_object().is_some_and(|m| m.is_empty()) {
                None
            } else {
                Some(raw)
            },
        }
    }
}

fn parse_capability(s: &str) -> Option<Capability> {
    match s.trim().to_ascii_lowercase().as_str() {
        "completion" => Some(Capability::Completion),
        "tools" | "tool_use" | "function_calling" => Some(Capability::Tools),
        "vision" => Some(Capability::Vision),
        "embedding" => Some(Capability::Embedding),
        "thinking" => Some(Capability::Thinking),
        "structured_output" | "json_schema" => Some(Capability::StructuredOutput),
        "image_generation" => Some(Capability::ImageGen),
        _ => None,
    }
}

/// Parse Ollama's "7.2B" / "13B" / "350M" style labels into an absolute
/// parameter count.
fn parse_parameter_size(s: &str) -> Option<u64> {
    let trimmed = s.trim();
    let (num_part, multiplier) = if let Some(prefix) = trimmed
        .strip_suffix('B')
        .or_else(|| trimmed.strip_suffix('b'))
    {
        (prefix, 1_000_000_000_u64)
    } else if let Some(prefix) = trimmed
        .strip_suffix('M')
        .or_else(|| trimmed.strip_suffix('m'))
    {
        (prefix, 1_000_000_u64)
    } else if let Some(prefix) = trimmed
        .strip_suffix('K')
        .or_else(|| trimmed.strip_suffix('k'))
    {
        (prefix, 1_000_u64)
    } else {
        return trimmed.parse::<u64>().ok();
    };
    let value: f64 = num_part.parse().ok()?;
    Some((value * multiplier as f64) as u64)
}

/// Scan `model_info` for the architecture-prefixed `*.context_length` key.
/// Ollama emits e.g. `llama.context_length`, `qwen2.context_length`.
fn extract_context_window(model_info: &serde_json::Map<String, serde_json::Value>) -> Option<u64> {
    for (k, v) in model_info {
        if k.ends_with(".context_length") {
            if let Some(n) = v.as_u64() {
                return Some(n);
            }
            if let Some(f) = v.as_f64()
                && f.is_finite()
                && f >= 0.0
            {
                return Some(f as u64);
            }
        }
    }
    None
}

/// Subset of Ollama's `/api/ps` response. Unknown fields are ignored;
/// the full per-model object is preserved on
/// [`RuntimeDescriptor::raw`].
#[derive(Debug, Deserialize)]
struct PsResponse {
    #[serde(default)]
    models: Vec<PsModel>,
}

#[derive(Debug, Deserialize)]
struct PsModel {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    context_length: Option<u64>,
    #[serde(default)]
    size_vram: Option<u64>,
    #[serde(default)]
    expires_at: Option<String>,
    /// Catch-all so unmodeled fields survive on `RuntimeDescriptor.raw`.
    #[serde(flatten)]
    extra: serde_json::Map<String, serde_json::Value>,
}

impl PsResponse {
    fn find_runtime(self, target: &str) -> Option<RuntimeDescriptor> {
        self.models
            .into_iter()
            .find(|m| m.model.as_deref() == Some(target) || m.name.as_deref() == Some(target))
            .map(|m| {
                let raw = if m.extra.is_empty() {
                    None
                } else {
                    Some(serde_json::Value::Object(m.extra))
                };
                RuntimeDescriptor {
                    provider: ProviderId::new("ollama"),
                    model: target.to_string(),
                    effective_context_window: m.context_length,
                    size_vram_bytes: m.size_vram,
                    expires_at: m.expires_at,
                    raw,
                }
            })
    }
}

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
    fn parameter_size_parses_units() {
        assert_eq!(parse_parameter_size("7B"), Some(7_000_000_000));
        assert_eq!(parse_parameter_size("7.2B"), Some(7_200_000_000));
        assert_eq!(parse_parameter_size("350M"), Some(350_000_000));
        assert_eq!(parse_parameter_size("1.5b"), Some(1_500_000_000));
        assert!(parse_parameter_size("garbage").is_none());
    }

    #[test]
    fn capability_parser_covers_known_strings() {
        assert_eq!(parse_capability("completion"), Some(Capability::Completion));
        assert_eq!(parse_capability("tools"), Some(Capability::Tools));
        assert_eq!(parse_capability("Vision"), Some(Capability::Vision));
        assert_eq!(parse_capability("thinking"), Some(Capability::Thinking));
        assert!(parse_capability("nonsense").is_none());
    }

    #[test]
    fn context_window_walks_arch_prefix() {
        let mut m = serde_json::Map::new();
        m.insert("llama.context_length".into(), serde_json::json!(131072));
        m.insert("llama.attention.heads".into(), serde_json::json!(32));
        assert_eq!(extract_context_window(&m), Some(131072));
    }

    #[test]
    fn parses_full_show_response() {
        let body = serde_json::json!({
            "capabilities": ["completion", "tools", "vision"],
            "details": {
                "family": "llama",
                "parameter_size": "7.2B",
                "quantization_level": "Q4_K_M",
            },
            "model_info": {
                "llama.context_length": 131072,
            },
        });
        let resp: ShowResponse = serde_json::from_value(body).unwrap();
        let desc = resp.into_descriptor("llama3.2:7b");
        assert_eq!(desc.provider, ProviderId::new("ollama"));
        assert_eq!(desc.context_window, Some(131072));
        assert_eq!(desc.family.as_deref(), Some("llama"));
        assert_eq!(desc.parameter_count, Some(7_200_000_000));
        assert_eq!(desc.quantization, Some(Quantization::Q4KM));
        assert!(desc.has_capability(Capability::Completion));
        assert!(desc.has_capability(Capability::Tools));
        assert!(desc.has_capability(Capability::Vision));
    }

    #[test]
    fn ps_response_matches_by_model_field() {
        let body = serde_json::json!({
            "models": [{
                "name": "lfm2.5-thinking:1.2b",
                "model": "lfm2.5-thinking:1.2b",
                "size_vram": 944946208_u64,
                "expires_at": "2026-05-15T10:25:31-04:00",
                "context_length": 4096,
                "details": { "family": "lfm2" }
            }]
        });
        let resp: PsResponse = serde_json::from_value(body).unwrap();
        let rt = resp
            .find_runtime("lfm2.5-thinking:1.2b")
            .expect("model present");
        assert_eq!(rt.provider, ProviderId::new("ollama"));
        assert_eq!(rt.effective_context_window, Some(4096));
        assert_eq!(rt.size_vram_bytes, Some(944_946_208));
        assert_eq!(rt.expires_at.as_deref(), Some("2026-05-15T10:25:31-04:00"));
        // Unmodeled fields land on `raw`.
        let raw = rt.raw.as_ref().expect("raw populated");
        assert!(raw.get("details").is_some(), "details preserved on raw");
    }

    #[test]
    fn ps_response_returns_none_for_unloaded_model() {
        let body = serde_json::json!({ "models": [] });
        let resp: PsResponse = serde_json::from_value(body).unwrap();
        assert!(resp.find_runtime("anything").is_none());
    }

    #[test]
    fn ps_response_matches_by_name_when_model_missing() {
        let body = serde_json::json!({
            "models": [{
                "name": "qwen2.5-coder:3b",
                "context_length": 32_768,
            }]
        });
        let resp: PsResponse = serde_json::from_value(body).unwrap();
        let rt = resp.find_runtime("qwen2.5-coder:3b").expect("matched");
        assert_eq!(rt.effective_context_window, Some(32_768));
    }
}

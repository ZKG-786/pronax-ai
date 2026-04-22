
use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// 3D Spatial coordinate for middleware response tracking
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ResponseCoordinate {
    /// X: Request sequence number
    pub request_seq: u64,
    /// Y: Response complexity tier
    pub complexity_tier: u32,
    /// Z: Processing latency depth (ms)
    pub latency_depth: u16,
    /// Guidance: Response confidence
    pub confidence: f32,
}

impl ResponseCoordinate {
    pub const fn new(seq: u64, tier: u32, latency: u16, conf: f32) -> Self {
        Self {
            request_seq: seq,
            complexity_tier: tier,
            latency_depth: latency,
            confidence: conf,
        }
    }

    pub const fn origin() -> Self {
        Self::new(0, 0, 0, 1.0)
    }

    /// Calculate response priority score
    pub fn priority_score(&self) -> f64 {
        let seq_factor = 1.0 / (1.0 + (self.request_seq as f64 * 0.0001));
        let tier_norm = self.complexity_tier as f64 / u32::MAX as f64;
        let latency_norm = (1000.0 - self.latency_depth as f64).max(0.0) / 1000.0;
        
        (seq_factor * 0.2 + tier_norm * 0.5 + latency_norm * 0.3) * self.confidence as f64
    }
}

/// Middleware error with 3D context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralMiddlewareError {
    pub code: u16,
    pub message: String,
    pub coordinate: ResponseCoordinate,
    pub timestamp: i64,
}

impl NeuralMiddlewareError {
    pub fn new(code: u16, message: impl Into<String>) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        
        Self {
            code,
            message: message.into(),
            coordinate: ResponseCoordinate::origin(),
            timestamp: now,
        }
    }

    pub fn with_coordinate(mut self, coord: ResponseCoordinate) -> Self {
        self.coordinate = coord;
        self
    }
}

/// Base response writer trait with 3D spatial tracking
pub trait NeuralResponseWriter: Send + Sync {
    fn write_chunk(&mut self, data: &[u8]) -> io::Result<usize>;
    fn write_error(&mut self, code: u16, message: &str) -> io::Result<usize>;
    fn coordinate(&self) -> ResponseCoordinate;
    fn set_coordinate(&mut self, coord: ResponseCoordinate);
    fn flush(&mut self) -> io::Result<()>;
}

/// Neural chat completion writer with streaming support
pub struct NeuralChatWriter {
    pub stream_enabled: bool,
    pub stream_options: Option<NeuralStreamOptions>,
    pub response_id: String,
    pub tool_call_emitted: bool,
    pub coordinate: ResponseCoordinate,
    pub output_buffer: Vec<u8>,
    pub headers: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralStreamOptions {
    pub include_usage: bool,
}

impl NeuralChatWriter {
    pub fn new(response_id: impl Into<String>) -> Self {
        Self {
            stream_enabled: false,
            stream_options: None,
            response_id: response_id.into(),
            tool_call_emitted: false,
            coordinate: ResponseCoordinate::origin(),
            output_buffer: Vec::new(),
            headers: HashMap::new(),
        }
    }

    pub fn with_streaming(mut self, enabled: bool) -> Self {
        self.stream_enabled = enabled;
        self
    }

    pub fn with_stream_options(mut self, options: NeuralStreamOptions) -> Self {
        self.stream_options = Some(options);
        self
    }

    /// Process chat response chunk
    pub fn process_chat_chunk(&mut self, chunk: &NeuralChatChunk) -> io::Result<Vec<u8>> {
        let json = serde_json::to_vec(chunk)?;
        
        if self.stream_enabled {
            let sse_format = format!("data: {}\n\n", String::from_utf8_lossy(&json));
            self.headers.insert("Content-Type".to_string(), "text/event-stream".to_string());
            Ok(sse_format.into_bytes())
        } else {
            self.headers.insert("Content-Type".to_string(), "application/json".to_string());
            Ok(json)
        }
    }

    /// Emit final stream markers
    pub fn finalize_stream(&mut self) -> io::Result<Vec<u8>> {
        if self.stream_enabled {
            Ok(b"data: [DONE]\n\n".to_vec())
        } else {
            Ok(Vec::new())
        }
    }
}

impl NeuralResponseWriter for NeuralChatWriter {
    fn write_chunk(&mut self, data: &[u8]) -> io::Result<usize> {
        self.output_buffer.extend_from_slice(data);
        Ok(data.len())
    }

    fn write_error(&mut self, code: u16, message: &str) -> io::Result<usize> {
        let error = NeuralMiddlewareError::new(code, message);
        let json = serde_json::to_vec(&error)?;
        self.output_buffer.extend_from_slice(&json);
        Ok(json.len())
    }

    fn coordinate(&self) -> ResponseCoordinate {
        self.coordinate
    }

    fn set_coordinate(&mut self, coord: ResponseCoordinate) {
        self.coordinate = coord;
    }

    fn flush(&mut self) -> io::Result<()> {
        // Write accumulated buffer to output
        self.output_buffer.clear();
        Ok(())
    }
}

/// Neural completion writer (legacy endpoint)
pub struct NeuralCompletionWriter {
    pub stream_enabled: bool,
    pub stream_options: Option<NeuralStreamOptions>,
    pub response_id: String,
    pub coordinate: ResponseCoordinate,
    pub output_buffer: Vec<u8>,
    pub headers: HashMap<String, String>,
}

impl NeuralCompletionWriter {
    pub fn new(response_id: impl Into<String>) -> Self {
        Self {
            stream_enabled: false,
            stream_options: None,
            response_id: response_id.into(),
            coordinate: ResponseCoordinate::origin(),
            output_buffer: Vec::new(),
            headers: HashMap::new(),
        }
    }

    pub fn with_streaming(mut self, enabled: bool) -> Self {
        self.stream_enabled = enabled;
        self
    }
}

impl NeuralResponseWriter for NeuralCompletionWriter {
    fn write_chunk(&mut self, data: &[u8]) -> io::Result<usize> {
        self.output_buffer.extend_from_slice(data);
        Ok(data.len())
    }

    fn write_error(&mut self, code: u16, message: &str) -> io::Result<usize> {
        let error = NeuralMiddlewareError::new(code, message);
        let json = serde_json::to_vec(&error)?;
        self.output_buffer.extend_from_slice(&json);
        Ok(json.len())
    }

    fn coordinate(&self) -> ResponseCoordinate {
        self.coordinate
    }

    fn set_coordinate(&mut self, coord: ResponseCoordinate) {
        self.coordinate = coord;
    }

    fn flush(&mut self) -> io::Result<()> {
        self.output_buffer.clear();
        Ok(())
    }
}

/// Neural model list writer
pub struct NeuralModelListWriter {
    pub coordinate: ResponseCoordinate,
    pub output_buffer: Vec<u8>,
}

impl NeuralModelListWriter {
    pub fn new() -> Self {
        Self {
            coordinate: ResponseCoordinate::origin(),
            output_buffer: Vec::new(),
        }
    }
}

impl NeuralResponseWriter for NeuralModelListWriter {
    fn write_chunk(&mut self, data: &[u8]) -> io::Result<usize> {
        self.output_buffer.extend_from_slice(data);
        Ok(data.len())
    }

    fn write_error(&mut self, code: u16, message: &str) -> io::Result<usize> {
        let error = NeuralMiddlewareError::new(code, message);
        let json = serde_json::to_vec(&error)?;
        self.output_buffer.extend_from_slice(&json);
        Ok(json.len())
    }

    fn coordinate(&self) -> ResponseCoordinate {
        self.coordinate
    }

    fn set_coordinate(&mut self, coord: ResponseCoordinate) {
        self.coordinate = coord;
    }

    fn flush(&mut self) -> io::Result<()> {
        self.output_buffer.clear();
        Ok(())
    }
}

/// Neural model retrieve writer
pub struct NeuralModelRetrieveWriter {
    pub model_id: String,
    pub coordinate: ResponseCoordinate,
    pub output_buffer: Vec<u8>,
}

impl NeuralModelRetrieveWriter {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            coordinate: ResponseCoordinate::origin(),
            output_buffer: Vec::new(),
        }
    }
}

impl NeuralResponseWriter for NeuralModelRetrieveWriter {
    fn write_chunk(&mut self, data: &[u8]) -> io::Result<usize> {
        self.output_buffer.extend_from_slice(data);
        Ok(data.len())
    }

    fn write_error(&mut self, code: u16, message: &str) -> io::Result<usize> {
        let error = NeuralMiddlewareError::new(code, message);
        let json = serde_json::to_vec(&error)?;
        self.output_buffer.extend_from_slice(&json);
        Ok(json.len())
    }

    fn coordinate(&self) -> ResponseCoordinate {
        self.coordinate
    }

    fn set_coordinate(&mut self, coord: ResponseCoordinate) {
        self.coordinate = coord;
    }

    fn flush(&mut self) -> io::Result<()> {
        self.output_buffer.clear();
        Ok(())
    }
}

/// Neural embedding writer
pub struct NeuralEmbeddingWriter {
    pub model_id: String,
    pub encoding_format: String,
    pub coordinate: ResponseCoordinate,
    pub output_buffer: Vec<u8>,
}

impl NeuralEmbeddingWriter {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            encoding_format: "float".to_string(),
            coordinate: ResponseCoordinate::origin(),
            output_buffer: Vec::new(),
        }
    }

    pub fn with_encoding(mut self, format: impl Into<String>) -> Self {
        self.encoding_format = format.into();
        self
    }
}

impl NeuralResponseWriter for NeuralEmbeddingWriter {
    fn write_chunk(&mut self, data: &[u8]) -> io::Result<usize> {
        self.output_buffer.extend_from_slice(data);
        Ok(data.len())
    }

    fn write_error(&mut self, code: u16, message: &str) -> io::Result<usize> {
        let error = NeuralMiddlewareError::new(code, message);
        let json = serde_json::to_vec(&error)?;
        self.output_buffer.extend_from_slice(&json);
        Ok(json.len())
    }

    fn coordinate(&self) -> ResponseCoordinate {
        self.coordinate
    }

    fn set_coordinate(&mut self, coord: ResponseCoordinate) {
        self.coordinate = coord;
    }

    fn flush(&mut self) -> io::Result<()> {
        self.output_buffer.clear();
        Ok(())
    }
}

/// Neural responses API writer
pub struct NeuralResponsesWriter {
    pub stream_converter: Option<NeuralStreamConverter>,
    pub model_id: String,
    pub stream_mode: bool,
    pub response_id: String,
    pub item_id: String,
    pub coordinate: ResponseCoordinate,
    pub output_buffer: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct NeuralStreamConverter {
    pub response_id: String,
    pub item_id: String,
    pub model: String,
    pub sequence: usize,
}

impl NeuralResponsesWriter {
    pub fn new(response_id: impl Into<String>, item_id: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            stream_converter: Some(NeuralStreamConverter {
                response_id: response_id.into(),
                item_id: item_id.into(),
                model: model.into(),
                sequence: 0,
            }),
            model_id: String::new(),
            stream_mode: false,
            response_id: String::new(),
            item_id: String::new(),
            coordinate: ResponseCoordinate::origin(),
            output_buffer: Vec::new(),
        }
    }

    pub fn with_streaming(mut self, enabled: bool) -> Self {
        self.stream_mode = enabled;
        self
    }

    /// Write SSE event
    pub fn write_event(&mut self, event_type: &str, data: &Value) -> io::Result<usize> {
        let json = serde_json::to_string(data)?;
        let sse = format!("event: {}\ndata: {}\n\n", event_type, json);
        self.output_buffer.extend_from_slice(sse.as_bytes());
        Ok(sse.len())
    }
}

impl NeuralResponseWriter for NeuralResponsesWriter {
    fn write_chunk(&mut self, data: &[u8]) -> io::Result<usize> {
        self.output_buffer.extend_from_slice(data);
        Ok(data.len())
    }

    fn write_error(&mut self, code: u16, message: &str) -> io::Result<usize> {
        let error = NeuralMiddlewareError::new(code, message);
        let json = serde_json::to_vec(&error)?;
        self.output_buffer.extend_from_slice(&json);
        Ok(json.len())
    }

    fn coordinate(&self) -> ResponseCoordinate {
        self.coordinate
    }

    fn set_coordinate(&mut self, coord: ResponseCoordinate) {
        self.coordinate = coord;
    }

    fn flush(&mut self) -> io::Result<()> {
        self.output_buffer.clear();
        Ok(())
    }
}

/// Neural image generation writer
pub struct NeuralImageWriter {
    pub coordinate: ResponseCoordinate,
    pub output_buffer: Vec<u8>,
    pub image_data: Option<String>,
}

impl NeuralImageWriter {
    pub fn new() -> Self {
        Self {
            coordinate: ResponseCoordinate::new(0, 500, 100, 0.9),
            output_buffer: Vec::new(),
            image_data: None,
        }
    }
}

impl NeuralResponseWriter for NeuralImageWriter {
    fn write_chunk(&mut self, data: &[u8]) -> io::Result<usize> {
        self.output_buffer.extend_from_slice(data);
        Ok(data.len())
    }

    fn write_error(&mut self, code: u16, message: &str) -> io::Result<usize> {
        let error = NeuralMiddlewareError::new(code, message);
        let json = serde_json::to_vec(&error)?;
        self.output_buffer.extend_from_slice(&json);
        Ok(json.len())
    }

    fn coordinate(&self) -> ResponseCoordinate {
        self.coordinate
    }

    fn set_coordinate(&mut self, coord: ResponseCoordinate) {
        self.coordinate = coord;
    }

    fn flush(&mut self) -> io::Result<()> {
        self.output_buffer.clear();
        Ok(())
    }
}

/// Neural audio transcription writer
pub struct NeuralTranscriptionWriter {
    pub response_format: String,
    pub accumulated_text: String,
    pub coordinate: ResponseCoordinate,
    pub output_buffer: Vec<u8>,
}

impl NeuralTranscriptionWriter {
    pub fn new() -> Self {
        Self {
            response_format: "json".to_string(),
            accumulated_text: String::new(),
            coordinate: ResponseCoordinate::new(0, 300, 150, 0.85),
            output_buffer: Vec::new(),
        }
    }

    pub fn with_format(mut self, format: impl Into<String>) -> Self {
        self.response_format = format.into();
        self
    }

    /// Accumulate transcription text
    pub fn accumulate(&mut self, text: &str) {
        self.accumulated_text.push_str(text);
    }

    /// Get final transcription
    pub fn finalize(&self) -> String {
        self.accumulated_text.trim().to_string()
    }
}

impl NeuralResponseWriter for NeuralTranscriptionWriter {
    fn write_chunk(&mut self, data: &[u8]) -> io::Result<usize> {
        self.output_buffer.extend_from_slice(data);
        Ok(data.len())
    }

    fn write_error(&mut self, code: u16, message: &str) -> io::Result<usize> {
        let error = NeuralMiddlewareError::new(code, message);
        let json = serde_json::to_vec(&error)?;
        self.output_buffer.extend_from_slice(&json);
        Ok(json.len())
    }

    fn coordinate(&self) -> ResponseCoordinate {
        self.coordinate
    }

    fn set_coordinate(&mut self, coord: ResponseCoordinate) {
        self.coordinate = coord;
    }

    fn flush(&mut self) -> io::Result<()> {
        self.output_buffer.clear();
        Ok(())
    }
}

/// Chat completion chunk structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralChatChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<NeuralChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<NeuralUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralChunkChoice {
    pub index: i32,
    pub delta: NeuralMessageDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralMessageDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<NeuralToolCallDelta>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralToolCallDelta {
    pub index: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<NeuralFunctionDelta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralFunctionDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

/// Usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralUsage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
}

/// Middleware request context with 3D tracking
#[derive(Debug, Clone)]
pub struct NeuralRequestContext {
    pub request_id: String,
    pub endpoint: String,
    pub start_time: i64,
    pub coordinate: ResponseCoordinate,
    pub headers: HashMap<String, String>,
}

impl NeuralRequestContext {
    pub fn new(request_id: impl Into<String>, endpoint: impl Into<String>) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        
        Self {
            request_id: request_id.into(),
            endpoint: endpoint.into(),
            start_time: now,
            coordinate: ResponseCoordinate::origin(),
            headers: HashMap::new(),
        }
    }

    /// Calculate latency in milliseconds
    pub fn latency_ms(&self) -> u16 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        ((now - self.start_time) * 1000).min(u16::MAX as i64) as u16
    }

    /// Update coordinate with current latency
    pub fn update_coordinate(&mut self, seq: u64, tier: u32) {
        self.coordinate = ResponseCoordinate::new(seq, tier, self.latency_ms(), 0.95);
    }
}

/// Middleware builder for constructing OpenAI-compatible endpoints
pub struct NeuralMiddlewareBuilder {
    request_seq: Arc<Mutex<u64>>,
}

impl NeuralMiddlewareBuilder {
    pub fn new() -> Self {
        Self {
            request_seq: Arc::new(Mutex::new(0)),
        }
    }

    /// Get next sequence number
    pub fn next_seq(&self) -> u64 {
        let mut seq = self.request_seq.lock().unwrap();
        *seq += 1;
        *seq
    }

    /// Create chat middleware context
    pub fn create_chat_context(&self) -> NeuralRequestContext {
        let seq = self.next_seq();
        let mut ctx = NeuralRequestContext::new(
            format!("chatcmpl_{}", seq),
            "/v1/chat/completions"
        );
        ctx.update_coordinate(seq, 100);
        ctx
    }

    /// Create completion middleware context
    pub fn create_completion_context(&self) -> NeuralRequestContext {
        let seq = self.next_seq();
        let mut ctx = NeuralRequestContext::new(
            format!("cmpl_{}", seq),
            "/v1/completions"
        );
        ctx.update_coordinate(seq, 80);
        ctx
    }

    /// Create embeddings middleware context
    pub fn create_embedding_context(&self) -> NeuralRequestContext {
        let seq = self.next_seq();
        let mut ctx = NeuralRequestContext::new(
            format!("emb_{}", seq),
            "/v1/embeddings"
        );
        ctx.update_coordinate(seq, 50);
        ctx
    }

    /// Create responses middleware context
    pub fn create_responses_context(&self) -> NeuralRequestContext {
        let seq = self.next_seq();
        let mut ctx = NeuralRequestContext::new(
            format!("resp_{}", seq),
            "/v1/responses"
        );
        ctx.update_coordinate(seq, 150);
        ctx
    }
}

impl Default for NeuralMiddlewareBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Decompression configuration
pub const MAX_DECOMPRESSED_SIZE: usize = 20 * 1024 * 1024; // 20MB

/// Compression handler trait
pub trait NeuralCompressionHandler {
    fn decompress_body(&self, body: &[u8], encoding: &str) -> io::Result<Vec<u8>>;
}

/// Default compression handler (no-op)
pub struct NeuralDefaultCompressor;

impl NeuralCompressionHandler for NeuralDefaultCompressor {
    fn decompress_body(&self, body: &[u8], _encoding: &str) -> io::Result<Vec<u8>> {
        // No compression, return as-is
        if body.len() > MAX_DECOMPRESSED_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Body exceeds maximum decompressed size"
            ));
        }
        Ok(body.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_response_coordinate() {
        let coord = ResponseCoordinate::new(100, 500, 50, 0.95);
        assert_eq!(coord.request_seq, 100);
        assert!(coord.priority_score() > 0.0);
    }

    #[test]
    fn test_middleware_error() {
        let error = NeuralMiddlewareError::new(400, "Bad request")
            .with_coordinate(ResponseCoordinate::origin());
        assert_eq!(error.code, 400);
        assert_eq!(error.message, "Bad request");
    }

    #[test]
    fn test_chat_writer() {
        let mut writer = NeuralChatWriter::new("chat_123")
            .with_streaming(true)
            .with_stream_options(NeuralStreamOptions { include_usage: true });
        
        assert!(writer.stream_enabled);
        assert!(writer.stream_options.is_some());
    }

    #[test]
    fn test_request_context() {
        let mut ctx = NeuralRequestContext::new("req_1", "/v1/chat");
        ctx.update_coordinate(1, 100);
        
        assert_eq!(ctx.endpoint, "/v1/chat");
        assert_eq!(ctx.coordinate.request_seq, 1);
        assert_eq!(ctx.coordinate.complexity_tier, 100);
    }

    #[test]
    fn test_middleware_builder() {
        let builder = NeuralMiddlewareBuilder::new();
        
        let seq1 = builder.next_seq();
        let seq2 = builder.next_seq();
        assert!(seq2 > seq1);
        
        let ctx = builder.create_chat_context();
        assert!(ctx.request_id.starts_with("chatcmpl_"));
    }

    #[test]
    fn test_chat_chunk() {
        let chunk = NeuralChatChunk {
            id: "chunk_1".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 1234567890,
            model: "gpt-4".to_string(),
            choices: vec![NeuralChunkChoice {
                index: 0,
                delta: NeuralMessageDelta {
                    role: Some("assistant".to_string()),
                    content: Some("Hello".to_string()),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };
        
        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(chunk.choices[0].delta.content, Some("Hello".to_string()));
    }

    #[test]
    fn test_embedding_writer() {
        let mut writer = NeuralEmbeddingWriter::new("text-embedding-3")
            .with_encoding("base64");
        
        assert_eq!(writer.model_id, "text-embedding-3");
        assert_eq!(writer.encoding_format, "base64");
        
        writer.write_chunk(b"test").unwrap();
        assert_eq!(writer.output_buffer.len(), 4);
    }

    #[test]
    fn test_transcription_writer() {
        let mut writer = NeuralTranscriptionWriter::new()
            .with_format("text");
        
        writer.accumulate("Hello world");
        assert_eq!(writer.finalize(), "Hello world");
    }

    #[test]
    fn test_responses_writer() {
        let mut writer = NeuralResponsesWriter::new("resp_1", "item_1", "gpt-4")
            .with_streaming(true);
        
        assert!(writer.stream_mode);
        
        let data = serde_json::json!({"test": true});
        let written = writer.write_event("test.event", &data).unwrap();
        assert!(written > 0);
    }

    #[test]
    fn test_usage_stats() {
        let usage = NeuralUsage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
        };
        
        assert_eq!(usage.total_tokens, 150);
    }

    #[test]
    fn test_compression_handler() {
        let compressor = NeuralDefaultCompressor;
        let body = b"test body";
        
        let result = compressor.decompress_body(body, "none");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), body.to_vec());
    }
}
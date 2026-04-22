use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::fs::ggml::pronax_ggml_types::SpatialTensorMetadata;

/// 3D spatial context for Anthropic middleware
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnthropicSpatialContext {
    /// Width - request complexity
    pub request_width: u32,
    /// Height - response depth  
    pub response_depth: u32,
    /// Depth - processing layers
    pub processing_depth: u32,
    /// Guidance scale
    pub guidance: f32,
}

impl AnthropicSpatialContext {
    pub const fn new(width: u32, height: u32, depth: u32, guidance: f32) -> Self {
        Self {
            request_width: width,
            response_depth: height,
            processing_depth: depth,
            guidance,
        }
    }

    pub const fn standard() -> Self {
        Self::new(1024, 512, 128, 1.0)
    }

    pub const fn high_capacity() -> Self {
        Self::new(2048, 1024, 256, 1.5)
    }

    pub fn to_metadata(&self) -> SpatialTensorMetadata {
        SpatialTensorMetadata::new(self.request_width, self.response_depth, self.processing_depth)
    }
}

impl Default for AnthropicSpatialContext {
    fn default() -> Self {
        Self::standard()
    }
}

impl fmt::Display for AnthropicSpatialContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AnthropicContext[{}x{}x{}@{}]",
            self.request_width, self.response_depth, self.processing_depth, self.guidance
        )
    }
}

/// Content block types for Anthropic messages
#[derive(Debug, Clone, PartialEq)]
pub enum ContentBlock {
    Text { text: String },
    ServerToolUse { id: String, name: String, input: String },
    WebSearchToolResult { tool_use_id: String, content: String },
    Image { source: ImageSource },
}

/// Image source for multimodal content
#[derive(Debug, Clone, PartialEq)]
pub struct ImageSource {
    pub media_type: String,
    pub data: String, // base64
}

/// Message for Anthropic API
#[derive(Debug, Clone, PartialEq)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: Vec<ContentBlock>,
}

/// Tool definition
#[derive(Debug, Clone, PartialEq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// Messages request
#[derive(Debug, Clone, PartialEq)]
pub struct MessagesRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    pub max_tokens: i32,
    pub stream: bool,
    pub tools: Vec<ToolDefinition>,
    pub system: Option<String>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    /// 3D spatial context
    pub spatial_context: AnthropicSpatialContext,
}

impl MessagesRequest {
    pub fn new(model: impl Into<String>, max_tokens: i32) -> Self {
        Self {
            model: model.into(),
            messages: Vec::new(),
            max_tokens,
            stream: false,
            tools: Vec::new(),
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            spatial_context: AnthropicSpatialContext::standard(),
        }
    }

    pub fn with_spatial(mut self, ctx: AnthropicSpatialContext) -> Self {
        self.spatial_context = ctx;
        self
    }
}

/// Usage statistics
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TokenUsage {
    pub input_tokens: i32,
    pub output_tokens: i32,
    /// 3D spatial token distribution
    pub spatial_dist: (f32, f32, f32),
}

impl TokenUsage {
    pub fn total(&self) -> i32 {
        self.input_tokens + self.output_tokens
    }

    pub fn with_spatial(mut self, x: f32, y: f32, z: f32) -> Self {
        self.spatial_dist = (x, y, z);
        self
    }
}

/// Messages response
#[derive(Debug, Clone, PartialEq)]
pub struct MessagesResponse {
    pub id: String,
    pub response_type: String,
    pub role: String,
    pub model: String,
    pub content: Vec<ContentBlock>,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: TokenUsage,
    /// 3D spatial metadata
    pub spatial_metadata: SpatialTensorMetadata,
}

/// Streaming events
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    MessageStart { message: MessagesResponse },
    ContentBlockStart { index: i32, content_block: ContentBlock },
    ContentBlockDelta { index: i32, delta: ContentDelta },
    ContentBlockStop { index: i32 },
    MessageDelta { stop_reason: Option<String>, usage: TokenUsage },
    MessageStop,
    Error { error: ApiError },
}

/// Content delta for streaming
#[derive(Debug, Clone, PartialEq)]
pub struct ContentDelta {
    pub delta_type: String,
    pub text: Option<String>,
}

/// API Error
#[derive(Debug, Clone, PartialEq)]
pub struct ApiError {
    pub error_type: String,
    pub message: String,
    pub code: Option<String>,
}

impl ApiError {
    pub fn new(error_type: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            error_type: error_type.into(),
            message: message.into(),
            code: None,
        }
    }

    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    pub fn invalid_request(msg: impl Into<String>) -> Self {
        Self::new("invalid_request_error", msg).with_code("400")
    }

    pub fn not_found(msg: impl Into<String>) -> Self {
        Self::new("not_found_error", msg).with_code("404")
    }

    pub fn api_error(msg: impl Into<String>) -> Self {
        Self::new("api_error", msg).with_code("500")
    }
}

/// Web search result
#[derive(Debug, Clone, PartialEq)]
pub struct WebSearchResult {
    pub title: String,
    pub url: String,
    pub content: String,
    /// 3D spatial relevance score
    pub relevance: f32,
}

/// Tool call from model
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub tool_type: String,
    pub function: FunctionCall,
}

/// Function call details
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: serde_json::Map<String, serde_json::Value>,
}

/// Titan Anthropic response writer with 3D spatial support
pub struct TitanAnthropicWriter {
    pub stream_mode: bool,
    pub message_id: String,
    pub spatial_context: AnthropicSpatialContext,
    pub usage_stats: TokenUsage,
    content_buffer: Vec<ContentBlock>,
    stream_index: i32,
}

impl TitanAnthropicWriter {
    pub fn new(message_id: impl Into<String>, stream: bool) -> Self {
        Self {
            stream_mode: stream,
            message_id: message_id.into(),
            spatial_context: AnthropicSpatialContext::standard(),
            usage_stats: TokenUsage::default(),
            content_buffer: Vec::new(),
            stream_index: 0,
        }
    }

    pub fn with_spatial(mut self, ctx: AnthropicSpatialContext) -> Self {
        self.spatial_context = ctx;
        self
    }

    /// Convert internal response to Anthropic format
    pub fn convert_response(
        &self,
        content: impl Into<String>,
        tool_calls: Vec<ToolCall>,
        stop_reason: Option<String>,
    ) -> MessagesResponse {
        let mut blocks = vec![ContentBlock::Text { text: content.into() }];
        
        // Add tool calls as content blocks
        for tc in tool_calls {
            blocks.push(ContentBlock::ServerToolUse {
                id: tc.id.clone(),
                name: tc.function.name.clone(),
                input: serde_json::to_string(&tc.function.arguments).unwrap_or_default(),
            });
        }

        MessagesResponse {
            id: self.message_id.clone(),
            response_type: "message".to_string(),
            role: "assistant".to_string(),
            model: String::new(),
            content: blocks,
            stop_reason,
            stop_sequence: None,
            usage: self.usage_stats,
            spatial_metadata: self.spatial_context.to_metadata(),
        }
    }

    /// Create streaming events
    pub fn create_stream_events(&self, content: impl Into<String>) -> Vec<StreamEvent> {
        let text = content.into();
        let mut events = Vec::new();

        // Message start
        events.push(StreamEvent::MessageStart {
            message: MessagesResponse {
                id: self.message_id.clone(),
                response_type: "message".to_string(),
                role: "assistant".to_string(),
                model: String::new(),
                content: vec![],
                stop_reason: None,
                stop_sequence: None,
                usage: self.usage_stats,
                spatial_metadata: self.spatial_context.to_metadata(),
            },
        });

        // Content block start
        events.push(StreamEvent::ContentBlockStart {
            index: 0,
            content_block: ContentBlock::Text { text: String::new() },
        });

        // Content delta
        events.push(StreamEvent::ContentBlockDelta {
            index: 0,
            delta: ContentDelta {
                delta_type: "text_delta".to_string(),
                text: Some(text),
            },
        });

        // Content block stop
        events.push(StreamEvent::ContentBlockStop { index: 0 });

        // Message delta
        events.push(StreamEvent::MessageDelta {
            stop_reason: Some("stop".to_string()),
            usage: self.usage_stats,
        });

        // Message stop
        events.push(StreamEvent::MessageStop);

        events
    }

    /// Update usage statistics
    pub fn update_usage(&mut self, input: i32, output: i32) {
        self.usage_stats.input_tokens = input;
        self.usage_stats.output_tokens = output;
    }
}

/// Web search loop handler with 3D spatial tracking
pub struct NeuralWebSearchHandler {
    max_loops: usize,
    spatial_context: AnthropicSpatialContext,
    loop_count: usize,
    accumulated_usage: TokenUsage,
    tool_use_history: Vec<String>,
}

impl NeuralWebSearchHandler {
    pub const MAX_WEB_SEARCH_LOOPS: usize = 3;

    pub fn new() -> Self {
        Self {
            max_loops: Self::MAX_WEB_SEARCH_LOOPS,
            spatial_context: AnthropicSpatialContext::standard(),
            loop_count: 0,
            accumulated_usage: TokenUsage::default(),
            tool_use_history: Vec::new(),
        }
    }

    pub fn with_spatial(mut self, ctx: AnthropicSpatialContext) -> Self {
        self.spatial_context = ctx;
        self
    }

    /// Execute web search with spatial tracking
    pub fn execute_search(
        &mut self,
        query: impl Into<String>,
    ) -> Result<Vec<WebSearchResult>, ApiError> {
        let query_str = query.into();
        
        // Simulate web search (in real impl, would call actual search API)
        let results = vec![
            WebSearchResult {
                title: format!("Search result for: {}", query_str),
                url: "https://example.com/result".to_string(),
                content: "Search content...".to_string(),
                relevance: 0.95,
            }
        ];

        // Update spatial tracking
        self.loop_count += 1;
        let x = (self.loop_count as f32 / self.max_loops as f32) * self.spatial_context.request_width as f32;
        let y = query_str.len() as f32 / 100.0 * self.spatial_context.response_depth as f32;
        let z = (self.loop_count % self.spatial_context.processing_depth as usize) as f32;
        
        self.accumulated_usage = self.accumulated_usage.with_spatial(x, y, z);
        self.tool_use_history.push(query_str);

        Ok(results)
    }

    /// Check if max loops reached
    pub fn should_continue(&self) -> bool {
        self.loop_count < self.max_loops
    }

    /// Build tool result content blocks
    pub fn build_tool_results(&self, results: &[WebSearchResult]) -> Vec<ContentBlock> {
        let mut blocks = Vec::new();
        
        for (idx, result) in results.iter().enumerate() {
            let tool_use_id = format!("toolu_{}_{}", self.message_id_prefix(), idx);
            
            // Tool use block
            blocks.push(ContentBlock::ServerToolUse {
                id: tool_use_id.clone(),
                name: "web_search".to_string(),
                input: format!("{{\"query\": \"{}\"}}", result.title),
            });

            // Tool result block
            blocks.push(ContentBlock::WebSearchToolResult {
                tool_use_id,
                content: format!("Title: {}\nURL: {}\nContent: {}", result.title, result.url, result.content),
            });
        }

        blocks
    }

    fn message_id_prefix(&self) -> String {
        format!("msg_{}", std::process::id())
    }

    /// Get current usage stats
    pub fn usage(&self) -> TokenUsage {
        self.accumulated_usage
    }
}

impl Default for NeuralWebSearchHandler {
    fn default() -> Self {
        Self::new()
    }
}

/// Middleware request transformer
pub struct AnthropicRequestTransformer {
    spatial_context: AnthropicSpatialContext,
    enable_web_search: bool,
    max_tokens_limit: i32,
}

impl AnthropicRequestTransformer {
    pub fn new() -> Self {
        Self {
            spatial_context: AnthropicSpatialContext::standard(),
            enable_web_search: true,
            max_tokens_limit: 4096,
        }
    }

    pub fn with_spatial(mut self, ctx: AnthropicSpatialContext) -> Self {
        self.spatial_context = ctx;
        self
    }

    /// Validate incoming request
    pub fn validate(&self, req: &MessagesRequest) -> Result<(), ApiError> {
        if req.model.is_empty() {
            return Err(ApiError::invalid_request("model is required"));
        }

        if req.max_tokens <= 0 {
            return Err(ApiError::invalid_request("max_tokens must be positive"));
        }

        if req.messages.is_empty() {
            return Err(ApiError::invalid_request("messages are required"));
        }

        if req.max_tokens > self.max_tokens_limit {
            return Err(ApiError::invalid_request(
                format!("max_tokens exceeds limit of {}", self.max_tokens_limit)
            ));
        }

        Ok(())
    }

    /// Transform to internal format
    pub fn transform(&self, req: &MessagesRequest) -> Result<NeuralChatRequest, ApiError> {
        let mut messages = Vec::new();
        
        // Add system message if present
        if let Some(system) = &req.system {
            messages.push(NeuralChatMessage::new("system", system.clone()));
        }

        // Convert messages
        for msg in &req.messages {
            for block in &msg.content {
                match block {
                    ContentBlock::Text { text } => {
                        messages.push(NeuralChatMessage::new(&msg.role, text.clone()));
                    }
                    ContentBlock::Image { source } => {
                        // Handle multimodal input
                        messages.push(NeuralChatMessage::with_image(&msg.role, source.data.clone()));
                    }
                    _ => {}
                }
            }
        }

        Ok(NeuralChatRequest {
            model: req.model.clone(),
            messages,
            stream: req.stream,
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            top_p: req.top_p,
            tools: req.tools.clone(),
            spatial_context: req.spatial_context,
        })
    }

    /// Check if request has web search tool
    pub fn has_web_search_tool(&self, tools: &[ToolDefinition]) -> bool {
        tools.iter().any(|t| t.name == "web_search")
    }

    /// Estimate input tokens with spatial calculation
    pub fn estimate_tokens(&self, req: &MessagesRequest) -> i32 {
        let base_count: usize = req.messages.iter()
            .map(|m| m.content.iter().map(|c| match c {
                ContentBlock::Text { text } => text.len() / 4,
                _ => 100, // Image estimation
            }).sum::<usize>())
            .sum();
        
        // Apply spatial scaling
        let scaled = (base_count as f32 * self.spatial_context.guidance) as i32;
        scaled.max(1)
    }
}

impl Default for AnthropicRequestTransformer {
    fn default() -> Self {
        Self::new()
    }
}

/// Internal chat message format
#[derive(Debug, Clone, PartialEq)]
pub struct NeuralChatMessage {
    pub role: String,
    pub content: String,
    pub images: Vec<String>,
    pub tool_calls: Vec<ToolCall>,
    pub tool_call_id: Option<String>,
}

impl NeuralChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            images: Vec::new(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    pub fn with_image(role: impl Into<String>, image_data: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: String::new(),
            images: vec![image_data.into()],
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }
}

/// Internal chat request
#[derive(Debug, Clone, PartialEq)]
pub struct NeuralChatRequest {
    pub model: String,
    pub messages: Vec<NeuralChatMessage>,
    pub stream: bool,
    pub max_tokens: i32,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub tools: Vec<ToolDefinition>,
    pub spatial_context: AnthropicSpatialContext,
}

/// Titan Anthropic middleware handler
pub struct TitanAnthropicMiddleware {
    transformer: AnthropicRequestTransformer,
    writer: Option<TitanAnthropicWriter>,
    web_search_handler: Option<NeuralWebSearchHandler>,
}

impl TitanAnthropicMiddleware {
    pub fn new() -> Self {
        Self {
            transformer: AnthropicRequestTransformer::new(),
            writer: None,
            web_search_handler: None,
        }
    }

    pub fn with_spatial(mut self, ctx: AnthropicSpatialContext) -> Self {
        self.transformer = self.transformer.with_spatial(ctx);
        self
    }

    /// Process incoming request
    pub fn process_request(
        &mut self,
        req: MessagesRequest,
    ) -> Result<ProcessedRequest, ApiError> {
        // Validate
        self.transformer.validate(&req)?;

        // Transform to internal format
        let chat_req = self.transformer.transform(&req)?;

        // Setup writer
        let message_id = generate_message_id();
        let estimated_tokens = self.transformer.estimate_tokens(&req);
        
        let writer = TitanAnthropicWriter::new(&message_id, req.stream)
            .with_spatial(req.spatial_context);
        self.writer = Some(writer);

        // Setup web search if needed
        if self.transformer.has_web_search_tool(&req.tools) {
            self.web_search_handler = Some(
                NeuralWebSearchHandler::new().with_spatial(req.spatial_context)
            );
        }

        Ok(ProcessedRequest {
            chat_request: chat_req,
            message_id,
            estimated_tokens,
            stream: req.stream,
        })
    }

    /// Process streaming response
    pub fn process_stream_chunk(
        &mut self,
        content: impl Into<String>,
        tool_calls: Vec<ToolCall>,
        usage: TokenUsage,
    ) -> Vec<StreamEvent> {
        if let Some(writer) = &self.writer {
            writer.update_usage(usage.input_tokens, usage.output_tokens);
            
            // Check for web search
            if let Some(handler) = &mut self.web_search_handler {
                for tc in &tool_calls {
                    if tc.function.name == "web_search" {
                        if let Some(query) = tc.function.arguments.get("query") {
                            if let Some(q) = query.as_str() {
                                if let Ok(results) = handler.execute_search(q) {
                                    // Web search executed - will be handled in final response
                                }
                            }
                        }
                    }
                }
            }

            writer.create_stream_events(content)
        } else {
            Vec::new()
        }
    }

    /// Generate final response
    pub fn finalize_response(
        &self,
        content: impl Into<String>,
        stop_reason: Option<String>,
    ) -> MessagesResponse {
        if let Some(writer) = &self.writer {
            let tool_calls = if let Some(handler) = &self.web_search_handler {
                // Include web search results in response
                Vec::new() // Simplified - would include tool calls
            } else {
                Vec::new()
            };

            writer.convert_response(content, tool_calls, stop_reason)
        } else {
            MessagesResponse {
                id: String::new(),
                response_type: "message".to_string(),
                role: "assistant".to_string(),
                model: String::new(),
                content: vec![ContentBlock::Text { text: content.into() }],
                stop_reason,
                stop_sequence: None,
                usage: TokenUsage::default(),
                spatial_metadata: SpatialTensorMetadata::new(0, 0, 0),
            }
        }
    }
}

impl Default for TitanAnthropicMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

/// Processed request output
#[derive(Debug, Clone)]
pub struct ProcessedRequest {
    pub chat_request: NeuralChatRequest,
    pub message_id: String,
    pub estimated_tokens: i32,
    pub stream: bool,
}

/// Generate unique message ID
fn generate_message_id() -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("msg_{:016x}", timestamp)
}

/// Format SSE event
pub fn format_sse_event(event_type: &str, data: &str) -> String {
    format!("event: {}\ndata: {}\n\n", event_type, data)
}

/// Extract query from tool call arguments
pub fn extract_query(args: &serde_json::Map<String, serde_json::Value>) -> Option<String> {
    args.get("query")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Type aliases
pub type AnthropicMiddleware = TitanAnthropicMiddleware;
pub type WebSearchHandler = NeuralWebSearchHandler;
pub type RequestTransformer = AnthropicRequestTransformer;
pub type AnthropicWriter = TitanAnthropicWriter;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_context() {
        let ctx = AnthropicSpatialContext::high_capacity();
        assert_eq!(ctx.request_width, 2048);
        assert_eq!(ctx.guidance, 1.5);
        
        let metadata = ctx.to_metadata();
        assert_eq!(metadata.width, 2048);
    }

    #[test]
    fn test_message_creation() {
        let msg = AnthropicMessage {
            role: "user".to_string(),
            content: vec![ContentBlock::Text { text: "Hello".to_string() }],
        };
        assert_eq!(msg.role, "user");
    }

    #[test]
    fn test_request_validation() {
        let transformer = RequestTransformer::new();
        
        let valid_req = MessagesRequest::new("claude-3", 1000);
        assert!(transformer.validate(&valid_req).is_ok());

        let invalid_req = MessagesRequest::new("", 0);
        assert!(transformer.validate(&invalid_req).is_err());
    }

    #[test]
    fn test_web_search_handler() {
        let mut handler = WebSearchHandler::new()
            .with_spatial(AnthropicSpatialContext::high_capacity());
        
        let results = handler.execute_search("test query").unwrap();
        assert!(!results.is_empty());
        assert!(handler.should_continue());
        
        // Run max loops
        for _ in 0..WebSearchHandler::MAX_WEB_SEARCH_LOOPS {
            let _ = handler.execute_search("query");
        }
        assert!(!handler.should_continue());
    }

    #[test]
    fn test_writer_conversion() {
        let writer = AnthropicWriter::new("msg_123", false);
        let response = writer.convert_response(
            "Hello world",
            Vec::new(),
            Some("stop".to_string()),
        );
        
        assert_eq!(response.id, "msg_123");
        assert_eq!(response.response_type, "message");
    }

    #[test]
    fn test_stream_events() {
        let writer = AnthropicWriter::new("msg_123", true);
        let events = writer.create_stream_events("Hello");
        
        assert!(!events.is_empty());
        assert!(matches!(events[0], StreamEvent::MessageStart { .. }));
    }

    #[test]
    fn test_error_creation() {
        let err = ApiError::invalid_request("test error");
        assert_eq!(err.error_type, "invalid_request_error");
        assert_eq!(err.code, Some("400".to_string()));
    }

    #[test]
    fn test_middleware_processing() {
        let mut middleware = AnthropicMiddleware::new();
        let req = MessagesRequest::new("claude-3", 1000);
        
        let processed = middleware.process_request(req).unwrap();
        assert!(!processed.message_id.is_empty());
        assert_eq!(processed.stream, false);
    }

    #[test]
    fn test_token_estimation() {
        let transformer = RequestTransformer::new();
        let mut req = MessagesRequest::new("claude-3", 1000);
        req.messages.push(AnthropicMessage {
            role: "user".to_string(),
            content: vec![ContentBlock::Text { text: "Hello world".to_string() }],
        });
        
        let tokens = transformer.estimate_tokens(&req);
        assert!(tokens > 0);
    }

    #[test]
    fn test_message_id_generation() {
        let id1 = generate_message_id();
        let id2 = generate_message_id();
        assert_ne!(id1, id2);
        assert!(id1.starts_with("msg_"));
    }

    #[test]
    fn test_extract_query() {
        let mut args = serde_json::Map::new();
        args.insert("query".to_string(), serde_json::json!("search term"));
        
        assert_eq!(extract_query(&args), Some("search term".to_string()));
    }
}
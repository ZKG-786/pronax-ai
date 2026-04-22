
use std::borrow::Cow;
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::str::Chars;
use std::sync::Arc;
use regex::Regex;
use serde::{Deserialize, Serialize};
use unicode_segmentation::UnicodeSegmentation;

/// 3D Spatial parsing coordinate for neural syntax analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SyntaxCoordinate {
    /// X: Character position in stream
    pub stream_offset: u64,
    /// Y: Nesting depth level
    pub nesting_depth: u16,
    /// Z: Semantic complexity tier (0=simple, max=complex)
    pub semantic_tier: u16,
    /// Guidance: Parsing confidence score
    pub confidence: f32,
}

impl SyntaxCoordinate {
    pub const fn new(offset: u64, depth: u16, tier: u16, conf: f32) -> Self {
        Self {
            stream_offset: offset,
            nesting_depth: depth,
            semantic_tier: tier,
            confidence: conf,
        }
    }

    pub const fn origin() -> Self {
        Self::new(0, 0, 0, 1.0)
    }

    /// Advance coordinate by character count
    pub fn advance(&mut self, chars: u64) {
        self.stream_offset += chars;
    }

    /// Calculate parsing complexity score
    pub fn complexity_score(&self) -> f64 {
        let offset_factor = (self.stream_offset as f64 / 10000.0).min(1.0);
        let depth_factor = self.nesting_depth as f64 / u16::MAX as f64;
        let tier_factor = self.semantic_tier as f64 / u16::MAX as f64;
        
        (offset_factor * 0.1 + depth_factor * 0.4 + tier_factor * 0.5) * self.confidence as f64
    }
}

impl fmt::Display for SyntaxCoordinate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "@{}[L{}-T{}-{:.2}]",
            self.stream_offset, self.nesting_depth, self.semantic_tier, self.confidence)
    }
}

/// Neural parsing state with 3D spatial tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeuralParsePhase {
    /// Searching for message delimiter
    SeekingDelimiter,
    /// Parsing metadata header
    ExtractingMetadata,
    /// Processing content body
    ProcessingBody,
    /// Handling nested structure
    NestedStructure,
}

impl NeuralParsePhase {
    /// Get 3D complexity tier for this phase
    pub fn complexity_tier(&self) -> u16 {
        match self {
            Self::SeekingDelimiter => 10,
            Self::ExtractingMetadata => 50,
            Self::ProcessingBody => 30,
            Self::NestedStructure => 100,
        }
    }

    /// Get confidence multiplier
    pub fn confidence_boost(&self) -> f32 {
        match self {
            Self::SeekingDelimiter => 0.95,
            Self::ExtractingMetadata => 0.85,
            Self::ProcessingBody => 0.90,
            Self::NestedStructure => 0.75,
        }
    }
}

impl fmt::Display for NeuralParsePhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::SeekingDelimiter => "SEEK_DELIM",
            Self::ExtractingMetadata => "EXTRACT_META",
            Self::ProcessingBody => "PROC_BODY",
            Self::NestedStructure => "NESTED",
        };
        write!(f, "{}", name)
    }
}

/// Semantic header extracted from harmony format
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SemanticHeader {
    /// Actor role (assistant, tool, etc.)
    pub actor_role: String,
    /// Communication channel
    pub channel: String,
    /// Target recipient
    pub target: String,
    /// 3D coordinate where header was parsed
    pub parse_location: SyntaxCoordinate,
    /// Raw header content
    pub raw_content: String,
}

impl SemanticHeader {
    /// Check if this is a tool invocation
    pub fn is_tool_call(&self) -> bool {
        self.actor_role == "tool" || !self.target.is_empty()
    }

    /// Check if channel indicates thinking/analysis
    pub fn is_analytical(&self) -> bool {
        matches!(self.channel.as_str(), "analysis" | "thinking" | "reasoning")
    }
}

/// Parser event types emitted during processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NeuralSyntaxEvent {
    /// Message boundary detected
    BoundaryDetected { 
        boundary_type: BoundaryType,
        coordinate: SyntaxCoordinate,
    },
    /// Header successfully parsed
    MetadataExtracted(SemanticHeader),
    /// Content chunk ready
    ContentFragment {
        payload: String,
        channel: String,
        position: SyntaxCoordinate,
    },
    /// Message complete
    TransmissionComplete {
        final_coordinate: SyntaxCoordinate,
        total_chars: u64,
    },
    /// Parser error with context
    ParseAnomaly {
        severity: AnomalySeverity,
        message: String,
        context: String,
        location: SyntaxCoordinate,
    },
}

/// Boundary types for message delimiters
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundaryType {
    Start,
    End,
    HeaderTerminator,
}

/// Severity levels for parsing anomalies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalySeverity {
    Notice,
    Warning,
    Critical,
}

/// Zero-copy token buffer with spatial tracking
pub struct NeuralTokenBuffer {
    /// Internal storage (cow for zero-copy sharing)
    segments: VecDeque<Cow<'static, str>>,
    /// Total character count
    total_len: usize,
    /// Current 3D position
    cursor: SyntaxCoordinate,
}

impl NeuralTokenBuffer {
    pub fn new() -> Self {
        Self {
            segments: VecDeque::new(),
            total_len: 0,
            cursor: SyntaxCoordinate::origin(),
        }
    }

    /// Append text (potentially zero-copy)
    pub fn ingest(&mut self, text: impl Into<Cow<'static, str>>) {
        let cow = text.into();
        self.total_len += cow.len();
        self.segments.push_back(cow);
    }

    /// Get concatenated view (may require allocation)
    pub fn consolidated(&self) -> String {
        self.segments.iter().map(|s| s.as_ref()).collect()
    }

    /// Check if contains pattern (zero-copy search)
    pub fn contains(&self, pattern: &str) -> bool {
        let combined = self.consolidated();
        combined.contains(pattern)
    }

    /// Split at pattern, returning before/after (consumes buffer)
    pub fn bifurcate(&mut self, pattern: &str) -> Option<(String, String)> {
        let combined = self.consolidated();
        if let Some(pos) = combined.find(pattern) {
            let before = combined[..pos].to_string();
            let after = combined[pos + pattern.len()..].to_string();
            self.clear();
            self.ingest(after.clone());
            return Some((before, after));
        }
        None
    }

    /// Clear all segments
    pub fn clear(&mut self) {
        self.segments.clear();
        self.total_len = 0;
    }

    /// Get current length
    pub fn len(&self) -> usize {
        self.total_len
    }

    pub fn is_empty(&self) -> bool {
        self.total_len == 0
    }

    /// Update cursor position
    pub fn advance_cursor(&mut self, chars: u64) {
        self.cursor.advance(chars);
    }
}

impl Default for NeuralTokenBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Core 3D neural syntax parser
pub struct NeuralSyntaxEngine {
    /// Current parsing phase
    phase: NeuralParsePhase,
    /// Active accumulation buffer
    buffer: NeuralTokenBuffer,
    /// Lifetime accumulation for statistics
    stream_total: u64,
    /// Delimiter tags
    config: ParserConfiguration,
    /// Last known coordinate
    last_coordinate: SyntaxCoordinate,
    /// Nesting stack for complex structures
    nesting_stack: Vec<SyntaxCoordinate>,
}

/// Parser configuration with delimiter tags
#[derive(Debug, Clone)]
pub struct ParserConfiguration {
    pub msg_start: String,
    pub msg_end: String,
    pub header_term: String,
    pub channel_tag: String,
    pub constrain_tag: String,
}

impl Default for ParserConfiguration {
    fn default() -> Self {
        Self {
            msg_start: "<|start|>".to_string(),
            msg_end: "<|end|>".to_string(),
            header_term: "<|message|>".to_string(),
            channel_tag: "<|channel|>".to_string(),
            constrain_tag: "<|constrain|>".to_string(),
        }
    }
}

impl NeuralSyntaxEngine {
    /// Create new parser with default configuration
    pub fn new() -> Self {
        Self {
            phase: NeuralParsePhase::SeekingDelimiter,
            buffer: NeuralTokenBuffer::new(),
            stream_total: 0,
            config: ParserConfiguration::default(),
            last_coordinate: SyntaxCoordinate::origin(),
            nesting_stack: Vec::new(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: ParserConfiguration) -> Self {
        Self {
            phase: NeuralParsePhase::SeekingDelimiter,
            buffer: NeuralTokenBuffer::new(),
            stream_total: 0,
            config,
            last_coordinate: SyntaxCoordinate::origin(),
            nesting_stack: Vec::new(),
        }
    }

    /// Get current parsing phase
    pub fn current_phase(&self) -> NeuralParsePhase {
        self.phase
    }

    /// Get stream statistics
    pub fn telemetry(&self) -> ParserTelemetry {
        ParserTelemetry {
            total_chars: self.stream_total,
            buffer_len: self.buffer.len() as u64,
            current_phase: self.phase,
            nesting_depth: self.nesting_stack.len() as u16,
            current_coordinate: self.last_coordinate,
        }
    }

    /// Inject implicit start tag (for assistant messages)
    pub fn inject_implicit_start(&mut self) {
        self.buffer.ingest(self.config.msg_start.clone() + "assistant");
    }

    /// Inject with prefill logic for chat continuity
    pub fn inject_prefill(&mut self, last_role: Option<&str>, last_content: Option<&str>, last_thinking: Option<&str>) {
        if let Some(role) = last_role {
            if role == "assistant" {
                if last_content.map_or(false, |c| !c.is_empty()) {
                    let prefill = format!("{}assistant{}{}final{}{}",
                        self.config.msg_start,
                        self.config.channel_tag,
                        self.config.header_term,
                        self.config.channel_tag,
                        self.config.header_term
                    );
                    self.buffer.ingest(prefill);
                    return;
                } else if last_thinking.map_or(false, |t| !t.is_empty()) {
                    let prefill = format!("{}assistant{}{}analysis{}{}",
                        self.config.msg_start,
                        self.config.channel_tag,
                        self.config.header_term,
                        self.config.channel_tag,
                        self.config.header_term
                    );
                    self.buffer.ingest(prefill);
                    return;
                }
            }
        }
        self.inject_implicit_start();
    }

    /// Process incoming content chunk, returns events
    pub fn process_chunk(&mut self, chunk: &str) -> Vec<NeuralSyntaxEvent> {
        self.buffer.ingest(chunk.to_string());
        self.stream_total += chunk.len() as u64;
        
        let mut events = Vec::new();
        let mut continue_parsing = true;

        while continue_parsing {
            let (new_events, should_continue) = self.digest();
            events.extend(new_events);
            continue_parsing = should_continue;
        }

        events
    }

    /// Internal digestion logic - state machine core
    fn digest(&mut self) -> (Vec<NeuralSyntaxEvent>, bool) {
        match self.phase {
            NeuralParsePhase::SeekingDelimiter => {
                if self.buffer.contains(&self.config.msg_start) {
                    if let Some((before, after)) = self.buffer.bifurcate(&self.config.msg_start) {
                        if !before.is_empty() {
                            events.push(NeuralSyntaxEvent::ParseAnomaly {
                                severity: AnomalySeverity::Warning,
                                message: "Content before message start".to_string(),
                                context: before.clone(),
                                location: self.last_coordinate,
                            });
                        }
                        
                        self.phase = NeuralParsePhase::ExtractingMetadata;
                        self.last_coordinate = SyntaxCoordinate::new(
                            self.stream_total, 0, 
                            self.phase.complexity_tier(),
                            self.phase.confidence_boost()
                        );
                        
                        return (vec![
                            NeuralSyntaxEvent::BoundaryDetected {
                                boundary_type: BoundaryType::Start,
                                coordinate: self.last_coordinate,
                            }
                        ], true);
                    }
                }
                (vec![], false)
            }
            
            NeuralParsePhase::ExtractingMetadata => {
                if self.buffer.contains(&self.config.header_term) {
                    if let Some((header_raw, after)) = self.buffer.bifurcate(&self.config.header_term) {
                        let header = self.parse_semantic_header(&header_raw);
                        self.phase = NeuralParsePhase::ProcessingBody;
                        
                        return (vec![
                            NeuralSyntaxEvent::MetadataExtracted(header)
                        ], true);
                    }
                }
                (vec![], false)
            }
            
            NeuralParsePhase::ProcessingBody => {
                // Check for message end
                if self.buffer.contains(&self.config.msg_end) {
                    if let Some((content, after)) = self.buffer.bifurcate(&self.config.msg_end) {
                        self.phase = NeuralParsePhase::SeekingDelimiter;
                        
                        let mut events = vec![];
                        if !content.is_empty() {
                            events.push(NeuralSyntaxEvent::ContentFragment {
                                payload: content,
                                channel: "final".to_string(),
                                position: self.last_coordinate,
                            });
                        }
                        events.push(NeuralSyntaxEvent::TransmissionComplete {
                            final_coordinate: self.last_coordinate,
                            total_chars: self.stream_total,
                        });
                        
                        return (events, true);
                    }
                }
                
                // Check for partial end tag overlap
                let overlap_len = self.calculate_overlap(&self.buffer.consolidated(), &self.config.msg_end);
                if overlap_len > 0 {
                    let combined = self.buffer.consolidated();
                    let safe_content = &combined[..combined.len() - overlap_len];
                    let remaining = &combined[combined.len() - overlap_len..];
                    
                    self.buffer.clear();
                    self.buffer.ingest(remaining.to_string());
                    
                    if safe_content.is_empty() {
                        return (vec![], false);
                    }
                    return (vec![
                        NeuralSyntaxEvent::ContentFragment {
                            payload: safe_content.to_string(),
                            channel: "partial".to_string(),
                            position: self.last_coordinate,
                        }
                    ], false);
                }
                
                // No end tag, emit all as content
                let content = self.buffer.consolidated();
                if !content.is_empty() {
                    self.buffer.clear();
                    return (vec![
                        NeuralSyntaxEvent::ContentFragment {
                            payload: content,
                            channel: "streaming".to_string(),
                            position: self.last_coordinate,
                        }
                    ], false);
                }
                
                (vec![], false)
            }
            
            NeuralParsePhase::NestedStructure => {
                // Handle nested parsing logic
                (vec![], false)
            }
        }
    }

    /// Parse semantic header from raw string
    fn parse_semantic_header(&self, raw: &str) -> SemanticHeader {
        let mut header = SemanticHeader {
            raw_content: raw.to_string(),
            parse_location: self.last_coordinate,
            ..Default::default()
        };

        // Normalize constrain tag
        let mut processed = raw.replace(&self.config.constrain_tag, 
            &format!(" {}", self.config.constrain_tag));
        processed = processed.trim().to_string();

        // Extract channel
        if let Some(idx) = processed.find(&self.config.channel_tag) {
            let before = &processed[..idx];
            let after = &processed[idx + self.config.channel_tag.len()..];
            
            // Channel name up to whitespace
            let channel_end = after.find(|c: char| c.is_whitespace())
                .unwrap_or(after.len());
            header.channel = after[..channel_end].to_string();
            
            processed = format!("{}{}", before, &after[channel_end..]);
            processed = processed.trim().to_string();
        }

        // Split into tokens
        let tokens: Vec<&str> = processed.split_whitespace().collect();
        
        if tokens.is_empty() {
            return header;
        }

        // First token is role
        let first = tokens[0];
        if first.starts_with("to=") {
            header.target = first[3..].to_string();
            header.actor_role = "tool".to_string();
        } else {
            header.actor_role = first.to_string();
        }

        // Check for recipient in remaining tokens
        if header.target.is_empty() && tokens.len() > 1 {
            if tokens[1].starts_with("to=") {
                header.target = tokens[1][3..].to_string();
            }
        }

        header
    }

    /// Calculate overlap between suffix of text and prefix of delimiter
    fn calculate_overlap(&self, text: &str, delim: &str) -> usize {
        let max_overlap = text.len().min(delim.len());
        for i in (1..=max_overlap).rev() {
            if text.ends_with(&delim[..i]) {
                return i;
            }
        }
        0
    }
}

impl Default for NeuralSyntaxEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Parser telemetry data
#[derive(Debug, Clone, Copy)]
pub struct ParserTelemetry {
    pub total_chars: u64,
    pub buffer_len: u64,
    pub current_phase: NeuralParsePhase,
    pub nesting_depth: u16,
    pub current_coordinate: SyntaxCoordinate,
}

/// High-level message processor with 3D spatial routing
pub struct NeuralMessageRouter {
    /// Core syntax engine
    engine: NeuralSyntaxEngine,
    /// Current message state
    msg_state: MessageRoutingState,
    /// Function name mapping
    fn_registry: Arc<FunctionNameRegistry>,
    /// Tool call accumulator
    tool_buffer: NeuralTokenBuffer,
}

/// Message routing states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageRoutingState {
    Standard,
    Analytical,
    ToolInvocation,
}

/// Function name registry for tool calls
pub struct FunctionNameRegistry {
    user_to_neural: HashMap<String, String>,
    neural_to_user: HashMap<String, String>,
    builtin_fns: Vec<String>,
}

impl FunctionNameRegistry {
    pub fn new() -> Self {
        Self {
            user_to_neural: HashMap::new(),
            neural_to_user: HashMap::new(),
            builtin_fns: vec![
                "browser.open".to_string(),
                "browser.search".to_string(),
                "browser.find".to_string(),
                "python".to_string(),
            ],
        }
    }

    /// Register function name mapping
    pub fn register(&mut self, user_name: &str) -> String {
        if self.builtin_fns.contains(&user_name.to_string()) {
            return user_name.to_string();
        }

        let neural_name = self.sanitize_identifier(user_name);
        
        // Handle duplicates
        let mut unique = neural_name.clone();
        let mut counter = 2;
        while self.neural_to_user.contains_key(&unique) {
            unique = format!("{}_{}", neural_name, counter);
            counter += 1;
        }

        self.user_to_neural.insert(user_name.to_string(), unique.clone());
        self.neural_to_user.insert(unique.clone(), user_name.to_string());
        
        unique
    }

    /// Lookup original from neural name
    pub fn lookup_original(&self, neural_name: &str) -> Option<&String> {
        self.neural_to_user.get(neural_name)
    }

    /// Sanitize to valid identifier
    fn sanitize_identifier(&self, name: &str) -> String {
        let mut result = String::new();
        
        for grapheme in name.graphemes(true) {
            let ch = grapheme.chars().next().unwrap_or('_');
            if ch.is_alphanumeric() || ch == '_' || ch == '$' {
                result.push(ch);
            } else if ch.is_whitespace() || ch == '-' || ch == '.' {
                result.push('_');
            }
            // Skip other characters
        }

        if result.is_empty() {
            return "unnamed".to_string();
        }

        // Prepend underscore if starts with digit
        if result.chars().next().unwrap().is_ascii_digit() {
            result = format!("_{}", result);
        }

        result
    }
}

impl Default for FunctionNameRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Tool call accumulator with state tracking
pub struct ToolCallAccumulator {
    state: ToolAccumulationState,
    buffer: NeuralTokenBuffer,
    current_tool: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolAccumulationState {
    Idle,
    Collecting,
}

impl ToolCallAccumulator {
    pub fn new() -> Self {
        Self {
            state: ToolAccumulationState::Idle,
            buffer: NeuralTokenBuffer::new(),
            current_tool: None,
        }
    }

    pub fn set_tool(&mut self, name: &str) {
        self.current_tool = Some(name.to_string());
        self.state = ToolAccumulationState::Collecting;
    }

    pub fn accumulate(&mut self, content: &str) {
        self.buffer.ingest(content.to_string());
    }

    pub fn drain(&mut self) -> (Option<String>, String) {
        let tool = self.current_tool.take();
        let content = self.buffer.consolidated();
        self.buffer.clear();
        self.state = ToolAccumulationState::Idle;
        (tool, content)
    }
}

impl Default for ToolCallAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

/// Router initialization and processing
impl NeuralMessageRouter {
    /// Create new message router
    pub fn new() -> Self {
        Self {
            engine: NeuralSyntaxEngine::new(),
            msg_state: MessageRoutingState::Standard,
            fn_registry: Arc::new(FunctionNameRegistry::new()),
            tool_buffer: NeuralTokenBuffer::new(),
        }
    }

    /// Initialize with tools and context
    pub fn initialize(
        &mut self,
        tools: Vec<String>,
        last_role: Option<&str>,
        last_content: Option<&str>,
        last_thinking: Option<&str>,
    ) -> Vec<String> {
        // Inject prefill
        self.engine.inject_prefill(last_role, last_content, last_thinking);

        // Register tool names
        let mut processed = Vec::new();
        for tool in tools {
            processed.push(self.fn_registry.register(&tool));
        }

        processed
    }

    /// Process content and route to appropriate outputs
    pub fn route_content(
        &mut self,
        chunk: &str,
        is_final: bool,
    ) -> (String, String, Option<(String, String)>) {
        let events = self.engine.process_chunk(chunk);
        
        let mut standard_output = String::new();
        let mut analytical_output = String::new();
        let mut tool_result: Option<(String, String)> = None;

        for event in events {
            match event {
                NeuralSyntaxEvent::MetadataExtracted(header) => {
                    // Route based on channel
                    match header.channel.as_str() {
                        "analysis" | "thinking" => {
                            if header.is_tool_call() {
                                self.msg_state = MessageRoutingState::ToolInvocation;
                            } else {
                                self.msg_state = MessageRoutingState::Analytical;
                            }
                        }
                        "commentary" => {
                            if header.is_tool_call() {
                                self.msg_state = MessageRoutingState::ToolInvocation;
                            } else {
                                self.msg_state = MessageRoutingState::Standard;
                            }
                        }
                        "final" => {
                            self.msg_state = MessageRoutingState::Standard;
                        }
                        _ => {}
                    }
                }
                
                NeuralSyntaxEvent::ContentFragment { payload, .. } => {
                    match self.msg_state {
                        MessageRoutingState::Standard => standard_output.push_str(&payload),
                        MessageRoutingState::Analytical => analytical_output.push_str(&payload),
                        MessageRoutingState::ToolInvocation => {
                            self.tool_buffer.ingest(payload);
                        }
                    }
                }
                
                NeuralSyntaxEvent::TransmissionComplete { .. } => {
                    self.msg_state = MessageRoutingState::Standard;
                }
                
                _ => {}
            }
        }

        // Process tool calls on final chunk
        if is_final && self.msg_state == MessageRoutingState::ToolInvocation {
            let (tool_name, args) = self.extract_tool_call();
            if let Some(name) = tool_name {
                tool_result = Some((name, args));
            }
        }

        (standard_output, analytical_output, tool_result)
    }

    /// Extract tool call from accumulated buffer
    fn extract_tool_call(&mut self) -> (Option<String>, String) {
        let content = self.tool_buffer.consolidated();
        self.tool_buffer.clear();
        
        // Try to parse as JSON arguments
        // Return tool name if available
        (None, content)
    }

    /// Check if tool support available
    pub fn has_tool_support(&self) -> bool {
        true
    }

    /// Check if thinking support available  
    pub fn has_analytical_support(&self) -> bool {
        true
    }
}

impl Default for NeuralMessageRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syntax_coordinate() {
        let coord = SyntaxCoordinate::new(1000, 5, 50, 0.95);
        assert_eq!(coord.stream_offset, 1000);
        assert!(coord.complexity_score() > 0.0);
    }

    #[test]
    fn test_parse_phases() {
        assert_eq!(NeuralParsePhase::ExtractingMetadata.complexity_tier(), 50);
        assert_eq!(NeuralParsePhase::NestedStructure.confidence_boost(), 0.75);
    }

    #[test]
    fn test_semantic_header() {
        let header = SemanticHeader {
            actor_role: "tool".to_string(),
            channel: "analysis".to_string(),
            target: "functions.calc".to_string(),
            parse_location: SyntaxCoordinate::origin(),
            raw_content: "tool to=functions.calc".to_string(),
        };
        
        assert!(header.is_tool_call());
        assert!(header.is_analytical());
    }

    #[test]
    fn test_token_buffer() {
        let mut buf = NeuralTokenBuffer::new();
        buf.ingest("Hello ".to_string());
        buf.ingest("World".to_string());
        
        assert_eq!(buf.consolidated(), "Hello World");
        assert!(buf.contains("World"));
    }

    #[test]
    fn test_function_registry() {
        let mut reg = FunctionNameRegistry::new();
        
        let neural1 = reg.register("my-function");
        let neural2 = reg.register("my-function");
        
        assert_ne!(neural1, neural2); // Duplicate handling
        assert_eq!(reg.lookup_original(&neural1), Some(&"my-function".to_string()));
    }

    #[test]
    fn test_builtin_function_passthrough() {
        let mut reg = FunctionNameRegistry::new();
        let name = reg.register("browser.search");
        assert_eq!(name, "browser.search");
    }

    #[test]
    fn test_sanitize_identifier() {
        let reg = FunctionNameRegistry::new();
        assert_eq!(reg.sanitize_identifier("my-fn"), "my_fn");
        assert_eq!(reg.sanitize_identifier("123abc"), "_123abc");
        assert_eq!(reg.sanitize_identifier(""), "unnamed");
    }

    #[test]
    fn test_parser_config() {
        let config = ParserConfiguration::default();
        assert_eq!(config.msg_start, "<|start|>");
    }

    #[test]
    fn test_tool_accumulator() {
        let mut acc = ToolCallAccumulator::new();
        acc.set_tool("functions.calc");
        acc.accumulate("{\"x\": 1}");
        
        let (tool, content) = acc.drain();
        assert_eq!(tool, Some("functions.calc".to_string()));
        assert_eq!(content, "{\"x\": 1}");
    }

    #[test]
    fn test_router_initialization() {
        let mut router = NeuralMessageRouter::new();
        let tools = vec!["calc".to_string(), "search".to_string()];
        
        let processed = router.initialize(tools, None, None, None);
        assert_eq!(processed.len(), 2);
    }
}
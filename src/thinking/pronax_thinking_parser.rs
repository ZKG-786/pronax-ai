use std::collections::VecDeque;

use serde::{Deserialize, Serialize};
use tracing::{debug, trace, warn};

// ============================================================================
// 3D SPATIAL THINKING METADATA
// ============================================================================

/// Neural 3D Spatial Thinking Parser Configuration
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpatialThinkingConfig {
    pub thought_width: u32,
    pub thought_height: u32,
    pub thought_depth: u32,
    pub guidance_scale: f32,
    pub enable_3d_parsing: bool,
}

impl Default for SpatialThinkingConfig {
    fn default() -> Self {
        Self {
            thought_width: 768,
            thought_height: 512,
            thought_depth: 256,
            guidance_scale: 0.9,
            enable_3d_parsing: true,
        }
    }
}

impl SpatialThinkingConfig {
    /// Create with custom dimensions
    pub fn with_dims(width: u32, height: u32, depth: u32) -> Self {
        Self {
            thought_width: width,
            thought_height: height,
            thought_depth: depth,
            ..Default::default()
        }
    }
    
    /// Compute thought workspace volume
    pub fn thought_volume(&self) -> u64 {
        self.thought_width as u64 * self.thought_height as u64 * self.thought_depth as u64
    }
}

// ============================================================================
// THINKING STATE
// ============================================================================

/// Thinking parser state machine states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TitanThinkingState {
    /// Looking for opening tag, no non-whitespace seen yet
    LookingForOpening,
    /// Opening tag seen, eating whitespace before thinking content
    ThinkingStartedEatingWhitespace,
    /// In thinking content, haven't seen closing tag yet
    Thinking,
    /// Closing tag seen, eating whitespace after it
    ThinkingDoneEatingWhitespace,
    /// Done with thinking, non-whitespace seen after closing tag
    ThinkingDone,
}

impl TitanThinkingState {
    /// Convert state to display string
    pub fn as_str(&self) -> &'static str {
        match self {
            TitanThinkingState::LookingForOpening => "LookingForOpening",
            TitanThinkingState::ThinkingStartedEatingWhitespace => "ThinkingStartedEatingWhitespace",
            TitanThinkingState::Thinking => "Thinking",
            TitanThinkingState::ThinkingDoneEatingWhitespace => "ThinkingDoneEatingWhitespace",
            TitanThinkingState::ThinkingDone => "ThinkingDone",
        }
    }
}

impl std::fmt::Display for TitanThinkingState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ============================================================================
// THINKING PARSER
// ============================================================================

/// Parses thinking content between opening and closing tags
pub struct TitanThinkingParser {
    /// Current parser state
    pub state: TitanThinkingState,
    /// Opening tag to look for
    pub opening_tag: String,
    /// Closing tag to look for
    pub closing_tag: String,
    /// Accumulator for buffering content
    accumulator: String,
    /// Spatial configuration for 3D parsing
    pub spatial_config: SpatialThinkingConfig,
    /// Thought chunks with 3D coordinates
    pub thought_chunks: Vec<TitanThoughtChunk>,
}

/// Represents a chunk of thought with 3D spatial coordinates
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TitanThoughtChunk {
    pub content: String,
    pub spatial_coords: (u32, u32, u32),
    pub chunk_index: usize,
}

impl TitanThinkingParser {
    /// Default thinking tags (for 3D spatial thinking)
    pub const DEFAULT_OPENING_TAG: &'static str = "<thinking>";
    pub const DEFAULT_CLOSING_TAG: &'static str = "</thinking>";
    
    /// Create new thinking parser
    pub fn new() -> Self {
        Self {
            state: TitanThinkingState::LookingForOpening,
            opening_tag: Self::DEFAULT_OPENING_TAG.to_string(),
            closing_tag: Self::DEFAULT_CLOSING_TAG.to_string(),
            accumulator: String::new(),
            spatial_config: SpatialThinkingConfig::default(),
            thought_chunks: Vec::new(),
        }
    }
    
    /// Create with custom tags
    pub fn with_tags(opening: impl Into<String>, closing: impl Into<String>) -> Self {
        Self {
            state: TitanThinkingState::LookingForOpening,
            opening_tag: opening.into(),
            closing_tag: closing.into(),
            accumulator: String::new(),
            spatial_config: SpatialThinkingConfig::default(),
            thought_chunks: Vec::new(),
        }
    }
    
    /// Create with spatial configuration
    pub fn with_spatial(mut self, config: SpatialThinkingConfig) -> Self {
        self.spatial_config = config;
        self
    }
    
    /// Add content and parse thinking tokens
    /// Returns (thinking_content, remaining_content)
        pub fn add_content(&mut self, content: &str) -> (String, String) {
        self.accumulator.push_str(content);
        
        let mut thinking_parts: Vec<String> = Vec::new();
        let mut remaining_parts: Vec<String> = Vec::new();
        
        let mut keep_looping = true;
        
        // Loop through states until no more unambiguous data
        while keep_looping {
            let (thinking, remaining, should_continue) = self.eat();
            
            if !thinking.is_empty() {
                thinking_parts.push(thinking);
            }
            if !remaining.is_empty() {
                remaining_parts.push(remaining);
            }
            
            keep_looping = should_continue;
        }
        
        let thinking_content = thinking_parts.join("");
        let remaining_content = remaining_parts.join("");
        
        // Store thought chunk with 3D coordinates if 3D parsing enabled
        if self.spatial_config.enable_3d_parsing && !thinking_content.is_empty() {
            let coords = self.calculate_spatial_coords(self.thought_chunks.len());
            self.thought_chunks.push(TitanThoughtChunk {
                content: thinking_content.clone(),
                spatial_coords: coords,
                chunk_index: self.thought_chunks.len(),
            });
        }
        
        (thinking_content, remaining_content)
    }
    
    /// Eat/parse accumulated content based on current state
    /// Returns (thinking, remaining, should_continue)
        fn eat(&mut self) -> (String, String, bool) {
        match self.state {
            TitanThinkingState::LookingForOpening => {
                let trimmed = self.accumulator.trim_start();
                
                if trimmed.starts_with(&self.opening_tag) {
                    // Found opening tag
                    let after = &trimmed[self.opening_tag.len()..];
                    let after = after.trim_start();
                    
                    self.accumulator.clear();
                    self.accumulator.push_str(after);
                    
                    if after.is_empty() {
                        self.state = TitanThinkingState::ThinkingStartedEatingWhitespace;
                    } else {
                        self.state = TitanThinkingState::Thinking;
                    }
                    
                    return ("".to_string(), "".to_string(), true);
                } else if self.opening_tag.starts_with(trimmed) && !trimmed.is_empty() {
                    // Partial opening tag seen, keep accumulating
                    return ("".to_string(), "".to_string(), false);
                } else if trimmed.is_empty() {
                    // Only whitespace, keep accumulating
                    return ("".to_string(), "".to_string(), false);
                } else {
                    // No opening tag found, thinking skipped
                    self.state = TitanThinkingState::ThinkingDone;
                    let untrimmed = self.accumulator.clone();
                    self.accumulator.clear();
                    return ("".to_string(), untrimmed, false);
                }
            }
            
            TitanThinkingState::ThinkingStartedEatingWhitespace => {
                let trimmed = self.accumulator.trim_start();
                self.accumulator.clear();
                
                if trimmed.is_empty() {
                    return ("".to_string(), "".to_string(), false);
                } else {
                    self.state = TitanThinkingState::Thinking;
                    self.accumulator.push_str(trimmed);
                    return ("".to_string(), "".to_string(), true);
                }
            }
            
            TitanThinkingState::Thinking => {
                let acc = self.accumulator.clone();
                
                if let Some(pos) = acc.find(&self.closing_tag) {
                    // Found closing tag
                    let thinking = &acc[..pos];
                    let remaining = &acc[pos + self.closing_tag.len()..];
                    let remaining = remaining.trim_start();
                    
                    self.accumulator.clear();
                    
                    if remaining.is_empty() {
                        self.state = TitanThinkingState::ThinkingDoneEatingWhitespace;
                    } else {
                        self.state = TitanThinkingState::ThinkingDone;
                    }
                    
                    return (thinking.to_string(), remaining.to_string(), false);
                } else if let Some(overlap_len) = self.calculate_overlap(&acc, &self.closing_tag) {
                    // Partial closing tag overlap detected
                    let thinking = &acc[..acc.len() - overlap_len];
                    let remaining = &acc[acc.len() - overlap_len..];
                    
                    self.accumulator.clear();
                    self.accumulator.push_str(remaining);
                    
                    return (thinking.to_string(), "".to_string(), false);
                } else {
                    // Pure thinking tokens, return them
                    let content = acc;
                    self.accumulator.clear();
                    return (content, "".to_string(), false);
                }
            }
            
            TitanThinkingState::ThinkingDoneEatingWhitespace => {
                let trimmed = self.accumulator.trim_start();
                self.accumulator.clear();
                
                if !trimmed.is_empty() {
                    self.state = TitanThinkingState::ThinkingDone;
                }
                
                return ("".to_string(), trimmed.to_string(), false);
            }
            
            TitanThinkingState::ThinkingDone => {
                let acc = self.accumulator.clone();
                self.accumulator.clear();
                return ("".to_string(), acc, false);
            }
        }
    }
    
    /// Calculate longest overlap between suffix of s and prefix of delim
        fn calculate_overlap(&self, s: &str, delim: &str) -> Option<usize> {
        let max = std::cmp::min(delim.len(), s.len());
        
        for i in (1..=max).rev() {
            if s.ends_with(&delim[..i]) {
                return Some(i);
            }
        }
        
        None
    }
    
    /// Calculate 3D spatial coordinates for thought chunks
    fn calculate_spatial_coords(&self, index: usize) -> (u32, u32, u32) {
        let width = self.spatial_config.thought_width as usize;
        let height = self.spatial_config.thought_height as usize;
        
        let x = (index % width) as u32;
        let y = ((index / width) % height) as u32;
        let z = (index / (width * height)) as u32;
        
        (x, y, z)
    }
    
    /// Reset parser state
    pub fn reset(&mut self) {
        self.state = TitanThinkingState::LookingForOpening;
        self.accumulator.clear();
        self.thought_chunks.clear();
    }
    
    /// Check if parsing is complete
    pub fn is_done(&self) -> bool {
        matches!(self.state, TitanThinkingState::ThinkingDone)
    }
    
    /// Get all thought chunks
    pub fn get_thought_chunks(&self) -> &[TitanThoughtChunk] {
        &self.thought_chunks
    }
    
    /// Get total thought content
    pub fn get_total_thought(&self) -> String {
        self.thought_chunks.iter()
            .map(|c| c.content.clone())
            .collect::<Vec<_>>()
            .join("")
    }
}

impl Default for TitanThinkingParser {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// STANDALONE FUNCTIONS
// ============================================================================

/// Extract thinking content from text
pub fn extract_thinking(content: &str, opening: &str, closing: &str) -> (String, String) {
    let mut parser = TitanThinkingParser::with_tags(opening, closing);
    parser.add_content(content)
}

/// Quick extract using default tags
pub fn extract_thinking_default(content: &str) -> (String, String) {
    let mut parser = TitanThinkingParser::new();
    parser.add_content(content)
}

/// Check if content contains thinking tags
pub fn has_thinking_tags(content: &str, opening: &str, closing: &str) -> bool {
    content.contains(opening) && content.contains(closing)
}

/// Strip thinking tags from content
pub fn strip_thinking(content: &str, opening: &str, closing: &str) -> String {
    let (thinking, remaining) = extract_thinking(content, opening, closing);
    format!("{}{}", thinking, remaining)
}

// ============================================================================
// THINKING ANALYZER
// ============================================================================

/// Analyzes thought patterns with 3D spatial metadata
pub struct TitanThinkingAnalyzer {
    parser: TitanThinkingParser,
    analysis_history: Vec<TitanThoughtAnalysis>,
}

/// Analysis result for a thought chunk
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TitanThoughtAnalysis {
    pub chunk_index: usize,
    pub content_length: usize,
    pub spatial_coords: (u32, u32, u32),
    pub complexity_score: f32,
}

impl TitanThinkingAnalyzer {
    /// Create new analyzer
    pub fn new() -> Self {
        Self {
            parser: TitanThinkingParser::new(),
            analysis_history: Vec::new(),
        }
    }
    
    /// Analyze thinking content
    pub fn analyze(&mut self, content: &str) -> TitanThoughtAnalysis {
        let (thinking, _) = self.parser.add_content(content);
        
        let chunk_index = self.analysis_history.len();
        let coords = self.parser.calculate_spatial_coords(chunk_index);
        
        // Calculate complexity score based on content
        let complexity = self.calculate_complexity(&thinking);
        
        let analysis = TitanThoughtAnalysis {
            chunk_index,
            content_length: thinking.len(),
            spatial_coords: coords,
            complexity_score: complexity,
        };
        
        self.analysis_history.push(analysis.clone());
        analysis
    }
    
    /// Calculate complexity score for thought content
    fn calculate_complexity(&self, content: &str) -> f32 {
        if content.is_empty() {
            return 0.0;
        }
        
        // Factors: length, sentence count, word diversity
        let word_count = content.split_whitespace().count() as f32;
        let sentence_count = content.split(|c| c == '.' || c == '!' || c == '?').count() as f32;
        let unique_words: std::collections::HashSet<_> = content.split_whitespace().collect();
        let diversity = unique_words.len() as f32 / word_count.max(1.0);
        
        // Normalize to 0-1 range
        let length_score = (word_count / 100.0).min(1.0);
        let sentence_score = (sentence_count / 10.0).min(1.0);
        
        (length_score + sentence_score + diversity) / 3.0
    }
    
    /// Get analysis history
    pub fn get_history(&self) -> &[TitanThoughtAnalysis] {
        &self.analysis_history
    }
}

impl Default for TitanThinkingAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_spatial_config_default() {
        let config = SpatialThinkingConfig::default();
        assert_eq!(config.thought_width, 768);
        assert_eq!(config.thought_height, 512);
        assert_eq!(config.thought_depth, 256);
        assert_eq!(config.guidance_scale, 0.9);
    }
    
    #[test]
    fn test_thought_volume() {
        let config = SpatialThinkingConfig::default();
        assert_eq!(config.thought_volume(), 768u64 * 512u64 * 256u64);
    }
    
    #[test]
    fn test_thinking_state_display() {
        assert_eq!(TitanThinkingState::LookingForOpening.to_string(), "LookingForOpening");
        assert_eq!(TitanThinkingState::Thinking.to_string(), "Thinking");
        assert_eq!(TitanThinkingState::ThinkingDone.to_string(), "ThinkingDone");
    }
    
    #[test]
    fn test_parser_creation() {
        let parser = TitanThinkingParser::new();
        assert_eq!(parser.opening_tag, "<thinking>");
        assert_eq!(parser.closing_tag, "</thinking>");
        assert!(matches!(parser.state, TitanThinkingState::LookingForOpening));
    }
    
    #[test]
    fn test_parser_with_custom_tags() {
        let parser = TitanThinkingParser::with_tags("[REASON]", "[/REASON]");
        assert_eq!(parser.opening_tag, "[REASON]");
        assert_eq!(parser.closing_tag, "[/REASON]");
    }
    
    #[test]
    fn test_simple_thinking_extraction() {
        let mut parser = TitanThinkingParser::new();
        let content = "<thinking>This is a thought</thinking>Rest of content";
        
        let (thinking, remaining) = parser.add_content(content);
        
        assert_eq!(thinking, "This is a thought");
        assert_eq!(remaining, "Rest of content");
    }
    
    #[test]
    fn test_thinking_with_whitespace() {
        let mut parser = TitanThinkingParser::new();
        let content = "  <thinking>  Thought content  </thinking>  Remaining  ";
        
        let (thinking, remaining) = parser.add_content(content);
        
        assert_eq!(thinking, "Thought content");
        assert_eq!(remaining, "Remaining");
    }
    
    #[test]
    fn test_no_thinking_tags() {
        let mut parser = TitanThinkingParser::new();
        let content = "No thinking tags here";
        
        let (thinking, remaining) = parser.add_content(content);
        
        assert!(thinking.is_empty());
        assert_eq!(remaining, "No thinking tags here");
    }
    
    #[test]
    fn test_partial_tag_accumulation() {
        let mut parser = TitanThinkingParser::new();
        
        // Send partial opening tag
        let (t1, r1) = parser.add_content("<thi");
        assert!(t1.is_empty() && r1.is_empty());
        
        // Complete the tag
        let (t2, r2) = parser.add_content("nking>Content</thinking>");
        assert_eq!(t2, "Content");
        assert!(r2.is_empty());
    }
    
    #[test]
    fn test_spatial_coords_calculation() {
        let parser = TitanThinkingParser::new();
        
        let coords0 = parser.calculate_spatial_coords(0);
        assert_eq!(coords0, (0, 0, 0));
        
        let coords1 = parser.calculate_spatial_coords(768);
        assert_eq!(coords1, (0, 1, 0));
        
        let coords2 = parser.calculate_spatial_coords(768 * 512);
        assert_eq!(coords2, (0, 0, 1));
    }
    
    #[test]
    fn test_thought_chunk_storage() {
        let mut parser = TitanThinkingParser::new();
        parser.add_content("<thinking>First thought</thinking>");
        parser.add_content("<thinking>Second thought</thinking>");
        
        assert_eq!(parser.thought_chunks.len(), 2);
        assert_eq!(parser.thought_chunks[0].content, "First thought");
        assert_eq!(parser.thought_chunks[1].content, "Second thought");
    }
    
    #[test]
    fn test_calculate_overlap() {
        let parser = TitanThinkingParser::new();
        
        // No overlap
        assert_eq!(parser.calculate_overlap("abc", "def"), None);
        
        // Partial overlap
        assert_eq!(parser.calculate_overlap("tho", "thinking"), Some(3));
        
        // Full overlap (prefix)
        assert_eq!(parser.calculate_overlap("think", "thinking"), Some(5));
    }
    
    #[test]
    fn test_standalone_functions() {
        let content = "<thinking>Deep thought</thinking>Afterthought";
        
        let (t, r) = extract_thinking_default(content);
        assert_eq!(t, "Deep thought");
        assert_eq!(r, "Afterthought");
        
        assert!(has_thinking_tags(content, "<thinking>", "</thinking>"));
        assert!(!has_thinking_tags("No tags", "<thinking>", "</thinking>"));
    }
    
    #[test]
    fn test_thinking_analyzer() {
        let mut analyzer = TitanThinkingAnalyzer::new();
        
        let analysis = analyzer.analyze("<thinking>Complex reasoning with multiple steps.</thinking>");
        
        assert_eq!(analysis.chunk_index, 0);
        assert!(analysis.content_length > 0);
        assert!(analysis.complexity_score > 0.0);
        assert_eq!(analysis.spatial_coords, (0, 0, 0));
    }
    
    #[test]
    fn test_parser_reset() {
        let mut parser = TitanThinkingParser::new();
        parser.add_content("<thinking>Thought</thinking>");
        
        assert!(matches!(parser.state, TitanThinkingState::ThinkingDone));
        assert!(!parser.thought_chunks.is_empty());
        
        parser.reset();
        
        assert!(matches!(parser.state, TitanThinkingState::LookingForOpening));
        assert!(parser.thought_chunks.is_empty());
    }
    
    #[test]
    fn test_multiple_states_in_single_call() {
        let mut parser = TitanThinkingParser::new();
        // This tests the loop in add_content
        let content = "<thinking>Part 1</thinking><thinking>Part 2</thinking>";
        
        let (t, r) = parser.add_content(content);
        
        assert!(t.contains("Part 1") || t.contains("Part 2"));
    }
    
    #[test]
    fn test_thought_chunk_3d_coords() {
        let mut parser = TitanThinkingParser::new();
        parser.add_content("<thinking>First</thinking>");
        parser.add_content("<thinking>Second</thinking>");
        
        let chunk0 = &parser.thought_chunks[0];
        let chunk1 = &parser.thought_chunks[1];
        
        assert_eq!(chunk0.spatial_coords, (0, 0, 0));
        assert_eq!(chunk1.spatial_coords, (1, 0, 0));
    }
    
    #[test]
    fn test_get_total_thought() {
        let mut parser = TitanThinkingParser::new();
        parser.add_content("<thinking>First</thinking>");
        parser.add_content("<thinking>Second</thinking>");
        
        let total = parser.get_total_thought();
        assert!(total.contains("First"));
        assert!(total.contains("Second"));
    }
    
    #[test]
    fn test_empty_content() {
        let mut parser = TitanThinkingParser::new();
        let (t, r) = parser.add_content("");
        
        assert!(t.is_empty());
        assert!(r.is_empty());
    }
    
    #[test]
    fn test_only_whitespace() {
        let mut parser = TitanThinkingParser::new();
        let (t, r) = parser.add_content("   \n\t   ");
        
        assert!(t.is_empty());
        assert!(r.is_empty());
    }
}
use std::collections::{HashMap, HashSet};
use std::io::{self, Write};
use std::sync::OnceLock;

use anyhow::{Context, Result};
use handlebars::Handlebars;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{debug, trace, warn};

// ============================================================================
// 3D SPATIAL TEMPLATE METADATA
// ============================================================================

/// Neural 3D Spatial Chat Template Configuration
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpatialTemplateConfig {
    pub template_width: u32,
    pub template_height: u32,
    pub template_depth: u32,
    pub guidance_scale: f32,
    pub enable_3d_rendering: bool,
    pub spatial_message_layout: bool,
}

impl Default for SpatialTemplateConfig {
    fn default() -> Self {
        Self {
            template_width: 1024,
            template_height: 768,
            template_depth: 256,
            guidance_scale: 0.91,
            enable_3d_rendering: true,
            spatial_message_layout: true,
        }
    }
}

impl SpatialTemplateConfig {
    /// Create with custom dimensions
    pub fn with_dims(width: u32, height: u32, depth: u32) -> Self {
        Self {
            template_width: width,
            template_height: height,
            template_depth: depth,
            ..Default::default()
        }
    }
    
    /// Compute template workspace volume
    pub fn template_volume(&self) -> u64 {
        self.template_width as u64 * self.template_height as u64 * self.template_depth as u64
    }
}

// ============================================================================
// CHAT MESSAGE
// ============================================================================

/// Represents a chat message
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TitanChatMessage {
    pub role: String,
    pub content: String,
    pub thinking: Option<String>,
    pub tool_calls: Vec<TitanToolCall>,
    pub tool_name: Option<String>,
    pub tool_call_id: Option<String>,
    pub images: Vec<String>,
    pub spatial_coords: Option<(u32, u32, u32)>,
}

impl TitanChatMessage {
    /// Create new message
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            thinking: None,
            tool_calls: Vec::new(),
            tool_name: None,
            tool_call_id: None,
            images: Vec::new(),
            spatial_coords: None,
        }
    }
    
    /// Create with 3D spatial coordinates
    pub fn with_spatial(mut self, x: u32, y: u32, z: u32) -> Self {
        self.spatial_coords = Some((x, y, z));
        self
    }
    
    /// Add thinking content
    pub fn with_thinking(mut self, thinking: impl Into<String>) -> Self {
        self.thinking = Some(thinking.into());
        self
    }
    
    /// Check if this is a system message
    pub fn is_system(&self) -> bool {
        self.role == "system"
    }
    
    /// Check if this is a user message
    pub fn is_user(&self) -> bool {
        self.role == "user"
    }
    
    /// Check if this is an assistant message
    pub fn is_assistant(&self) -> bool {
        self.role == "assistant"
    }
    
    /// Check if this is a tool message
    pub fn is_tool(&self) -> bool {
        self.role == "tool"
    }
}

// ============================================================================
// TOOL CALL
// ============================================================================

/// Represents a tool/function call
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TitanToolCall {
    pub id: String,
    pub function: TitanToolFunction,
    pub index: usize,
}

/// Tool function details
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TitanToolFunction {
    pub name: String,
    pub arguments: String,
}

/// Represents a tool definition
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TitanTool {
    pub tool_type: String,
    pub function: TitanToolDefinition,
}

/// Tool function definition
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TitanToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

// ============================================================================
// TEMPLATE VALUES
// ============================================================================

/// Template values for rendering
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TitanTemplateValues {
    pub messages: Vec<TitanChatMessage>,
    pub tools: Vec<TitanTool>,
    pub system: Option<String>,
    pub prompt: Option<String>,
    pub suffix: Option<String>,
    pub response: String,
    pub think: bool,
    pub think_level: Option<String>,
    pub is_think_set: bool,
    pub force_legacy: bool,
}

impl TitanTemplateValues {
    /// Create new template values
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Add a message
    pub fn add_message(mut self, role: impl Into<String>, content: impl Into<String>) -> Self {
        self.messages.push(TitanChatMessage::new(role, content));
        self
    }
    
    /// Set system prompt
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }
    
    /// Set user prompt
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }
}

// ============================================================================
// TEMPLATE REGISTRY
// ============================================================================

/// Named template entry
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TitanNamedTemplate {
    pub name: String,
    pub template: String,
    pub bytes: Vec<u8>,
    pub stop_words: Vec<String>,
}

impl TitanNamedTemplate {
    /// Create new named template
    pub fn new(name: impl Into<String>, template: impl Into<String>) -> Self {
        let template_str = template.into();
        Self {
            name: name.into(),
            template: template_str.clone(),
            bytes: template_str.into_bytes(),
            stop_words: Vec::new(),
        }
    }
    
    /// Create with stop words
    pub fn with_stops(mut self, stops: Vec<String>) -> Self {
        self.stop_words = stops;
        self
    }
}

/// Template registry for managing named templates
pub struct TitanTemplateRegistry {
    templates: HashMap<String, TitanNamedTemplate>,
    spatial_config: SpatialTemplateConfig,
}

impl TitanTemplateRegistry {
    /// Create new registry
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
            spatial_config: SpatialTemplateConfig::default(),
        }
    }
    
    /// Create with spatial config
    pub fn with_spatial(mut self, config: SpatialTemplateConfig) -> Self {
        self.spatial_config = config;
        self
    }
    
    /// Register a template
    pub fn register(&mut self, template: TitanNamedTemplate) {
        self.templates.insert(template.name.clone(), template);
    }
    
    /// Get template by name
    pub fn get(&self, name: &str) -> Option<&TitanNamedTemplate> {
        self.templates.get(name)
    }
    
    /// Find template by fuzzy matching
    pub fn find_by_template(&self, template_str: &str) -> Option<&TitanNamedTemplate> {
        // Simple exact match for now
        // In production, use levenshtein distance
        self.templates.values()
            .find(|t| t.template == template_str)
    }
    
    /// Get all template names
    pub fn template_names(&self) -> Vec<&String> {
        self.templates.keys().collect()
    }
}

impl Default for TitanTemplateRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// CHAT TEMPLATE ENGINE
// ============================================================================

pub struct TitanChatTemplateEngine {
    handlebars: Handlebars<'static>,
    raw_template: String,
    spatial_config: SpatialTemplateConfig,
}

impl TitanChatTemplateEngine {
    /// Default template string
    pub const DEFAULT_TEMPLATE: &'static str = "{{ .Prompt }}";
    
    /// Create new template engine
    pub fn new() -> Self {
        let mut handlebars = Handlebars::new();
        handlebars.set_strict_mode(false);
        
        // Register built-in helpers
        Self::register_helpers(&mut handlebars);
        
        Self {
            handlebars,
            raw_template: Self::DEFAULT_TEMPLATE.to_string(),
            spatial_config: SpatialTemplateConfig::default(),
        }
    }
    
    /// Create with spatial configuration
    pub fn with_spatial(mut self, config: SpatialTemplateConfig) -> Self {
        self.spatial_config = config;
        self
    }
    
    /// Parse template string
        pub fn parse(&mut self, template: impl Into<String>) -> Result<&Self> {
        let template_str = template.into();
        self.raw_template = template_str.clone();
        
        // Register the template
        self.handlebars.register_template_string("chat", &template_str)
            .with_context(|| format!("Failed to parse template: {}", template_str))?;
        
        Ok(self)
    }
    
    /// Execute template with values
        pub fn execute(&self, values: &TitanTemplateValues) -> Result<String> {
        let data = self.prepare_template_data(values)?;
        
        let result = self.handlebars.render("chat", &data)
            .with_context(|| "Failed to render template")?;
        
        Ok(result)
    }
    
    /// Get template as string
    pub fn to_string(&self) -> &str {
        &self.raw_template
    }
    
    /// Check if template contains a string
    pub fn contains(&self, s: &str) -> bool {
        self.raw_template.contains(s)
    }
    
    /// Get template variables
    pub fn vars(&self) -> Vec<String> {
        // Extract variable names from template
        let mut vars = HashSet::new();
        
        // Simple regex-based extraction
        let var_pattern = regex::Regex::new(r"\{\{\s*\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")
            .unwrap();
        
        for cap in var_pattern.captures_iter(&self.raw_template) {
            if let Some(var_name) = cap.get(1) {
                vars.insert(var_name.as_str().to_lowercase());
            }
        }
        
        vars.into_iter().collect()
    }
    
    /// Register built-in template helpers
    fn register_helpers(handlebars: &mut Handlebars) {
        // JSON helper
        handlebars.register_helper("json", Box::new(
            |h: &handlebars::Helper<'_, '_>,
             _: &handlebars::Handlebars<'_>,
             _: &handlebars::Context,
             _: &mut handlebars::RenderContext<'_, '_>,
             out: &mut dyn handlebars::Output| -> handlebars::HelperResult {
                let param = h.param(0).ok_or_else(|| {
                    handlebars::RenderError::new("json helper requires a parameter")
                })?;
                
                let value = param.value();
                let json_str = serde_json::to_string(value).unwrap_or_default();
                out.write(&json_str)?;
                Ok(())
            }
        ));
        
        // Current date helper
        handlebars.register_helper("currentDate", Box::new(
            |_h: &handlebars::Helper<'_, '_>,
             _: &handlebars::Handlebars<'_>,
             _: &handlebars::Context,
             _: &mut handlebars::RenderContext<'_, '_>,
             out: &mut dyn handlebars::Output| -> handlebars::HelperResult {
                let date = chrono::Local::now().format("%Y-%m-%d").to_string();
                out.write(&date)?;
                Ok(())
            }
        ));
        
        // Yesterday date helper
        handlebars.register_helper("yesterdayDate", Box::new(
            |_h: &handlebars::Helper<'_, '_>,
             _: &handlebars::Handlebars<'_>,
             _: &handlebars::Context,
             _: &mut handlebars::RenderContext<'_, '_>,
             out: &mut dyn handlebars::Output| -> handlebars::HelperResult {
                let yesterday = chrono::Local::now() - chrono::Duration::days(1);
                let date = yesterday.format("%Y-%m-%d").to_string();
                out.write(&date)?;
                Ok(())
            }
        ));
    }
    
    /// Prepare template data from values
    fn prepare_template_data(&self, values: &TitanTemplateValues) -> Result<Value> {
        let mut data = serde_json::Map::new();
        
        // Collate messages
        let (system, messages) = self.collate_messages(&values.messages);
        
        // Add basic fields
        data.insert("System".to_string(), Value::String(system));
        data.insert("Messages".to_string(), serde_json::to_value(&messages)?);
        data.insert("Tools".to_string(), serde_json::to_value(&values.tools)?);
        data.insert("Response".to_string(), Value::String(values.response.clone()));
        data.insert("Think".to_string(), Value::Bool(values.think));
        
        if let Some(ref level) = values.think_level {
            data.insert("ThinkLevel".to_string(), Value::String(level.clone()));
        }
        
        data.insert("IsThinkSet".to_string(), Value::Bool(values.is_think_set));
        
        // Add prompt/suffix if provided
        if let Some(ref prompt) = values.prompt {
            data.insert("Prompt".to_string(), Value::String(prompt.clone()));
        }
        
        if let Some(ref suffix) = values.suffix {
            data.insert("Suffix".to_string(), Value::String(suffix.clone()));
        }
        
        // Add spatial metadata if enabled
        if self.spatial_config.enable_3d_rendering {
            let spatial = serde_json::json!({
                "width": self.spatial_config.template_width,
                "height": self.spatial_config.template_height,
                "depth": self.spatial_config.template_depth,
                "guidance": self.spatial_config.guidance_scale,
            });
            data.insert("Spatial".to_string(), spatial);
        }
        
        Ok(Value::Object(data))
    }
    
    /// Collate messages (merge consecutive same-role messages)
    fn collate_messages(&self, messages: &[TitanChatMessage]) -> (String, Vec<TitanChatMessage>) {
        let mut system_parts = Vec::new();
        let mut collated: Vec<TitanChatMessage> = Vec::new();
        
        for msg in messages {
            if msg.is_system() {
                system_parts.push(msg.content.clone());
            }
            
            // Merge consecutive messages of same role (except tool messages)
            if let Some(last) = collated.last_mut() {
                if last.role == msg.role && !msg.is_tool() {
                    last.content.push_str("\n\n");
                    last.content.push_str(&msg.content);
                } else {
                    collated.push(msg.clone());
                }
            } else {
                collated.push(msg.clone());
            }
        }
        
        let system = system_parts.join("\n\n");
        (system, collated)
    }
}

impl Default for TitanChatTemplateEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TEMPLATE TOOLS CONVERSION
// ============================================================================

/// Convert tools to template-compatible format
pub fn convert_tools_for_template(tools: &[TitanTool]) -> Vec<Value> {
    tools.iter().map(|tool| {
        serde_json::json!({
            "type": tool.tool_type,
            "function": {
                "name": tool.function.name,
                "description": tool.function.description,
                "parameters": tool.function.parameters,
            }
        })
    }).collect()
}

/// Convert messages to template-compatible format
pub fn convert_messages_for_template(messages: &[TitanChatMessage]) -> Vec<Value> {
    messages.iter().map(|msg| {
        let mut obj = serde_json::Map::new();
        obj.insert("role".to_string(), Value::String(msg.role.clone()));
        obj.insert("content".to_string(), Value::String(msg.content.clone()));
        
        if let Some(ref thinking) = msg.thinking {
            obj.insert("thinking".to_string(), Value::String(thinking.clone()));
        }
        
        if !msg.tool_calls.is_empty() {
            let calls: Vec<Value> = msg.tool_calls.iter().map(|tc| {
                serde_json::json!({
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                    "index": tc.index,
                })
            }).collect();
            obj.insert("tool_calls".to_string(), Value::Array(calls));
        }
        
        if let Some(ref tool_name) = msg.tool_name {
            obj.insert("tool_name".to_string(), Value::String(tool_name.clone()));
        }
        
        if let Some(ref tool_call_id) = msg.tool_call_id {
            obj.insert("tool_call_id".to_string(), Value::String(tool_call_id.clone()));
        }
        
        Value::Object(obj)
    }).collect()
}

// ============================================================================
// COMMON TEMPLATES
// ============================================================================

/// Get default chat template
pub fn default_chat_template() -> &'static str {
    "{{ .Prompt }}"
}

/// Get Llama 3 chat template
pub fn llama3_chat_template() -> &'static str {
    r#"{{ bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% elif message['role'] == 'user' %}{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% elif message['role'] == 'assistant' %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"#
}

/// Get Mistral chat template
pub fn mistral_chat_template() -> &'static str {
    r#"{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}"#
}

/// Get ChatML template
pub fn chatml_template() -> &'static str {
    r#"{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"#
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_spatial_config_default() {
        let config = SpatialTemplateConfig::default();
        assert_eq!(config.template_width, 1024);
        assert_eq!(config.template_height, 768);
        assert_eq!(config.template_depth, 256);
        assert_eq!(config.guidance_scale, 0.91);
    }
    
    #[test]
    fn test_template_volume() {
        let config = SpatialTemplateConfig::default();
        let volume = config.template_volume();
        assert_eq!(volume, 1024u64 * 768u64 * 256u64);
    }
    
    #[test]
    fn test_chat_message_creation() {
        let msg = TitanChatMessage::new("user", "Hello");
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "Hello");
        assert!(msg.is_user());
        assert!(!msg.is_system());
    }
    
    #[test]
    fn test_chat_message_with_spatial() {
        let msg = TitanChatMessage::new("user", "Hello")
            .with_spatial(10, 20, 30);
        assert_eq!(msg.spatial_coords, Some((10, 20, 30)));
    }
    
    #[test]
    fn test_template_values() {
        let values = TitanTemplateValues::new()
            .add_message("user", "Hello")
            .add_message("assistant", "Hi there")
            .with_system("You are helpful");
        
        assert_eq!(values.messages.len(), 2);
        assert_eq!(values.system, Some("You are helpful".to_string()));
    }
    
    #[test]
    fn test_chat_template_engine() {
        let mut engine = TitanChatTemplateEngine::new();
        engine.parse("{{ .Prompt }}").unwrap();
        
        let values = TitanTemplateValues::new()
            .with_prompt("Hello, world!");
        
        let result = engine.execute(&values).unwrap();
        assert!(result.contains("Hello, world!"));
    }
    
    #[test]
    fn test_template_with_messages() {
        let mut engine = TitanChatTemplateEngine::new();
        engine.parse("{{#each .Messages}}{{@index}}: {{role}} - {{content}}\n{{/each}}").unwrap();
        
        let values = TitanTemplateValues::new()
            .add_message("user", "Hello")
            .add_message("assistant", "Hi!");
        
        let result = engine.execute(&values).unwrap();
        assert!(result.contains("user"));
        assert!(result.contains("assistant"));
    }
    
    #[test]
    fn test_template_contains() {
        let mut engine = TitanChatTemplateEngine::new();
        engine.parse("{{ .Prompt }}{{ .System }}").unwrap();
        
        assert!(engine.contains("Prompt"));
        assert!(engine.contains("System"));
        assert!(!engine.contains("Response"));
    }
    
    #[test]
    fn test_template_vars() {
        let mut engine = TitanChatTemplateEngine::new();
        engine.parse("{{ .Prompt }}{{ .System }}{{ .Messages }}").unwrap();
        
        let vars = engine.vars();
        assert!(vars.contains(&"prompt".to_string()));
        assert!(vars.contains(&"system".to_string()));
        assert!(vars.contains(&"messages".to_string()));
    }
    
    #[test]
    fn test_named_template() {
        let template = TitanNamedTemplate::new("test", "{{ .Prompt }}")
            .with_stops(vec!["[INST]".to_string(), "[/INST]".to_string()]);
        
        assert_eq!(template.name, "test");
        assert_eq!(template.stop_words.len(), 2);
    }
    
    #[test]
    fn test_template_registry() {
        let mut registry = TitanTemplateRegistry::new();
        
        let template = TitanNamedTemplate::new("llama3", llama3_chat_template());
        registry.register(template);
        
        assert!(registry.get("llama3").is_some());
        assert_eq!(registry.template_names().len(), 1);
    }
    
    #[test]
    fn test_tool_conversion() {
        let tools = vec![
            TitanTool {
                tool_type: "function".to_string(),
                function: TitanToolDefinition {
                    name: "test".to_string(),
                    description: "Test tool".to_string(),
                    parameters: serde_json::json!({"type": "object"}),
                },
            }
        ];
        
        let converted = convert_tools_for_template(&tools);
        assert_eq!(converted.len(), 1);
    }
    
    #[test]
    fn test_message_conversion() {
        let messages = vec![
            TitanChatMessage::new("user", "Hello"),
            TitanChatMessage::new("assistant", "Hi!"),
        ];
        
        let converted = convert_messages_for_template(&messages);
        assert_eq!(converted.len(), 2);
    }
    
    #[test]
    fn test_builtin_templates() {
        assert!(!default_chat_template().is_empty());
        assert!(llama3_chat_template().contains("<|start_header_id|>"));
        assert!(mistral_chat_template().contains("[INST]"));
        assert!(chatml_template().contains("<|im_start|>"));
    }
    
    #[test]
    fn test_tool_call_creation() {
        let call = TitanToolCall {
            id: "call_123".to_string(),
            function: TitanToolFunction {
                name: "test".to_string(),
                arguments: "{}".to_string(),
            },
            index: 0,
        };
        
        assert_eq!(call.id, "call_123");
        assert_eq!(call.function.name, "test");
    }
}
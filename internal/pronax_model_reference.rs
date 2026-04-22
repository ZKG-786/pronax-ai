use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use crate::fs::ggml::pronax_ggml_types::SpatialTensorMetadata;

/// Neural model source location with 3D spatial awareness
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum NeuralModelSource {
    /// Source not specified
    #[default]
    Unspecified,
    /// Local filesystem storage
    Local,
    /// Cloud/distributed storage
    Cloud,
    /// 3D spatial-distributed storage (ProNax unique)
    SpatialDistributed {
        shard_x: u32,
        shard_y: u32,
        shard_z: u32,
    },
    /// Edge/on-device storage
    Edge,
    /// Hybrid local + cloud
    Hybrid,
}

impl NeuralModelSource {
    /// Convert to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Unspecified => "auto",
            Self::Local => "local",
            Self::Cloud => "cloud",
            Self::SpatialDistributed { .. } => "spatial",
            Self::Edge => "edge",
            Self::Hybrid => "hybrid",
        }
    }

    /// Check if source requires network
    #[inline]
    pub fn requires_network(&self) -> bool {
        matches!(self, Self::Cloud | Self::Hybrid | Self::SpatialDistributed { .. })
    }

    /// Check if source is local-only
    #[inline]
    pub fn is_local(&self) -> bool {
        matches!(self, Self::Local | Self::Edge)
    }

    /// Get 3D spatial coordinates if applicable
    pub fn spatial_coords(&self) -> Option<(u32, u32, u32)> {
        match self {
            Self::SpatialDistributed { shard_x, shard_y, shard_z } => {
                Some((*shard_x, *shard_y, *shard_z))
            }
            _ => None,
        }
    }

    /// Create spatial distributed source from metadata
    pub fn from_spatial(metadata: &SpatialTensorMetadata) -> Self {
        Self::SpatialDistributed {
            shard_x: metadata.width,
            shard_y: metadata.height,
            shard_z: metadata.depth,
        }
    }
}

impl fmt::Display for NeuralModelSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SpatialDistributed { shard_x, shard_y, shard_z } => {
                write!(f, "spatial[{}x{}x{}]", shard_x, shard_y, shard_z)
            }
            _ => write!(f, "{}", self.as_str()),
        }
    }
}

/// Error types for model reference operations
#[derive(Debug, Clone, PartialEq)]
pub enum NeuralModelRefError {
    EmptyReference,
    ConflictingSourceSuffix { suffix: String },
    InvalidFormat { input: String, reason: String },
    InvalidSpatialCoords { x: u32, y: u32, z: u32 },
    NestedSourceTag { base: String, nested: String },
    UnsupportedSource { source: String },
    MissingNamespace { model: String },
    InvalidCharacters { chars: String },
}

impl fmt::Display for NeuralModelRefError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyReference => {
                write!(f, "Model reference cannot be empty")
            }
            Self::ConflictingSourceSuffix { suffix } => {
                write!(f, "Conflicting source suffix: {}", suffix)
            }
            Self::InvalidFormat { input, reason } => {
                write!(f, "Invalid model reference '{}': {}", input, reason)
            }
            Self::InvalidSpatialCoords { x, y, z } => {
                write!(f, "Invalid 3D spatial coordinates: {}x{}x{}", x, y, z)
            }
            Self::NestedSourceTag { base, nested } => {
                write!(f, "Nested source tag detected: {} contains {}", base, nested)
            }
            Self::UnsupportedSource { source } => {
                write!(f, "Unsupported model source: {}", source)
            }
            Self::MissingNamespace { model } => {
                write!(f, "Model reference missing namespace: {}", model)
            }
            Self::InvalidCharacters { chars } => {
                write!(f, "Invalid characters in model reference: {}", chars)
            }
        }
    }
}

impl std::error::Error for NeuralModelRefError {}

/// Parsed neural model reference with 3D spatial metadata
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NeuralModelReference {
    /// Original unparsed reference string
    pub original: String,
    /// Base model identifier (without source suffix)
    pub base_id: String,
    /// Source location
    pub source: NeuralModelSource,
    /// Model namespace (e.g., "library", "username")
    pub namespace: Option<String>,
    /// Model name
    pub name: String,
    /// Model tag/version
    pub tag: Option<String>,
    /// 3D spatial location for distributed models
    pub spatial_location: Option<SpatialTensorMetadata>,
    /// Whether source was explicitly specified
    pub explicit_source: bool,
}

impl NeuralModelReference {
    /// Create new model reference
    pub fn new(
        original: impl Into<String>,
        base_id: impl Into<String>,
        source: NeuralModelSource,
    ) -> Self {
        let original = original.into();
        let base_id = base_id.into();
        let (namespace, name, tag) = Self::parse_components(&base_id);
        
        let spatial_location = match &source {
            NeuralModelSource::SpatialDistributed { shard_x, shard_y, shard_z } => {
                Some(SpatialTensorMetadata::new(*shard_x, *shard_y, *shard_z))
            }
            _ => None,
        };

        Self {
            original,
            base_id,
            source,
            namespace,
            name,
            tag,
            spatial_location,
            explicit_source: false,
        }
    }

    /// Parse model reference from string
    pub fn parse(raw: &str) -> Result<Self, NeuralModelRefError> {
        let trimmed = raw.trim();
        
        if trimmed.is_empty() {
            return Err(NeuralModelRefError::EmptyReference);
        }

        // Validate characters
        if let Some(invalid) = trimmed.chars().find(|c| {
            !c.is_alphanumeric() && !matches!(c, '/' | ':' | '-' | '_' | '.' | '@')
        }) {
            return Err(NeuralModelRefError::InvalidCharacters {
                chars: invalid.to_string(),
            });
        }

        // Parse source suffix
        let (base_id, source, explicit) = Self::parse_source_suffix(trimmed)?;
        
        // Check for nested source tags
        if explicit {
            let (_, _, nested) = Self::parse_source_suffix(base_id.as_str())?;
            if nested {
                return Err(NeuralModelRefError::NestedSourceTag {
                    base: base_id.clone(),
                    nested: trimmed.to_string(),
                });
            }
        }

        let (namespace, name, tag) = Self::parse_components(&base_id);
        
        let spatial_location = match &source {
            NeuralModelSource::SpatialDistributed { shard_x, shard_y, shard_z } => {
                Some(SpatialTensorMetadata::new(*shard_x, *shard_y, *shard_z))
            }
            _ => None,
        };

        Ok(Self {
            original: trimmed.to_string(),
            base_id,
            source,
            namespace,
            name,
            tag,
            spatial_location,
            explicit_source: explicit,
        })
    }

    /// Parse source suffix from reference
    fn parse_source_suffix(input: &str) -> Result<(String, NeuralModelSource, bool), NeuralModelRefError> {
        // Check for spatial format: model@x,y,z
        if let Some(at_idx) = input.rfind('@') {
            let base = &input[..at_idx];
            let coords_str = &input[at_idx + 1..];
            
            if let Ok((x, y, z)) = Self::parse_spatial_coords(coords_str) {
                return Ok((
                    base.to_string(),
                    NeuralModelSource::SpatialDistributed { shard_x: x, shard_y: y, shard_z: z },
                    true,
                ));
            }
        }

        // Check for standard suffixes
        if let Some(colon_idx) = input.rfind(':') {
            let suffix = input[colon_idx + 1..].trim().to_lowercase();
            let base = input[..colon_idx].to_string();
            
            match suffix.as_str() {
                "cloud" => return Ok((base, NeuralModelSource::Cloud, true)),
                "local" => return Ok((base, NeuralModelSource::Local, true)),
                "edge" => return Ok((base, NeuralModelSource::Edge, true)),
                "hybrid" => return Ok((base, NeuralModelSource::Hybrid, true)),
                _ => {
                    // Check for -cloud suffix in tag
                    if suffix.ends_with("-cloud") && !suffix.contains('/') {
                        let tag_without_cloud = &suffix[..suffix.len() - 6];
                        return Ok((
                            format!("{}:{}", base, tag_without_cloud),
                            NeuralModelSource::Cloud,
                            true,
                        ));
                    }
                }
            }
        }

        // No explicit source
        Ok((input.to_string(), NeuralModelSource::Unspecified, false))
    }

    /// Parse 3D spatial coordinates
    fn parse_spatial_coords(s: &str) -> Result<(u32, u32, u32), NeuralModelRefError> {
        let parts: Vec<&str> = s.split(',').collect();
        if parts.len() != 3 {
            return Err(NeuralModelRefError::InvalidSpatialCoords { x: 0, y: 0, z: 0 });
        }

        let x = parts[0].trim().parse().map_err(|_| NeuralModelRefError::InvalidSpatialCoords { x: 0, y: 0, z: 0 })?;
        let y = parts[1].trim().parse().map_err(|_| NeuralModelRefError::InvalidSpatialCoords { x: 0, y: 0, z: 0 })?;
        let z = parts[2].trim().parse().map_err(|_| NeuralModelRefError::InvalidSpatialCoords { x: 0, y: 0, z: 0 })?;

        Ok((x, y, z))
    }

    /// Parse model components (namespace, name, tag)
    fn parse_components(base_id: &str) -> (Option<String>, String, Option<String>) {
        // Format: namespace/name:tag or name:tag or name
        let parts: Vec<&str> = base_id.rsplitn(2, '/').collect();
        
        let (name_part, namespace) = if parts.len() == 2 {
            (parts[0].to_string(), Some(parts[1].to_string()))
        } else {
            (base_id.to_string(), None)
        };

        // Parse name and tag
        let name_parts: Vec<&str> = name_part.rsplitn(2, ':').collect();
        if name_parts.len() == 2 && !name_parts[0].contains('/') {
            (namespace, name_parts[1].to_string(), Some(name_parts[0].to_string()))
        } else {
            (namespace, name_part, None)
        }
    }

    /// Check if reference has explicit cloud source
    #[inline]
    pub fn is_cloud(&self) -> bool {
        matches!(self.source, NeuralModelSource::Cloud)
    }

    /// Check if reference has explicit local source
    #[inline]
    pub fn is_local(&self) -> bool {
        matches!(self.source, NeuralModelSource::Local)
    }

    /// Check if reference has spatial distributed source
    #[inline]
    pub fn is_spatial(&self) -> bool {
        matches!(self.source, NeuralModelSource::SpatialDistributed { .. })
    }

    /// Check if reference has explicit tag
    #[inline]
    pub fn has_explicit_tag(&self) -> bool {
        self.tag.is_some()
    }

    /// Get full qualified name
    pub fn qualified_name(&self) -> String {
        let mut result = String::new();
        
        if let Some(ns) = &self.namespace {
            result.push_str(ns);
            result.push('/');
        }
        
        result.push_str(&self.name);
        
        if let Some(tag) = &self.tag {
            result.push(':');
            result.push_str(tag);
        }
        
        result
    }

    /// Get canonical form for pulling
    pub fn canonical_pull_name(&self) -> String {
        let base = self.qualified_name();
        
        if self.is_cloud() {
            if self.has_explicit_tag() {
                format!("{}-cloud", base)
            } else {
                format!("{}:cloud", base)
            }
        } else {
            base
        }
    }

    /// Strip source tag and return base
    pub fn strip_source(&self) -> (String, bool) {
        if self.is_cloud() {
            (self.qualified_name(), true)
        } else {
            (self.original.clone(), false)
        }
    }

    /// With spatial coordinates (creates new reference)
    pub fn with_spatial(&self, x: u32, y: u32, z: u32) -> Self {
        let mut new_ref = self.clone();
        new_ref.source = NeuralModelSource::SpatialDistributed { shard_x: x, shard_y: y, shard_z: z };
        new_ref.spatial_location = Some(SpatialTensorMetadata::new(x, y, z));
        new_ref
    }

    /// Get cache key for this reference
    pub fn cache_key(&self) -> String {
        format!("{}:{}", self.base_id, self.source)
    }
}

impl fmt::Display for NeuralModelReference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.qualified_name())?;
        
        if self.explicit_source {
            match &self.source {
                NeuralModelSource::SpatialDistributed { .. } => {
                    // Already included in name
                }
                _ => {
                    write!(f, ":{}", self.source.as_str())?;
                }
            }
        }
        
        Ok(())
    }
}

/// Model reference registry with zero-copy caching
pub struct NeuralModelRegistry {
    references: HashMap<String, Arc<NeuralModelReference>>,
    spatial_index: HashMap<(u32, u32, u32), Vec<String>>,
}

impl NeuralModelRegistry {
    /// Create new empty registry
    pub fn new() -> Self {
        Self {
            references: HashMap::new(),
            spatial_index: HashMap::new(),
        }
    }

    /// Register a model reference
    pub fn register(&mut self, reference: NeuralModelReference) -> Arc<NeuralModelReference> {
        let key = reference.cache_key();
        
        // Add to spatial index if applicable
        if let Some(coords) = reference.source.spatial_coords() {
            self.spatial_index
                .entry(coords)
                .or_default()
                .push(key.clone());
        }
        
        let arc = Arc::new(reference);
        self.references.insert(key, arc.clone());
        arc
    }

    /// Parse and register in one step
    pub fn parse_and_register(&mut self, raw: &str) -> Result<Arc<NeuralModelReference>, NeuralModelRefError> {
        let reference = NeuralModelReference::parse(raw)?;
        Ok(self.register(reference))
    }

    /// Get reference by key
    pub fn get(&self, key: &str) -> Option<Arc<NeuralModelReference>> {
        self.references.get(key).cloned()
    }

    /// Find by spatial coordinates
    pub fn find_by_spatial(&self, x: u32, y: u32, z: u32) -> Vec<Arc<NeuralModelReference>> {
        self.spatial_index
            .get(&(x, y, z))
            .map(|keys| {
                keys.iter()
                    .filter_map(|k| self.get(k))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all cloud references
    pub fn cloud_references(&self) -> Vec<Arc<NeuralModelReference>> {
        self.references
            .values()
            .filter(|r| r.is_cloud())
            .cloned()
            .collect()
    }

    /// Get all local references
    pub fn local_references(&self) -> Vec<Arc<NeuralModelReference>> {
        self.references
            .values()
            .filter(|r| r.is_local())
            .cloned()
            .collect()
    }

    /// Count of registered references
    #[inline]
    pub fn len(&self) -> usize {
        self.references.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.references.is_empty()
    }

    /// Clear all references
    pub fn clear(&mut self) {
        self.references.clear();
        self.spatial_index.clear();
    }

    /// Get statistics
    pub fn stats(&self) -> ModelRegistryStats {
        let total = self.references.len();
        let cloud = self.cloud_references().len();
        let local = self.local_references().len();
        let spatial = self.references.values().filter(|r| r.is_spatial()).count();

        ModelRegistryStats {
            total,
            cloud,
            local,
            spatial,
            unspecified: total - cloud - local - spatial,
        }
    }
}

impl Default for NeuralModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Registry statistics
#[derive(Debug, Clone, Copy)]
pub struct ModelRegistryStats {
    pub total: usize,
    pub cloud: usize,
    pub local: usize,
    pub spatial: usize,
    pub unspecified: usize,
}

impl fmt::Display for ModelRegistryStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ModelRegistryStats {{ total: {}, cloud: {}, local: {}, spatial: {}, auto: {} }}",
            self.total, self.cloud, self.local, self.spatial, self.unspecified
        )
    }
}

/// Helper functions
pub fn is_cloud_source(raw: &str) -> bool {
    NeuralModelReference::parse(raw)
        .map(|r| r.is_cloud())
        .unwrap_or(false)
}

pub fn is_local_source(raw: &str) -> bool {
    NeuralModelReference::parse(raw)
        .map(|r| r.is_local())
        .unwrap_or(false)
}

pub fn strip_cloud_tag(raw: &str) -> (String, bool) {
    NeuralModelReference::parse(raw)
        .map(|r| r.strip_source())
        .unwrap_or_else(|_| (raw.to_string(), false))
}

pub fn normalize_for_pull(raw: &str) -> Result<(String, bool), NeuralModelRefError> {
    let reference = NeuralModelReference::parse(raw)?;
    let is_cloud = reference.is_cloud();
    Ok((reference.canonical_pull_name(), is_cloud))
}

/// Type aliases
pub type ModelSource = NeuralModelSource;
pub type ModelReference = NeuralModelReference;
pub type ModelRefError = NeuralModelRefError;
pub type ModelRegistry = NeuralModelRegistry;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_basic() {
        let reference = ModelReference::parse("llama3.1").unwrap();
        assert_eq!(reference.name, "llama3.1");
        assert_eq!(reference.source, ModelSource::Unspecified);
        assert!(!reference.explicit_source);
    }

    #[test]
    fn test_parse_with_namespace() {
        let reference = ModelReference::parse("library/llama3.1").unwrap();
        assert_eq!(reference.namespace, Some("library".to_string()));
        assert_eq!(reference.name, "llama3.1");
    }

    #[test]
    fn test_parse_with_tag() {
        let reference = ModelReference::parse("llama3.1:latest").unwrap();
        assert_eq!(reference.name, "llama3.1");
        assert_eq!(reference.tag, Some("latest".to_string()));
    }

    #[test]
    fn test_parse_full() {
        let reference = ModelReference::parse("library/llama3.1:8b").unwrap();
        assert_eq!(reference.namespace, Some("library".to_string()));
        assert_eq!(reference.name, "llama3.1");
        assert_eq!(reference.tag, Some("8b".to_string()));
    }

    #[test]
    fn test_parse_cloud_source() {
        let reference = ModelReference::parse("llama3.1:cloud").unwrap();
        assert!(reference.is_cloud());
        assert!(reference.explicit_source);
    }

    #[test]
    fn test_parse_local_source() {
        let reference = ModelReference::parse("llama3.1:local").unwrap();
        assert!(reference.is_local());
        assert!(reference.explicit_source);
    }

    #[test]
    fn test_parse_spatial_source() {
        let reference = ModelReference::parse("llama3.1@2,3,4").unwrap();
        assert!(reference.is_spatial());
        assert_eq!(reference.source.spatial_coords(), Some((2, 3, 4)));
    }

    #[test]
    fn test_qualified_name() {
        let reference = ModelReference::parse("library/llama3.1:8b").unwrap();
        assert_eq!(reference.qualified_name(), "library/llama3.1:8b");
    }

    #[test]
    fn test_canonical_pull_cloud() {
        let reference = ModelReference::parse("llama3.1:cloud").unwrap();
        assert_eq!(reference.canonical_pull_name(), "llama3.1:cloud");
    }

    #[test]
    fn test_canonical_pull_cloud_with_tag() {
        let reference = ModelReference::parse("llama3.1:8b:cloud").unwrap();
        assert_eq!(reference.canonical_pull_name(), "llama3.1:8b-cloud");
    }

    #[test]
    fn test_strip_cloud() {
        let reference = ModelReference::parse("llama3.1:cloud").unwrap();
        let (base, stripped) = reference.strip_source();
        assert_eq!(base, "llama3.1");
        assert!(stripped);
    }

    #[test]
    fn test_empty_reference_error() {
        let result = ModelReference::parse("");
        assert!(matches!(result, Err(ModelRefError::EmptyReference)));
    }

    #[test]
    fn test_nested_source_error() {
        let result = ModelReference::parse("llama3.1:cloud:local");
        assert!(matches!(result, Err(ModelRefError::NestedSourceTag { .. })));
    }

    #[test]
    fn test_invalid_characters_error() {
        let result = ModelReference::parse("llama$3.1");
        assert!(matches!(result, Err(ModelRefError::InvalidCharacters { .. })));
    }

    #[test]
    fn test_source_helpers() {
        assert!(is_cloud_source("llama3.1:cloud"));
        assert!(!is_cloud_source("llama3.1"));
        
        assert!(is_local_source("llama3.1:local"));
        assert!(!is_local_source("llama3.1:cloud"));
    }

    #[test]
    fn test_registry() {
        let mut registry = ModelRegistry::new();
        
        let ref1 = registry.parse_and_register("llama3.1:cloud").unwrap();
        let ref2 = registry.parse_and_register("llama3.2:local").unwrap();
        let ref3 = registry.parse_and_register("model@1,2,3").unwrap();
        
        assert_eq!(registry.len(), 3);
        assert_eq!(registry.cloud_references().len(), 1);
        assert_eq!(registry.local_references().len(), 1);
        
        let spatial = registry.find_by_spatial(1, 2, 3);
        assert_eq!(spatial.len(), 1);
    }

    #[test]
    fn test_with_spatial() {
        let base = ModelReference::parse("llama3.1").unwrap();
        let spatial_ref = base.with_spatial(10, 20, 30);
        
        assert!(spatial_ref.is_spatial());
        assert_eq!(spatial_ref.spatial_location.unwrap().volume(), 10 * 20 * 30);
    }

    #[test]
    fn test_cache_key() {
        let reference = ModelReference::parse("llama3.1:cloud").unwrap();
        assert!(reference.cache_key().contains("llama3.1"));
        assert!(reference.cache_key().contains("cloud"));
    }
}
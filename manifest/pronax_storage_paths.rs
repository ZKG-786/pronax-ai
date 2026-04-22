use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use once_cell::sync::Lazy;
use regex::Regex;

use crate::fs::ggml::pronax_ggml_types::SpatialTensorMetadata;

/// Error types for storage path operations
#[derive(Debug, Clone, PartialEq)]
pub enum NeuralPathError {
    InvalidDigestFormat { digest: String },
    InvalidModelName { name: String },
    PathTraversalAttempt { path: String },
    DirectoryCreationFailed { path: String, reason: String },
    PruneFailed { path: String, reason: String },
    NotADirectory { path: String },
}

impl fmt::Display for NeuralPathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDigestFormat { digest } => {
                write!(f, "Invalid digest format: {}", digest)
            }
            Self::InvalidModelName { name } => {
                write!(f, "Invalid model name: {}", name)
            }
            Self::PathTraversalAttempt { path } => {
                write!(f, "Path traversal attempt detected: {}", path)
            }
            Self::DirectoryCreationFailed { path, reason } => {
                write!(f, "Failed to create directory {}: {}", path, reason)
            }
            Self::PruneFailed { path, reason } => {
                write!(f, "Failed to prune directory {}: {}", path, reason)
            }
            Self::NotADirectory { path } => {
                write!(f, "Not a directory: {}", path)
            }
        }
    }
}

impl std::error::Error for NeuralPathError {}

/// 3D spatial storage layout configuration
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpatialStorageLayout {
    /// Width dimension for storage sharding
    pub width_shard: u32,
    /// Height dimension for storage tiers
    pub height_tiers: u32,
    /// Depth dimension for storage layers
    pub depth_layers: u32,
    /// Whether to use spatial hashing
    pub use_spatial_hash: bool,
}

impl SpatialStorageLayout {
    /// Create new 3D storage layout
    pub const fn new(width: u32, height: u32, depth: u32) -> Self {
        Self {
            width_shard: width,
            height_tiers: height,
            depth_layers: depth,
            use_spatial_hash: true,
        }
    }

    /// Default layout (1x1x1 for flat storage)
    pub const fn flat() -> Self {
        Self::new(1, 1, 1)
    }

    /// Hierarchical layout (4x4x4 for distributed storage)
    pub const fn distributed() -> Self {
        Self::new(4, 4, 4)
    }

    /// Calculate spatial bucket for a digest
    pub fn spatial_bucket(&self, digest: &str) -> (u32, u32, u32) {
        if !self.use_spatial_hash || digest.len() < 12 {
            return (0, 0, 0);
        }

        // Use first 12 chars of digest (48 bits) for 3D bucketing
        let hash_val = u64::from_str_radix(&digest[..12], 16).unwrap_or(0);
        
        let w = ((hash_val >> 32) & 0xFFFF) % self.width_shard;
        let h = ((hash_val >> 16) & 0xFFFF) % self.height_tiers;
        let d = (hash_val & 0xFFFF) % self.depth_layers;
        
        (w as u32, h as u32, d as u32)
    }

    /// Convert to metadata struct
    pub fn to_metadata(&self) -> SpatialTensorMetadata {
        SpatialTensorMetadata::new(self.width_shard, self.height_tiers, self.depth_layers)
    }
}

impl Default for SpatialStorageLayout {
    fn default() -> Self {
        Self::flat()
    }
}

/// Digest validation regex pattern
static DIGEST_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^sha256[:-][0-9a-fA-F]{64}$").expect("Invalid digest regex pattern")
});

/// Titan storage path manager with 3D spatial layout
pub struct TitanStoragePathManager {
    /// Base models directory
    base_path: PathBuf,
    /// 3D spatial layout configuration
    spatial_layout: SpatialStorageLayout,
    /// Cache for resolved paths
    path_cache: HashMap<String, Arc<PathBuf>>,
    /// Manifest subdirectory name
    manifest_dir: String,
    /// Blobs subdirectory name
    blobs_dir: String,
    /// Directory permissions (unix-style)
    dir_permissions: u32,
}

impl TitanStoragePathManager {
    /// Create new path manager with default layout
    pub fn new(base_path: impl AsRef<Path>) -> Self {
        Self::with_layout(base_path, SpatialStorageLayout::default())
    }

    /// Create with custom spatial layout
    pub fn with_layout(base_path: impl AsRef<Path>, layout: SpatialStorageLayout) -> Self {
        Self {
            base_path: base_path.as_ref().to_path_buf(),
            spatial_layout: layout,
            path_cache: HashMap::new(),
            manifest_dir: "manifests".to_string(),
            blobs_dir: "blobs".to_string(),
            dir_permissions: 0o755,
        }
    }

    /// With custom subdirectory names
    pub fn with_directories(
        mut self,
        manifest: impl Into<String>,
        blobs: impl Into<String>,
    ) -> Self {
        self.manifest_dir = manifest.into();
        self.blobs_dir = blobs.into();
        self
    }

    /// Get base path
    #[inline]
    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    /// Get spatial layout
    #[inline]
    pub fn spatial_layout(&self) -> &SpatialStorageLayout {
        &self.spatial_layout
    }

    /// Get manifests directory path
    pub fn manifests_path(&self) -> Result<PathBuf, NeuralPathError> {
        let path = self.base_path.join(&self.manifest_dir);
        
        // Validate path (prevent traversal)
        if !self.is_path_safe(&path) {
            return Err(NeuralPathError::PathTraversalAttempt {
                path: path.display().to_string(),
            });
        }

        // Create directory if needed
        std::fs::create_dir_all(&path).map_err(|e| NeuralPathError::DirectoryCreationFailed {
            path: path.display().to_string(),
            reason: e.to_string(),
        })?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let permissions = std::fs::Permissions::from_mode(self.dir_permissions);
            let _ = std::fs::set_permissions(&path, permissions);
        }

        Ok(path)
    }

    /// Get path for a specific model name
    pub fn model_manifest_path(&self, model_name: &str) -> Result<PathBuf, NeuralPathError> {
        // Validate model name
        if model_name.is_empty() || model_name.contains("..") || model_name.starts_with('/') {
            return Err(NeuralPathError::InvalidModelName {
                name: model_name.to_string(),
            });
        }

        let manifests = self.manifests_path()?;
        
        // Convert model name to safe filepath
        // Format: namespace/model:tag -> namespace/model/tag.json
        let safe_name = self.sanitize_model_name(model_name);
        let path = manifests.join(&safe_name);

        // Validate final path
        if !self.is_path_safe(&path) {
            return Err(NeuralPathError::PathTraversalAttempt {
                path: path.display().to_string(),
            });
        }

        Ok(path)
    }

    /// Get blobs directory path for a digest
    pub fn blob_path(&self, digest: &str) -> Result<PathBuf, NeuralPathError> {
        // Validate digest format
        if !digest.is_empty() && !DIGEST_REGEX.is_match(digest) {
            return Err(NeuralPathError::InvalidDigestFormat {
                digest: digest.to_string(),
            });
        }

        // Normalize digest (replace : with -)
        let normalized = digest.replace(':', "-");
        
        // Calculate 3D spatial bucket
        let (w, h, d) = self.spatial_layout.spatial_bucket(&normalized);
        
        // Build path: base/blobs/w/h/d/sha256-xxx or base/blobs/sha256-xxx (flat)
        let mut path = self.base_path.join(&self.blobs_dir);
        
        if self.spatial_layout.use_spatial_hash && !normalized.is_empty() {
            path = path.join(format!("{}/{}/{}", w, h, d));
        }
        
        if !normalized.is_empty() {
            path = path.join(&normalized);
        }

        // Validate path
        if !self.is_path_safe(&path) {
            return Err(NeuralPathError::PathTraversalAttempt {
                path: path.display().to_string(),
            });
        }

        // Create parent directories
        let parent = if normalized.is_empty() {
            &path
        } else {
            path.parent().unwrap_or(&path)
        };

        std::fs::create_dir_all(parent).map_err(|e| {
            NeuralPathError::DirectoryCreationFailed {
                path: parent.display().to_string(),
                reason: e.to_string(),
            }
        })?;

        Ok(path)
    }

    /// Get cache path for a digest
    pub fn cache_path(&self, digest: &str) -> Result<PathBuf, NeuralPathError> {
        // Validate digest
        if !digest.is_empty() && !DIGEST_REGEX.is_match(digest) {
            return Err(NeuralPathError::InvalidDigestFormat {
                digest: digest.to_string(),
            });
        }

        let normalized = digest.replace(':', "-");
        let cache_dir = self.base_path.join("cache");
        let path = cache_dir.join(&normalized);

        if !self.is_path_safe(&path) {
            return Err(NeuralPathError::PathTraversalAttempt {
                path: path.display().to_string(),
            });
        }

        std::fs::create_dir_all(&cache_dir).map_err(|e| {
            NeuralPathError::DirectoryCreationFailed {
                path: cache_dir.display().to_string(),
                reason: e.to_string(),
            }
        })?;

        Ok(path)
    }

    /// Sanitize model name for filesystem
    fn sanitize_model_name(&self, name: &str) -> String {
        // Replace problematic characters
        let sanitized = name
            .replace(':', "/")
            .replace('\\', "/")
            .replace("//", "/");

        // Remove leading/trailing slashes and dots
        sanitized.trim_start_matches('/').trim_end_matches('/').to_string()
    }

    /// Check if path is safe (no directory traversal)
    fn is_path_safe(&self, path: &Path) -> bool {
        let Ok(canonical_base) = self.base_path.canonicalize() else {
            // If base doesn't exist, check components
            return !path.components().any(|c| {
                matches!(c, std::path::Component::ParentDir)
            });
        };

        let Ok(canonical_path) = path.canonicalize() else {
            // Path doesn't exist yet, check components
            return !path.components().any(|c| {
                matches!(c, std::path::Component::ParentDir)
            }) && path.starts_with(&self.base_path);
        };

        canonical_path.starts_with(canonical_base)
    }

    /// Prune empty directories recursively
    pub fn prune_empty_directories(&self, path: &Path) -> Result<(), NeuralPathError> {
        self.prune_recursive(path, &self.base_path)
    }

    fn prune_recursive(&self, path: &Path, stop_at: &Path) -> Result<(), NeuralPathError> {
        // Don't go above the stop point
        if !path.starts_with(stop_at) || path == stop_at {
            return Ok(());
        }

        let metadata = fs::metadata(path).map_err(|e| NeuralPathError::NotADirectory {
            path: path.display().to_string(),
        })?;

        // Skip symlinks and non-directories
        if !metadata.is_dir() || metadata.file_type().is_symlink() {
            return Ok(());
        }

        // Read directory entries
        let entries = fs::read_dir(path).map_err(|e| NeuralPathError::PruneFailed {
            path: path.display().to_string(),
            reason: e.to_string(),
        })?;

        // Recurse into subdirectories
        for entry in entries.flatten() {
            let entry_path = entry.path();
            if entry_path.is_dir() {
                self.prune_recursive(&entry_path, stop_at)?;
            }
        }

        // Check if directory is now empty
        let remaining = fs::read_dir(path).map_err(|e| NeuralPathError::PruneFailed {
            path: path.display().to_string(),
            reason: e.to_string(),
        })?;

        if remaining.count() == 0 {
            fs::remove_dir(path).map_err(|e| NeuralPathError::PruneFailed {
                path: path.display().to_string(),
                reason: e.to_string(),
            })?;
        }

        Ok(())
    }

    /// Get storage statistics
    pub fn storage_stats(&self) -> io::Result<StorageStatistics> {
        let manifests = self.manifests_path()?;
        let blobs = self.base_path.join(&self.blobs_dir);

        let manifest_count = self.count_files(&manifests)?;
        let (blob_count, blob_bytes) = self.count_and_size(&blobs)?;

        Ok(StorageStatistics {
            manifest_count,
            blob_count,
            total_blob_bytes: blob_bytes,
            spatial_layout: self.spatial_layout,
        })
    }

    fn count_files(&self, path: &Path) -> io::Result<usize> {
        if !path.exists() {
            return Ok(0);
        }

        let mut count = 0;
        for entry in walkdir::WalkDir::new(path).into_iter().flatten() {
            if entry.file_type().is_file() {
                count += 1;
            }
        }
        Ok(count)
    }

    fn count_and_size(&self, path: &Path) -> io::Result<(usize, u64)> {
        if !path.exists() {
            return Ok((0, 0));
        }

        let mut count = 0;
        let mut bytes = 0u64;

        for entry in walkdir::WalkDir::new(path).into_iter().flatten() {
            if entry.file_type().is_file() {
                count += 1;
                if let Ok(metadata) = entry.metadata() {
                    bytes += metadata.len();
                }
            }
        }

        Ok((count, bytes))
    }

    /// Clear path cache
    pub fn clear_cache(&mut self) {
        self.path_cache.clear();
    }

    /// Get cached path (zero-copy lookup)
    pub fn get_cached_path(&self, key: &str) -> Option<Arc<PathBuf>> {
        self.path_cache.get(key).cloned()
    }

    /// Insert into cache
    pub fn cache_path(&mut self, key: impl Into<String>, path: PathBuf) {
        self.path_cache.insert(key.into(), Arc::new(path));
    }
}

impl Default for TitanStoragePathManager {
    fn default() -> Self {
        Self::new("models")
    }
}

/// Storage statistics
#[derive(Debug, Clone, Copy)]
pub struct StorageStatistics {
    pub manifest_count: usize,
    pub blob_count: usize,
    pub total_blob_bytes: u64,
    pub spatial_layout: SpatialStorageLayout,
}

impl fmt::Display for StorageStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "StorageStatistics {{ manifests: {}, blobs: {}, total: {}MB, layout: {}x{}x{} }}",
            self.manifest_count,
            self.blob_count,
            self.total_blob_bytes / 1_048_576,
            self.spatial_layout.width_shard,
            self.spatial_layout.height_tiers,
            self.spatial_layout.depth_layers
        )
    }
}

/// Builder for storage path manager
pub struct StoragePathManagerBuilder {
    base_path: PathBuf,
    spatial_layout: SpatialStorageLayout,
    manifest_dir: String,
    blobs_dir: String,
}

impl StoragePathManagerBuilder {
    /// Create new builder
    pub fn new(base_path: impl AsRef<Path>) -> Self {
        Self {
            base_path: base_path.as_ref().to_path_buf(),
            spatial_layout: SpatialStorageLayout::default(),
            manifest_dir: "manifests".to_string(),
            blobs_dir: "blobs".to_string(),
        }
    }

    /// With spatial layout
    pub fn spatial_layout(mut self, layout: SpatialStorageLayout) -> Self {
        self.spatial_layout = layout;
        self
    }

    /// With distributed layout (4x4x4)
    pub fn distributed(mut self) -> Self {
        self.spatial_layout = SpatialStorageLayout::distributed();
        self
    }

    /// With directory names
    pub fn directories(
        mut self,
        manifest: impl Into<String>,
        blobs: impl Into<String>,
    ) -> Self {
        self.manifest_dir = manifest.into();
        self.blobs_dir = blobs.into();
        self
    }

    /// Build path manager
    pub fn build(self) -> TitanStoragePathManager {
        TitanStoragePathManager {
            base_path: self.base_path,
            spatial_layout: self.spatial_layout,
            path_cache: HashMap::new(),
            manifest_dir: self.manifest_dir,
            blobs_dir: self.blobs_dir,
            dir_permissions: 0o755,
        }
    }
}

/// Type aliases
pub type PathManager = TitanStoragePathManager;
pub type PathManagerBuilder = StoragePathManagerBuilder;
pub type StorageLayout = SpatialStorageLayout;
pub type PathError = NeuralPathError;

/// Global path manager instance (lazy singleton)
static GLOBAL_PATH_MANAGER: Lazy<std::sync::Mutex<TitanStoragePathManager>> =
    Lazy::new(|| {
        std::sync::Mutex::new(TitanStoragePathManager::default())
    });

/// Get global path manager
pub fn global_path_manager() -> std::sync::MutexGuard<'static, TitanStoragePathManager> {
    GLOBAL_PATH_MANAGER.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_digest_validation() {
        // Valid digests
        assert!(DIGEST_REGEX.is_match("sha256-abc123abc123abc123abc123abc123abc123abc123abc123abc123abc123abc1"));
        assert!(DIGEST_REGEX.is_match("sha256:abc123abc123abc123abc123abc123abc123abc123abc123abc123abc123abc1"));
        
        // Invalid digests
        assert!(!DIGEST_REGEX.is_match(""));
        assert!(!DIGEST_REGEX.is_match("sha256-short"));
        assert!(!DIGEST_REGEX.is_match("md5-abc123abc123abc123abc123abc123ab"));
    }

    #[test]
    fn test_spatial_layout() {
        let flat = StorageLayout::flat();
        assert_eq!(flat.spatial_bucket("any"), (0, 0, 0));

        let distributed = StorageLayout::distributed();
        let (w, h, d) = distributed.spatial_bucket("sha256-abc123abc123abc123abc123abc123abc123abc123abc123abc123abc123abc1");
        
        assert!(w < 4);
        assert!(h < 4);
        assert!(d < 4);
    }

    #[test]
    fn test_manifests_path() {
        let temp = TempDir::new().unwrap();
        let manager = PathManager::new(temp.path());
        
        let path = manager.manifests_path().unwrap();
        assert!(path.exists());
        assert!(path.ends_with("manifests"));
    }

    #[test]
    fn test_model_manifest_path() {
        let temp = TempDir::new().unwrap();
        let manager = PathManager::new(temp.path());
        
        let path = manager.model_manifest_path("namespace/model").unwrap();
        assert!(path.to_string_lossy().contains("namespace"));
        assert!(path.to_string_lossy().contains("model"));
    }

    #[test]
    fn test_blob_path() {
        let temp = TempDir::new().unwrap();
        let manager = PathManager::new(temp.path());
        
        let digest = "sha256-abc123abc123abc123abc123abc123abc123abc123abc123abc123abc123abc1";
        let path = manager.blob_path(digest).unwrap();
        
        assert!(path.to_string_lossy().contains("blobs"));
        assert!(path.to_string_lossy().contains("abc123"));
    }

    #[test]
    fn test_invalid_digest() {
        let temp = TempDir::new().unwrap();
        let manager = PathManager::new(temp.path());
        
        let result = manager.blob_path("invalid-digest");
        assert!(matches!(result, Err(PathError::InvalidDigestFormat { .. })));
    }

    #[test]
    fn test_path_traversal_protection() {
        let temp = TempDir::new().unwrap();
        let manager = PathManager::new(temp.path());
        
        let result = manager.model_manifest_path("../etc/passwd");
        assert!(matches!(result, Err(PathError::InvalidModelName { .. })));
    }

    #[test]
    fn test_spatial_blob_path() {
        let temp = TempDir::new().unwrap();
        let manager = PathManagerBuilder::new(temp.path())
            .distributed()
            .build();
        
        let digest = "sha256-abc123abc123abc123abc123abc123abc123abc123abc123abc123abc123abc1";
        let path = manager.blob_path(digest).unwrap();
        
        // Should have 3D structure
        let path_str = path.to_string_lossy();
        assert!(path_str.contains("blobs"));
        // The path should contain numeric components for 3D bucketing
    }

    #[test]
    fn test_sanitize_model_name() {
        let manager = PathManager::new("/tmp");
        
        assert_eq!(manager.sanitize_model_name("model:tag"), "model/tag");
        assert_eq!(manager.sanitize_model_name("/model/"), "model");
        assert_eq!(manager.sanitize_model_name("ns::model"), "ns/model");
    }

    #[test]
    fn test_builder_pattern() {
        let temp = TempDir::new().unwrap();
        let manager = PathManagerBuilder::new(temp.path())
            .spatial_layout(StorageLayout::distributed())
            .directories("meta", "data")
            .build();
        
        let manifests = manager.manifests_path().unwrap();
        assert!(manifests.to_string_lossy().contains("meta"));
    }

    #[test]
    fn test_cache_operations() {
        let temp = TempDir::new().unwrap();
        let mut manager = PathManager::new(temp.path());
        
        let path = PathBuf::from("/test/path");
        manager.cache_path("test_key", path.clone());
        
        let cached = manager.get_cached_path("test_key");
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().as_ref(), &path);
    }
}
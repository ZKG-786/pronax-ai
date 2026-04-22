
use std::collections::HashMap;
use std::fmt;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bytes::{Bytes, BytesMut};
use ed25519_dalek::{Signer, SigningKey, VerifyingKey};
use reqwest::{Client, ClientBuilder, Method, Request, Response, StatusCode, Url};
use serde::{Deserialize, Serialize};
use sha2::{Digest as Sha2Digest, Sha256};
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::{mpsc, oneshot, Mutex, Notify, RwLock as TokioRwLock, Semaphore};
use tokio::task::JoinHandle;
use tokio::time::{sleep, timeout};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};

/// 3D Spatial coordinate for registry operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RegistryCoordinate {
    pub request_sequence: u64,
    pub network_tier: u16,
    pub cache_depth: u8,
    pub reliability_score: f32,
}

impl RegistryCoordinate {
    pub const fn new(seq: u64, tier: u16, depth: u8, reliability: f32) -> Self {
        Self {
            request_sequence: seq,
            network_tier: tier,
            cache_depth: depth,
            reliability_score: reliability,
        }
    }

    pub const fn standard() -> Self {
        Self::new(0, 500, 5, 0.95)
    }

    pub const fn high_priority() -> Self {
        Self::new(0, 900, 10, 0.99)
    }

    pub const fn cached() -> Self {
        Self::new(0, 300, 3, 0.999)
    }

    /// Calculate operation priority
    pub fn priority_score(&self) -> u64 {
        let seq_factor = 1000u64.saturating_sub(self.request_sequence);
        let tier_boost = self.network_tier as u64 * 100;
        let depth_norm = self.cache_depth as u64 * 10;
        let reliability_boost = (self.reliability_score * 1000.0) as u64;
        
        seq_factor + tier_boost + depth_norm + reliability_boost
    }
}

/// Registry operation status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegistryOperationStatus {
    Idle,
    Resolving,
    Authenticating,
    Pulling,
    Pushing,
    Chunking,
    Verifying,
    Completed,
    Failed,
    Cancelled,
}

impl fmt::Display for RegistryOperationStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Model digest with validation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NeuralDigest {
    pub algorithm: String,
    pub hash: String,
    pub full: String,
}

impl NeuralDigest {
    pub fn new(algorithm: impl Into<String>, hash: impl Into<String>) -> Self {
        let algo = algorithm.into();
        let h = hash.into();
        let full = format!("{}:{}", algo, h);
        
        Self {
            algorithm: algo,
            hash: h,
            full,
        }
    }

    pub fn from_bytes(data: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let result = hasher.finalize();
        
        Self::new("sha256", format!("{:x}", result))
    }

    pub fn is_valid(&self) -> bool {
        !self.algorithm.is_empty() && !self.hash.is_empty()
    }

    pub fn short(&self) -> String {
        if self.hash.len() > 12 {
            format!("{}:{}", self.algorithm, &self.hash[..12])
        } else {
            self.full.clone()
        }
    }
}

impl fmt::Display for NeuralDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.full)
    }
}

impl FromStr for NeuralDigest {
    type Err = RegistryError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.splitn(2, ':').collect();
        if parts.len() != 2 {
            return Err(RegistryError::InvalidDigest(s.to_string()));
        }
        
        Ok(Self::new(parts[0], parts[1]))
    }
}

/// Model layer with 3D spatial metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralLayer {
    pub digest: NeuralDigest,
    pub media_type: String,
    pub size: i64,
    pub coordinate: RegistryCoordinate,
}

impl NeuralLayer {
    pub fn new(digest: NeuralDigest, media_type: impl Into<String>, size: i64) -> Self {
        Self {
            digest,
            media_type: media_type.into(),
            size,
            coordinate: RegistryCoordinate::standard(),
        }
    }

    pub fn with_coordinate(mut self, coord: RegistryCoordinate) -> Self {
        self.coordinate = coord;
        self
    }
}

/// Model manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralManifest {
    pub name: String,
    pub digest: NeuralDigest,
    pub layers: Vec<NeuralLayer>,
    pub config: Option<NeuralLayer>,
    pub metadata: HashMap<String, String>,
    pub coordinate: RegistryCoordinate,
    pub created_at: u64,
}

impl NeuralManifest {
    pub fn new(name: impl Into<String>, digest: NeuralDigest) -> Self {
        Self {
            name: name.into(),
            digest,
            layers: Vec::new(),
            config: None,
            metadata: HashMap::new(),
            coordinate: RegistryCoordinate::standard(),
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    pub fn total_size(&self) -> i64 {
        let mut size = 0i64;
        if let Some(config) = &self.config {
            size += config.size;
        }
        for layer in &self.layers {
            size += layer.size;
        }
        size
    }

    pub fn layer_by_digest(&self, digest: &NeuralDigest) -> Option<&NeuralLayer> {
        self.layers.iter().find(|l| &l.digest == digest)
    }

    pub fn add_layer(mut self, layer: NeuralLayer) -> Self {
        self.layers.push(layer);
        self
    }
}

/// Model name with validation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NeuralModelName {
    pub host: String,
    pub namespace: String,
    pub model: String,
    pub tag: String,
    pub digest: Option<NeuralDigest>,
    pub scheme: String,
}

impl NeuralModelName {
    pub fn new(
        host: impl Into<String>,
        namespace: impl Into<String>,
        model: impl Into<String>,
        tag: impl Into<String>,
    ) -> Self {
        Self {
            host: host.into(),
            namespace: namespace.into(),
            model: model.into(),
            tag: tag.into(),
            digest: None,
            scheme: "https".to_string(),
        }
    }

    pub fn with_digest(mut self, digest: NeuralDigest) -> Self {
        self.digest = Some(digest);
        self
    }

    pub fn with_scheme(mut self, scheme: impl Into<String>) -> Self {
        self.scheme = scheme.into();
        self
    }

    pub fn is_fully_qualified(&self) -> bool {
        !self.host.is_empty() && !self.model.is_empty()
    }

    pub fn manifest_url(&self) -> String {
        format!(
            "{}://{}/v2/{}/{}/manifests/{}",
            self.scheme, self.host, self.namespace, self.model, self.tag
        )
    }

    pub fn blob_url(&self, digest: &NeuralDigest) -> String {
        format!(
            "{}://{}/v2/{}/{}/blobs/{}",
            self.scheme, self.host, self.namespace, self.model, digest
        )
    }

    pub fn chunksums_url(&self, digest: &NeuralDigest) -> String {
        format!(
            "{}://{}/v2/{}/{}/chunksums/{}",
            self.scheme, self.host, self.namespace, self.model, digest
        )
    }

    pub fn upload_url(&self, digest: &NeuralDigest) -> String {
        format!(
            "{}://{}/v2/{}/{}/blobs/uploads/?digest={}",
            self.scheme, self.host, self.namespace, self.model, digest
        )
    }

    pub fn merge(&self, other: &Self) -> Self {
        let mut merged = self.clone();
        
        if merged.host.is_empty() && !other.host.is_empty() {
            merged.host = other.host.clone();
        }
        if merged.namespace.is_empty() && !other.namespace.is_empty() {
            merged.namespace = other.namespace.clone();
        }
        if merged.tag.is_empty() && !other.tag.is_empty() {
            merged.tag = other.tag.clone();
        }
        if merged.scheme == "https" && other.scheme != "https" {
            merged.scheme = other.scheme.clone();
        }
        
        merged
    }
}

impl fmt::Display for NeuralModelName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(digest) = &self.digest {
            write!(f, "{}@{}", self.name_without_digest(), digest)
        } else {
            write!(f, "{}", self.name_without_digest())
        }
    }
}

impl NeuralModelName {
    fn name_without_digest(&self) -> String {
        format!(
            "{}://{}/{}/{}:{}",
            self.scheme, self.host, self.namespace, self.model, self.tag
        )
    }
}

impl FromStr for NeuralModelName {
    type Err = RegistryError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (scheme, rest) = if let Some(pos) = s.find("://") {
            (s[..pos].to_string(), &s[pos + 3..])
        } else {
            ("https".to_string(), s)
        };

        let (name_part, digest) = if let Some(pos) = rest.rfind('@') {
            let digest = NeuralDigest::from_str(&rest[pos + 1..])?;
            (&rest[..pos], Some(digest))
        } else {
            (rest, None)
        };

        let parts: Vec<&str> = name_part.split('/').collect();
        if parts.len() < 2 {
            return Err(RegistryError::InvalidName(s.to_string()));
        }

        let host = parts[0].to_string();
        let (namespace, model_tag) = if parts.len() == 2 {
            ("library".to_string(), parts[1])
        } else {
            (parts[1].to_string(), parts[2])
        };

        let (model, tag) = if let Some(pos) = model_tag.find(':') {
            (model_tag[..pos].to_string(), model_tag[pos + 1..].to_string())
        } else {
            (model_tag.to_string(), "latest".to_string())
        };

        Ok(Self {
            host,
            namespace,
            model,
            tag,
            digest,
            scheme,
        })
    }
}

/// Chunk range for partial downloads
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeuralChunkRange {
    pub start: i64,
    pub end: i64,
}

impl NeuralChunkRange {
    pub fn new(start: i64, end: i64) -> Self {
        Self { start, end }
    }

    pub fn size(&self) -> i64 {
        self.end - self.start + 1
    }
}

impl fmt::Display for NeuralChunkRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{}", self.start, self.end)
    }
}

impl FromStr for NeuralChunkRange {
    type Err = RegistryError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split('-').collect();
        if parts.len() != 2 {
            return Err(RegistryError::InvalidChunkRange(s.to_string()));
        }

        let start = parts[0].parse::<i64>()
            .map_err(|_| RegistryError::InvalidChunkRange(s.to_string()))?;
        let end = parts[1].parse::<i64>()
            .map_err(|_| RegistryError::InvalidChunkRange(s.to_string()))?;

        if start > end {
            return Err(RegistryError::InvalidChunkRange(s.to_string()));
        }

        Ok(Self::new(start, end))
    }
}

/// Chunk summary for chunked downloads
#[derive(Debug, Clone)]
pub struct NeuralChunkSum {
    pub url: String,
    pub chunk: NeuralChunkRange,
    pub digest: NeuralDigest,
    pub coordinate: RegistryCoordinate,
}

/// Progress tracking for registry operations
#[derive(Debug, Clone)]
pub struct RegistryProgress {
    pub operation_id: Uuid,
    pub layer_digest: Option<NeuralDigest>,
    pub status: RegistryOperationStatus,
    pub total_bytes: i64,
    pub completed_bytes: i64,
    pub error: Option<String>,
    pub timestamp: u64,
    pub coordinate: RegistryCoordinate,
}

impl RegistryProgress {
    pub fn new(operation_id: Uuid) -> Self {
        Self {
            operation_id,
            layer_digest: None,
            status: RegistryOperationStatus::Idle,
            total_bytes: 0,
            completed_bytes: 0,
            error: None,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            coordinate: RegistryCoordinate::standard(),
        }
    }

    pub fn percentage(&self) -> f64 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        (self.completed_bytes as f64 / self.total_bytes as f64) * 100.0
    }
}

use uuid::Uuid;

/// Authentication token generator
pub struct NeuralAuthToken {
    signing_key: SigningKey,
}

impl NeuralAuthToken {
    pub fn new(signing_key: SigningKey) -> Self {
        Self { signing_key }
    }

    pub fn generate(&self, url: &str) -> Result<String, RegistryError> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let check_url = format!("{}?ts={}", url, timestamp);
        
        // Get public key in SSH format
        let verifying_key = self.signing_key.verifying_key();
        let public_key_bytes = verifying_key.to_bytes();
        let pub_key_b64 = BASE64.encode(&public_key_bytes);
        
        // Create signature
        let message = format!("GET,{},", check_url);
        let signature = self.signing_key.sign(message.as_bytes());
        let sig_b64 = BASE64.encode(&signature.to_bytes());
        
        // Assemble token: <checkData>:<pubKey>:<signature>
        let check_data_b64 = BASE64.encode(check_url.as_bytes());
        Ok(format!("{}:{}:{}", check_data_b64, pub_key_b64, sig_b64))
    }
}

/// Registry configuration
#[derive(Debug, Clone)]
pub struct NeuralRegistryConfig {
    pub cache_dir: PathBuf,
    pub user_agent: String,
    pub max_streams: usize,
    pub chunking_threshold: i64,
    pub read_timeout: Duration,
    pub default_mask: String,
    pub retry_attempts: u32,
}

impl Default for NeuralRegistryConfig {
    fn default() -> Self {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        let cache_dir = home.join(".pronax").join("models");
        
        Self {
            cache_dir,
            user_agent: format!(
                "pronax/{version} (3D-Neural-Registry)",
                version = env!("CARGO_PKG_VERSION", "0.1.0")
            ),
            max_streams: num_cpus::get(),
            chunking_threshold: 64 * 1024 * 1024, // 64MB
            read_timeout: Duration::from_secs(30),
            default_mask: "registry.pronax.ai/library/_:latest".to_string(),
            retry_attempts: 5,
        }
    }
}

/// Registry error types
#[derive(Debug, Clone)]
pub enum RegistryError {
    ModelNotFound(String),
    InvalidManifest(String),
    InvalidName(String),
    InvalidDigest(String),
    InvalidChunkRange(String),
    CacheError(String),
    NetworkError(String),
    HttpError(u16, String),
    Unauthorized,
    Forbidden,
    Incomplete(String),
    AuthenticationError(String),
    FileError(String),
    SerializationError(String),
    Cancelled,
    Unknown(String),
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelNotFound(s) => write!(f, "Model not found: {}", s),
            Self::InvalidManifest(s) => write!(f, "Invalid manifest: {}", s),
            Self::InvalidName(s) => write!(f, "Invalid name: {}", s),
            Self::InvalidDigest(s) => write!(f, "Invalid digest: {}", s),
            Self::InvalidChunkRange(s) => write!(f, "Invalid chunk range: {}", s),
            Self::CacheError(s) => write!(f, "Cache error: {}", s),
            Self::NetworkError(s) => write!(f, "Network error: {}", s),
            Self::HttpError(code, s) => write!(f, "HTTP error {}: {}", code, s),
            Self::Unauthorized => write!(f, "Unauthorized"),
            Self::Forbidden => write!(f, "Forbidden"),
            Self::Incomplete(s) => write!(f, "Incomplete: {}", s),
            Self::AuthenticationError(s) => write!(f, "Authentication error: {}", s),
            Self::FileError(s) => write!(f, "File error: {}", s),
            Self::SerializationError(s) => write!(f, "Serialization error: {}", s),
            Self::Cancelled => write!(f, "Operation cancelled"),
            Self::Unknown(s) => write!(f, "Unknown error: {}", s),
        }
    }
}

impl std::error::Error for RegistryError {}

/// Registry response error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryResponseError {
    pub code: Option<String>,
    pub message: String,
    #[serde(skip)]
    pub status: u16,
}

impl RegistryResponseError {
    pub fn is_temporary(&self) -> bool {
        self.status >= 500
    }
}

impl fmt::Display for RegistryResponseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Registry error (status {}): ", self.status)?;
        if let Some(code) = &self.code {
            write!(f, "[{}] ", code)?;
        }
        write!(f, "{}", self.message)
    }
}

/// Operation result
#[derive(Debug, Clone)]
pub struct RegistryOperationResult {
    pub success: bool,
    pub manifest: Option<NeuralManifest>,
    pub total_bytes: i64,
    pub transferred_bytes: i64,
    pub duration_secs: u64,
    pub error: Option<String>,
}

/// Neural Registry Client
pub struct NeuralRegistryClient {
    config: NeuralRegistryConfig,
    client: Client,
    auth_token: Option<NeuralAuthToken>,
    active_operations: Arc<TokioRwLock<HashMap<Uuid, Arc<NeuralRegistryOperation>>>>,
    progress_tx: mpsc::Sender<RegistryProgress>,
    semaphore: Arc<Semaphore>,
}

/// Active registry operation
pub struct NeuralRegistryOperation {
    pub id: Uuid,
    pub name: NeuralModelName,
    pub status: Arc<RwLock<RegistryOperationStatus>>,
    pub progress: Arc<RwLock<RegistryProgress>>,
    pub cancelled: AtomicBool,
    pub created_at: Instant,
}

impl NeuralRegistryClient {
    pub fn new(config: NeuralRegistryConfig) -> Result<Self, RegistryError> {
        let client = ClientBuilder::new()
            .timeout(config.read_timeout)
            .user_agent(&config.user_agent)
            .build()
            .map_err(|e| RegistryError::NetworkError(e.to_string()))?;

        let (progress_tx, _) = mpsc::channel(100);
        
        Ok(Self {
            config,
            client,
            auth_token: None,
            active_operations: Arc::new(TokioRwLock::new(HashMap::new())),
            progress_tx,
            semaphore: Arc::new(Semaphore::new(num_cpus::get())),
        })
    }

    pub fn with_auth(mut self, signing_key: SigningKey) -> Self {
        self.auth_token = Some(NeuralAuthToken::new(signing_key));
        self
    }

    /// Parse and complete a model name
    pub fn complete_name(&self, name: &str) -> Result<NeuralModelName, RegistryError> {
        let mut parsed = NeuralModelName::from_str(name)?;
        
        if !parsed.is_fully_qualified() {
            let default_mask = NeuralModelName::from_str(&self.config.default_mask)?;
            parsed = parsed.merge(&default_mask);
        }
        
        Ok(parsed)
    }

    /// Resolve a manifest from the remote registry
    pub async fn resolve_remote(
        &self,
        name: &NeuralModelName,
    ) -> Result<NeuralManifest, RegistryError> {
        let url = name.manifest_url();
        let request = self.build_request(Method::GET, &url, None).await?;
        
        let response = self.send_request(request).await?;
        let data = response.bytes().await
            .map_err(|e| RegistryError::NetworkError(e.to_string()))?;
        
        let mut manifest: NeuralManifest = serde_json::from_slice(&data)
            .map_err(|e| RegistryError::SerializationError(e.to_string()))?;
        
        manifest.name = name.to_string();
        manifest.digest = NeuralDigest::from_bytes(&data);
        
        Ok(manifest)
    }

    /// Push a model to the registry
    pub async fn push(
        &self,
        name: &str,
        from: Option<&str>,
    ) -> Result<RegistryOperationResult, RegistryError> {
        let start_time = Instant::now();
        let operation_id = Uuid::new_v4();
        
        // Parse and complete name
        let target_name = self.complete_name(name)?;
        let source_name = if let Some(from) = from {
            self.complete_name(from)?
        } else {
            target_name.clone()
        };
        
        // Resolve local manifest
        let manifest = self.resolve_local(&source_name).await?;
        
        // Verify layers exist
        for layer in &manifest.layers {
            let blob_path = self.config.cache_dir.join("blobs").join(&layer.digest.hash);
            if !blob_path.exists() {
                return Err(RegistryError::FileError(
                    format!("Blob not found: {}", layer.digest)
                ));
            }
        }
        
        // Create operation
        let operation = Arc::new(NeuralRegistryOperation {
            id: operation_id,
            name: target_name.clone(),
            status: Arc::new(RwLock::new(RegistryOperationStatus::Pushing)),
            progress: Arc::new(RwLock::new(RegistryProgress::new(operation_id))),
            cancelled: AtomicBool::new(false),
            created_at: Instant::now(),
        });
        
        {
            let mut ops = self.active_operations.write().await;
            ops.insert(operation_id, operation.clone());
        }
        
        // Push layers in parallel
        let total_size = manifest.total_size();
        let mut completed = 0i64;
        let mut handles: Vec<JoinHandle<Result<i64, RegistryError>>> = Vec::new();
        
        for layer in &manifest.layers {
            let client = self.clone();
            let name = target_name.clone();
            let layer = layer.clone();
            let permit = self.semaphore.clone().acquire_owned().await
                .map_err(|_| RegistryError::Unknown("Failed to acquire semaphore".to_string()))?;
            
            let handle = tokio::spawn(async move {
                let _permit = permit;
                
                // Check if already cached remotely
                let upload_url = name.upload_url(&layer.digest);
                let request = client.build_request(Method::POST, &upload_url, None).await?;
                let response = client.send_request(request).await?;
                
                // If Location header is empty, blob already exists
                let location = response.headers().get("Location")
                    .and_then(|v| v.to_str().ok())
                    .map(|s| s.to_string());
                
                if location.is_none() {
                    // Already cached
                    return Ok(layer.size);
                }
                
                // Upload blob
                let blob_path = client.config.cache_dir.join("blobs").join(&layer.digest.hash);
                let file = fs::File::open(&blob_path).await
                    .map_err(|e| RegistryError::FileError(e.to_string()))?;
                
                let request = client.build_request(Method::PUT, &location.unwrap(), Some(layer.size)).await?;
                let request = request.body(file);
                let _response = client.send_request(request).await?;
                
                Ok(layer.size)
            });
            
            handles.push(handle);
        }
        
        // Wait for all uploads
        for handle in handles {
            match handle.await {
                Ok(Ok(size)) => completed += size,
                Ok(Err(e)) => return Err(e),
                Err(e) => return Err(RegistryError::Unknown(e.to_string())),
            }
        }
        
        // Commit manifest
        let manifest_url = target_name.manifest_url();
        let manifest_data = serde_json::to_vec(&manifest)
            .map_err(|e| RegistryError::SerializationError(e.to_string()))?;
        
        let request = self.build_request(Method::PUT, &manifest_url, Some(manifest_data.len() as i64)).await?;
        let request = request.body(manifest_data);
        let _response = self.send_request(request).await?;
        
        let duration = start_time.elapsed().as_secs();
        
        // Update status
        {
            let mut status = operation.status.write().unwrap();
            *status = RegistryOperationStatus::Completed;
        }
        
        Ok(RegistryOperationResult {
            success: true,
            manifest: Some(manifest),
            total_bytes: total_size,
            transferred_bytes: completed,
            duration_secs: duration,
            error: None,
        })
    }

    /// Pull a model from the registry
    pub async fn pull(
        &self,
        name: &str,
    ) -> Result<RegistryOperationResult, RegistryError> {
        let start_time = Instant::now();
        let operation_id = Uuid::new_v4();
        
        // Parse and complete name
        let target_name = self.complete_name(name)?;
        
        // Resolve remote manifest
        let manifest = self.resolve_remote(&target_name).await?;
        
        if manifest.layers.is_empty() {
            return Err(RegistryError::InvalidManifest("No layers in manifest".to_string()));
        }
        
        // Create operation
        let operation = Arc::new(NeuralRegistryOperation {
            id: operation_id,
            name: target_name.clone(),
            status: Arc::new(RwLock::new(RegistryOperationStatus::Pulling)),
            progress: Arc::new(RwLock::new(RegistryProgress::new(operation_id))),
            cancelled: AtomicBool::new(false),
            created_at: Instant::now(),
        });
        
        {
            let mut ops = self.active_operations.write().await;
            ops.insert(operation_id, operation.clone());
        }
        
        // Collect all layers including config
        let mut layers = manifest.layers.clone();
        if let Some(config) = &manifest.config {
            layers.push(config.clone());
        }
        
        // Download layers
        let total_size = manifest.total_size();
        let mut completed = AtomicI64::new(0);
        let mut handles: Vec<JoinHandle<Result<i64, RegistryError>>> = Vec::new();
        
        for layer in &layers {
            let client = self.clone();
            let name = target_name.clone();
            let layer = layer.clone();
            let completed_ref = Arc::new(completed.clone());
            let permit = self.semaphore.clone().acquire_owned().await
                .map_err(|_| RegistryError::Unknown("Failed to acquire semaphore".to_string()))?;
            
            let handle = tokio::spawn(async move {
                let _permit = permit;
                
                // Check if already cached locally
                let blob_path = client.config.cache_dir.join("blobs").join(&layer.digest.hash);
                if blob_path.exists() {
                    let metadata = fs::metadata(&blob_path).await
                        .map_err(|e| RegistryError::FileError(e.to_string()))?;
                    if metadata.len() as i64 == layer.size {
                        completed_ref.fetch_add(layer.size, Ordering::SeqCst);
                        return Ok(layer.size);
                    }
                }
                
                // Determine if chunking is needed
                let downloaded = if layer.size > client.config.chunking_threshold {
                    client.download_chunked(&name, &layer).await?
                } else {
                    client.download_single(&name, &layer).await?
                };
                
                completed_ref.fetch_add(downloaded, Ordering::SeqCst);
                Ok(downloaded)
            });
            
            handles.push(handle);
        }
        
        // Wait for all downloads
        let mut total_downloaded = 0i64;
        for handle in handles {
            match handle.await {
                Ok(Ok(size)) => total_downloaded += size,
                Ok(Err(e)) => return Err(e),
                Err(e) => return Err(RegistryError::Unknown(e.to_string())),
            }
        }
        
        // Save manifest
        let manifest_path = self.config.cache_dir.join("manifests").join(&target_name.to_string());
        if let Some(parent) = manifest_path.parent() {
            fs::create_dir_all(parent).await
                .map_err(|e| RegistryError::CacheError(e.to_string()))?;
        }
        
        let manifest_data = serde_json::to_vec(&manifest)
            .map_err(|e| RegistryError::SerializationError(e.to_string()))?;
        fs::write(&manifest_path, manifest_data).await
            .map_err(|e| RegistryError::CacheError(e.to_string()))?;
        
        let duration = start_time.elapsed().as_secs();
        
        // Update status
        {
            let mut status = operation.status.write().unwrap();
            *status = RegistryOperationStatus::Completed;
        }
        
        Ok(RegistryOperationResult {
            success: true,
            manifest: Some(manifest),
            total_bytes: total_size,
            transferred_bytes: total_downloaded,
            duration_secs: duration,
            error: None,
        })
    }

    /// Download a single layer
    async fn download_single(
        &self,
        name: &NeuralModelName,
        layer: &NeuralLayer,
    ) -> Result<i64, RegistryError> {
        let url = name.blob_url(&layer.digest);
        let request = self.build_request(Method::GET, &url, None).await?;
        
        let response = self.send_request(request).await?;
        let data = response.bytes().await
            .map_err(|e| RegistryError::NetworkError(e.to_string()))?;
        
        // Verify digest
        let computed_digest = NeuralDigest::from_bytes(&data);
        if computed_digest.hash != layer.digest.hash {
            return Err(RegistryError::InvalidManifest(
                format!("Digest mismatch: expected {}, got {}", layer.digest, computed_digest)
            ));
        }
        
        // Save to cache
        let blob_path = self.config.cache_dir.join("blobs").join(&layer.digest.hash);
        if let Some(parent) = blob_path.parent() {
            fs::create_dir_all(parent).await
                .map_err(|e| RegistryError::CacheError(e.to_string()))?;
        }
        
        fs::write(&blob_path, data).await
            .map_err(|e| RegistryError::CacheError(e.to_string()))?;
        
        Ok(layer.size)
    }

    /// Download a layer in chunks
    async fn download_chunked(
        &self,
        name: &NeuralModelName,
        layer: &NeuralLayer,
    ) -> Result<i64, RegistryError> {
        // Get chunksums
        let chunksums_url = name.chunksums_url(&layer.digest);
        let request = self.build_request(Method::GET, &chunksums_url, None).await?;
        let response = self.send_request(request).await?;
        
        let blob_url = response.headers().get("Content-Location")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
            .ok_or_else(|| RegistryError::NetworkError("Missing Content-Location".to_string()))?;
        
        let body = response.text().await
            .map_err(|e| RegistryError::NetworkError(e.to_string()))?;
        
        // Parse chunksums
        let mut chunksums = Vec::new();
        for line in body.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 2 {
                continue;
            }
            
            let digest = NeuralDigest::from_str(parts[0])?;
            let chunk = NeuralChunkRange::from_str(parts[1])?;
            
            chunksums.push(NeuralChunkSum {
                url: blob_url.clone(),
                chunk,
                digest,
                coordinate: RegistryCoordinate::standard(),
            });
        }
        
        // Download chunks in parallel
        let blob_path = self.config.cache_dir.join("blobs").join(&layer.digest.hash);
        if let Some(parent) = blob_path.parent() {
            fs::create_dir_all(parent).await
                .map_err(|e| RegistryError::CacheError(e.to_string()))?;
        }
        
        // Create sparse file
        let file = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&blob_path)
            .await
            .map_err(|e| RegistryError::FileError(e.to_string()))?;
        
        file.set_len(layer.size as u64).await
            .map_err(|e| RegistryError::FileError(e.to_string()))?;
        
        // Download chunks
        let mut handles: Vec<JoinHandle<Result<(), RegistryError>>> = Vec::new();
        
        for chunksum in &chunksums {
            let client = self.clone();
            let chunksum = chunksum.clone();
            let blob_path = blob_path.clone();
            let permit = self.semaphore.clone().acquire_owned().await
                .map_err(|_| RegistryError::Unknown("Failed to acquire semaphore".to_string()))?;
            
            let handle = tokio::spawn(async move {
                let _permit = permit;
                
                // Build range request
                let range_header = format!("bytes={}-{}", chunksum.chunk.start, chunksum.chunk.end);
                
                let request = client.build_request(Method::GET, &chunksum.url, None).await?;
                let request = request.header("Range", range_header);
                let response = client.send_request(request).await?;
                
                let data = response.bytes().await
                    .map_err(|e| RegistryError::NetworkError(e.to_string()))?;
                
                // Verify chunk digest
                let computed_digest = NeuralDigest::from_bytes(&data);
                if computed_digest.hash != chunksum.digest.hash {
                    return Err(RegistryError::InvalidManifest(
                        format!("Chunk digest mismatch: expected {}, got {}", chunksum.digest, computed_digest)
                    ));
                }
                
                // Write chunk to file at correct offset
                let mut file = fs::OpenOptions::new()
                    .write(true)
                    .open(&blob_path)
                    .await
                    .map_err(|e| RegistryError::FileError(e.to_string()))?;
                
                file.seek(tokio::io::SeekFrom::Start(chunksum.chunk.start as u64)).await
                    .map_err(|e| RegistryError::FileError(e.to_string()))?;
                file.write_all(&data).await
                    .map_err(|e| RegistryError::FileError(e.to_string()))?;
                
                Ok(())
            });
            
            handles.push(handle);
        }
        
        // Wait for all chunks
        for handle in handles {
            match handle.await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(e),
                Err(e) => return Err(RegistryError::Unknown(e.to_string())),
            }
        }
        
        Ok(layer.size)
    }

    /// Resolve a manifest from the local cache
    pub async fn resolve_local(&self, name: &NeuralModelName) -> Result<NeuralManifest, RegistryError> {
        let manifest_path = self.config.cache_dir.join("manifests").join(&name.to_string());
        let data = fs::read(&manifest_path).await.map_err(|e| RegistryError::CacheError(e.to_string()))?;
        let mut manifest: NeuralManifest = serde_json::from_slice(&data).map_err(|e| RegistryError::SerializationError(e.to_string()))?;
        manifest.digest = NeuralDigest::from_bytes(&data);
        Ok(manifest)
    }

    /// Build HTTP request with auth
    async fn build_request(&self, method: Method, url: &str, content_length: Option<i64>) -> Result<reqwest::RequestBuilder, RegistryError> {
        let mut request = self.client.request(method, url);
        if let Some(len) = content_length { request = request.header("Content-Length", len); }
        if let Some(auth) = &self.auth_token {
            let token = auth.generate(url)?;
            request = request.header("Authorization", format!("Bearer {}", token));
        }
        Ok(request)
    }

    /// Send HTTP request and handle errors
    async fn send_request(&self, request: reqwest::RequestBuilder) -> Result<Response, RegistryError> {
        let response = request.send().await.map_err(|e| RegistryError::NetworkError(e.to_string()))?;
        match response.status() {
            StatusCode::OK | StatusCode::CREATED | StatusCode::ACCEPTED => Ok(response),
            StatusCode::NOT_FOUND => Err(RegistryError::ModelNotFound("Model not found".to_string())),
            StatusCode::UNAUTHORIZED => Err(RegistryError::Unauthorized),
            StatusCode::FORBIDDEN => Err(RegistryError::Forbidden),
            status => {
                let body = response.text().await.map_err(|e| RegistryError::NetworkError(e.to_string()))?;
                Err(RegistryError::HttpError(status.as_u16(), body))
            }
        }
    }

    /// Get operation progress
    pub async fn get_progress(&self, operation_id: Uuid) -> Option<RegistryProgress> {
        let ops = self.active_operations.read().await;
        ops.get(&operation_id).map(|op| op.progress.read().unwrap().clone())
    }

    /// Cancel operation
    pub async fn cancel_operation(&self, operation_id: Uuid) -> Result<(), RegistryError> {
        let ops = self.active_operations.read().await;
        if let Some(op) = ops.get(&operation_id) {
            op.cancelled.store(true, Ordering::SeqCst);
            let mut status = op.status.write().unwrap();
            *status = RegistryOperationStatus::Cancelled;
            Ok(())
        } else {
            Err(RegistryError::Unknown("Operation not found".to_string()))
        }
    }

    /// Cleanup completed operations
    pub async fn cleanup_operations(&self) -> usize {
        let mut ops = self.active_operations.write().await;
        let before = ops.len();
        ops.retain(|_, op| {
            let status = op.status.read().unwrap();
            !matches!(*status, RegistryOperationStatus::Completed | RegistryOperationStatus::Failed | RegistryOperationStatus::Cancelled)
        });
        before - ops.len()
    }

    /// Get stats
    pub async fn stats(&self) -> RegistryClientStats {
        let ops = self.active_operations.read().await;
        let (mut active, mut completed, mut failed) = (0usize, 0usize, 0usize);
        for op in ops.values() {
            let status = op.status.read().unwrap();
            match *status {
                RegistryOperationStatus::Pulling | RegistryOperationStatus::Pushing | RegistryOperationStatus::Resolving => active += 1,
                RegistryOperationStatus::Completed => completed += 1,
                RegistryOperationStatus::Failed => failed += 1,
                _ => {}
            }
        }
        RegistryClientStats { total_operations: ops.len(), active_operations: active, completed_operations: completed, failed_operations: failed }
    }
}

impl Clone for NeuralRegistryClient {
    fn clone(&self) -> Self {
        let (progress_tx, _) = mpsc::channel(100);
        Self { config: self.config.clone(), client: self.client.clone(), auth_token: None, active_operations: self.active_operations.clone(), progress_tx, semaphore: self.semaphore.clone() }
    }
}

/// Registry client stats
#[derive(Debug, Clone)]
pub struct RegistryClientStats {
    pub total_operations: usize,
    pub active_operations: usize,
    pub completed_operations: usize,
    pub failed_operations: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_coordinate() {
        let coord = RegistryCoordinate::high_priority();
        assert_eq!(coord.network_tier, 900);
        assert_eq!(coord.reliability_score, 0.99);
        let score = coord.priority_score();
        assert!(score > 0);
    }

    #[test]
    fn test_neural_digest() {
        let digest = NeuralDigest::new("sha256", "1234abcdef");
        assert_eq!(digest.algorithm, "sha256");
        assert!(digest.is_valid());
        let from_str = NeuralDigest::from_str("sha256:1234abcdef").unwrap();
        assert_eq!(digest, from_str);
    }

    #[test]
    fn test_neural_model_name() {
        let name = NeuralModelName::new("registry.example.com", "library", "test-model", "latest");
        assert!(name.is_fully_qualified());
        assert_eq!(name.host, "registry.example.com");
    }

    #[test]
    fn test_neural_chunk_range() {
        let chunk = NeuralChunkRange::new(0, 1023);
        assert_eq!(chunk.size(), 1024);
    }

    #[test]
    fn test_registry_progress() {
        let progress = RegistryProgress::new(Uuid::new_v4());
        assert_eq!(progress.percentage(), 0.0);
    }

    #[test]
    fn test_registry_error() {
        let err = RegistryError::ModelNotFound("test".to_string());
        assert_eq!(err.to_string(), "Model not found: test");
    }
}
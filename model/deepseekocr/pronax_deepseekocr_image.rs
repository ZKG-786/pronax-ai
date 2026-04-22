use std::io::Cursor;

use image::{imageops, DynamicImage, ImageBuffer, Rgba, GrayImage, Luma};

use crate::convert::pronax_converter_core::ConversionCoordinate;
use crate::fs::ggml::pronax_ggml_types::SpatialTensorMetadata;
use crate::model::pronax_image_processor::{ImageDimensions3D, ImageProcessor3D, PadColor, ResizeMethod};

/// DeepSeekOCR image processing errors
#[derive(Debug, Clone)]
pub enum DeepSeekOcrError {
    InvalidImage(String),
    ProcessingError(String),
    DimensionError(String),
    NormalizationError(String),
}

impl std::fmt::Display for DeepSeekOcrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidImage(s) => write!(f, "Invalid image: {}", s),
            Self::ProcessingError(s) => write!(f, "Processing error: {}", s),
            Self::DimensionError(s) => write!(f, "Dimension error: {}", s),
            Self::NormalizationError(s) => write!(f, "Normalization error: {}", s),
        }
    }
}

impl std::error::Error for DeepSeekOcrError {}

/// Aspect ratio representation for tiling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileAspectRatio {
    pub x: u32,
    pub y: u32,
}

impl TileAspectRatio {
    /// Create new ratio
    pub fn new(x: u32, y: u32) -> Self {
        Self { x, y }
    }
    
    /// Total tiles
    pub fn total_tiles(&self) -> u32 {
        self.x * self.y
    }
    
    /// Aspect ratio as float
    pub fn as_f32(&self) -> f32 {
        self.x as f32 / self.y as f32
    }
    
    /// 3D spatial metadata
    pub fn to_spatial(&self, tile_size: u32) -> SpatialTensorMetadata {
        SpatialTensorMetadata::new(
            self.x * tile_size,
            self.y * tile_size,
            3,
        )
    }
}

/// 3D-aware image tile with spatial position
#[derive(Debug, Clone)]
pub struct SpatialImageTile {
    /// Tile image data
    pub image: DynamicImage,
    /// Position in tile grid
    pub grid_position: (u32, u32),
    /// 3D spatial coordinate
    pub spatial_coord: ConversionCoordinate,
    /// Normalized pixel data [C, H, W]
    pub normalized_data: Vec<f32>,
}

impl SpatialImageTile {
    /// Get tile dimensions
    pub fn dimensions(&self) -> ImageDimensions3D {
        let (w, h) = (self.image.width(), self.image.height());
        ImageDimensions3D::new_2d(w, h, 3)
    }
}

/// DeepSeekOCR image processing configuration
#[derive(Debug, Clone, Copy)]
pub struct DeepSeekOcrConfig {
    /// Minimum number of tiles
    pub min_tiles: u32,
    /// Maximum number of tiles
    pub max_tiles: u32,
    /// Size of each tile (square)
    pub tile_size: u32,
    /// Base size for thumbnail
    pub base_size: u32,
    /// Image normalization mean
    pub mean: [f32; 3],
    /// Image normalization std
    pub std: [f32; 3],
    /// 3D spatial depth for tiles
    pub spatial_depth: u8,
}

impl DeepSeekOcrConfig {
    /// Default DeepSeekOCR configuration
    pub fn default_ocr() -> Self {
        Self {
            min_tiles: 2,
            max_tiles: 9,
            tile_size: 640,
            base_size: 1024,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
            spatial_depth: 16,
        }
    }
    
    /// Generate all valid tile ratios
    pub fn generate_target_ratios(&self) -> Vec<TileAspectRatio> {
        let mut ratios = Vec::new();
        
        for n in self.min_tiles..=self.max_tiles {
            for i in 1..=n {
                for j in 1..=n {
                    let tiles = i * j;
                    if tiles >= self.min_tiles && tiles <= self.max_tiles {
                        let ratio = TileAspectRatio::new(i, j);
                        if !ratios.contains(&ratio) {
                            ratios.push(ratio);
                        }
                    }
                }
            }
        }
        
        ratios
    }
}

impl Default for DeepSeekOcrConfig {
    fn default() -> Self {
        Self::default_ocr()
    }
}

/// 3D-aware DeepSeekOCR image processor
pub struct DeepSeekOcrImageProcessor {
    pub config: DeepSeekOcrConfig,
}

impl DeepSeekOcrImageProcessor {
    /// Create new processor
    pub fn new(config: DeepSeekOcrConfig) -> Self {
        Self { config }
    }
    
    /// Process image for OCR - returns tiles and thumbnail
    pub fn process_image(
        &self,
        image_data: &[u8],
    ) -> Result<ProcessedImage3D, DeepSeekOcrError> {
        // Decode image
        let img = image::load_from_memory(image_data)
            .map_err(|e| DeepSeekOcrError::InvalidImage(format!("Failed to decode: {}", e)))?;
        
        let (width, height) = (img.width(), img.height());
        
        // Generate valid tile ratios
        let target_ratios = self.config.generate_target_ratios();
        
        // Find best aspect ratio
        let best_ratio = self.find_best_aspect_ratio(&target_ratios, width, height);
        
        let target_width = self.config.tile_size * best_ratio.x;
        let target_height = self.config.tile_size * best_ratio.y;
        let num_blocks = best_ratio.total_tiles();
        
        // Resize image to target dimensions
        let resized = ImageProcessor3D::resize(
            &img,
            ImageDimensions3D::new_2d(target_width, target_height, 3),
            ResizeMethod::Bilinear,
        ).map_err(|e| DeepSeekOcrError::ProcessingError(e.to_string()))?;
        
        // Extract tiles with 3D spatial metadata
        let mut tiles: Vec<SpatialImageTile> = Vec::with_capacity(num_blocks as usize);
        let mut all_patches_data: Vec<f32> = Vec::new();
        
        for block_idx in 0..num_blocks {
            let tile_x = block_idx % best_ratio.x;
            let tile_y = block_idx / best_ratio.x;
            
            let src_x = tile_x * self.config.tile_size;
            let src_y = tile_y * self.config.tile_size;
            
            // Extract tile
            let tile_img = imageops::crop_imm(
                &resized.to_rgba8(),
                src_x,
                src_y,
                self.config.tile_size,
                self.config.tile_size,
            ).to_image();
            
            // Normalize tile
            let normalized = self.normalize_tile(&tile_img)?;
            
            // 3D spatial coordinate
            let spatial = ConversionCoordinate::new(
                (tile_x * self.config.tile_size) as u64,
                (tile_y * self.config.tile_size) as u16,
                block_idx as u8,
                1.0,
            );
            
            // Append to combined data
            all_patches_data.extend(&normalized);
            
            tiles.push(SpatialImageTile {
                image: DynamicImage::ImageRgba8(tile_img),
                grid_position: (tile_x, tile_y),
                spatial_coord: spatial,
                normalized_data: normalized,
            });
        }
        
        // Create thumbnail
        let thumbnail = self.create_thumbnail(&img)?;
        
        Ok(ProcessedImage3D {
            tiles,
            thumbnail,
            tile_ratio: best_ratio,
            combined_tiles_data: all_patches_data,
            spatial_metadata: best_ratio.to_spatial(self.config.tile_size),
        })
    }
    
    /// Find best aspect ratio for tiling
    fn find_best_aspect_ratio(
        &self,
        target_ratios: &[TileAspectRatio],
        width: u32,
        height: u32,
    ) -> TileAspectRatio {
        let real_ratio = width as f32 / height as f32;
        let mut best_diff = f32::MAX;
        let mut best = TileAspectRatio::new(1, 1);
        
        for target in target_ratios {
            let target_ratio = target.as_f32();
            let diff = (real_ratio - target_ratio).abs();
            
            if diff < best_diff {
                best_diff = diff;
                best = *target;
            } else if (diff - best_diff).abs() < f32::EPSILON {
                // Tie-breaker: prefer larger tile count if image is big
                let pixel_count = width * height;
                let threshold = (self.config.tile_size * self.config.tile_size * best.x * best.y) / 2;
                
                if pixel_count as f32 > 0.5 * threshold as f32 {
                    // Prefer the one with more tiles if image is large
                    if target.total_tiles() > best.total_tiles() {
                        best = *target;
                    }
                }
            }
        }
        
        best
    }
    
    /// Normalize a single tile
    fn normalize_tile(&self, tile: &ImageBuffer<Rgba<u8>, Vec<u8>>) -> Result<Vec<f32>, DeepSeekOcrError> {
        let (w, h) = (tile.width(), tile.height());
        let mut data = Vec::with_capacity((w * h * 3) as usize);
        
        // Channel-first order [C, H, W]
        let mut r_vals = Vec::with_capacity((w * h) as usize);
        let mut g_vals = Vec::with_capacity((w * h) as usize);
        let mut b_vals = Vec::with_capacity((w * h) as usize);
        
        for y in 0..h {
            for x in 0..w {
                let pixel = tile.get_pixel(x, y);
                let r = pixel[0] as f32 / 255.0;
                let g = pixel[1] as f32 / 255.0;
                let b = pixel[2] as f32 / 255.0;
                
                r_vals.push((r - self.config.mean[0]) / self.config.std[0]);
                g_vals.push((g - self.config.mean[1]) / self.config.std[1]);
                b_vals.push((b - self.config.mean[2]) / self.config.std[2]);
            }
        }
        
        data.extend(r_vals);
        data.extend(g_vals);
        data.extend(b_vals);
        
        Ok(data)
    }
    
    /// Create thumbnail image
    fn create_thumbnail(&self, img: &DynamicImage) -> Result<Thumbnail3D, DeepSeekOcrError> {
        // Composite over gray background
        let gray = PadColor::RGB(127, 127, 127);
        let composited = ImageProcessor3D::composite(img, gray)
            .map_err(|e| DeepSeekOcrError::ProcessingError(e.to_string()))?;
        
        // Pad to base size
        let padded = ImageProcessor3D::pad(
            &composited,
            ImageDimensions3D::new_2d(self.config.base_size, self.config.base_size, 3),
            gray,
            ResizeMethod::Bilinear,
        ).map_err(|e| DeepSeekOcrError::ProcessingError(e.to_string()))?;
        
        // Normalize
        let normalized = self.normalize_full_image(&padded)?;
        
        Ok(Thumbnail3D {
            image: padded,
            normalized_data: normalized,
            dimensions: ImageDimensions3D::new_2d(self.config.base_size, self.config.base_size, 3),
            spatial_position: ConversionCoordinate::new(0, 0, self.config.spatial_depth, 1.0),
        })
    }
    
    /// Normalize full image for thumbnail
    fn normalize_full_image(&self, img: &DynamicImage) -> Result<Vec<f32>, DeepSeekOcrError> {
        let rgba = img.to_rgba8();
        let (w, h) = (rgba.width(), rgba.height());
        let mut data = Vec::with_capacity((w * h * 3) as usize);
        
        let mut r_vals = Vec::with_capacity((w * h) as usize);
        let mut g_vals = Vec::with_capacity((w * h) as usize);
        let mut b_vals = Vec::with_capacity((w * h) as usize);
        
        for y in 0..h {
            for x in 0..w {
                let pixel = rgba.get_pixel(x, y);
                let r = pixel[0] as f32 / 255.0;
                let g = pixel[1] as f32 / 255.0;
                let b = pixel[2] as f32 / 255.0;
                
                r_vals.push((r - self.config.mean[0]) / self.config.std[0]);
                g_vals.push((g - self.config.mean[1]) / self.config.std[1]);
                b_vals.push((b - self.config.mean[2]) / self.config.std[2]);
            }
        }
        
        data.extend(r_vals);
        data.extend(g_vals);
        data.extend(b_vals);
        
        Ok(data)
    }
    
    /// Process batch of images
    pub fn process_batch(
        &self,
        images: &[Vec<u8>],
    ) -> Result<Vec<ProcessedImage3D>, DeepSeekOcrError> {
        images.iter()
            .map(|img| self.process_image(img))
            .collect()
    }
}

impl Default for DeepSeekOcrImageProcessor {
    fn default() -> Self {
        Self::new(DeepSeekOcrConfig::default())
    }
}

/// Processed image with 3D spatial metadata
#[derive(Debug, Clone)]
pub struct ProcessedImage3D {
    /// Individual tiles
    pub tiles: Vec<SpatialImageTile>,
    /// Thumbnail image
    pub thumbnail: Thumbnail3D,
    /// Tile aspect ratio used
    pub tile_ratio: TileAspectRatio,
    /// Combined tiles data for model input
    pub combined_tiles_data: Vec<f32>,
    /// 3D spatial metadata
    pub spatial_metadata: SpatialTensorMetadata,
}

impl ProcessedImage3D {
    /// Get tile tensor shape [blocks, 3, tile_size, tile_size]
    pub fn tiles_shape(&self, tile_size: u32) -> Vec<usize> {
        vec![
            self.tiles.len(),
            3,
            tile_size as usize,
            tile_size as usize,
        ]
    }
    
    /// Get thumbnail tensor shape [3, base_size, base_size]
    pub fn thumbnail_shape(&self, base_size: u32) -> Vec<usize> {
        vec![3, base_size as usize, base_size as usize]
    }
    
    /// Get 3D spatial bounds
    pub fn spatial_bounds(&self) -> SpatialTensorMetadata {
        self.spatial_metadata
    }
    
    /// Total number of tiles
    pub fn num_tiles(&self) -> usize {
        self.tiles.len()
    }
}

/// Thumbnail image with 3D metadata
#[derive(Debug, Clone)]
pub struct Thumbnail3D {
    /// Thumbnail image
    pub image: DynamicImage,
    /// Normalized pixel data
    pub normalized_data: Vec<f32>,
    /// Image dimensions
    pub dimensions: ImageDimensions3D,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
}

/// 3D OCR processing result
#[derive(Debug, Clone)]
pub struct OcrResult3D {
    /// Recognized text
    pub text: String,
    /// Confidence scores
    pub confidence: Vec<f32>,
    /// 3D spatial positions for each character/region
    pub char_positions: Vec<ConversionCoordinate>,
    /// Page layout information
    pub layout: PageLayout3D,
}

/// Page layout with 3D spatial information
#[derive(Debug, Clone)]
pub struct PageLayout3D {
    /// Text regions with bounding boxes
    pub regions: Vec<TextRegion3D>,
    /// Page dimensions
    pub page_dimensions: ImageDimensions3D,
    /// 3D depth layer for text hierarchy
    pub depth_layers: u8,
}

/// Text region with 3D spatial bounding box
#[derive(Debug, Clone)]
pub struct TextRegion3D {
    /// Region bounding box [x1, y1, x2, y2]
    pub bbox: [u32; 4],
    /// Region text content
    pub text: String,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
    /// Text confidence
    pub confidence: f32,
    /// Region type (header, paragraph, etc.)
    pub region_type: RegionType,
}

/// Text region type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionType {
    Header,
    Paragraph,
    Caption,
    Table,
    Figure,
    Footer,
    Unknown,
}

impl RegionType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Header => "header",
            Self::Paragraph => "paragraph",
            Self::Caption => "caption",
            Self::Table => "table",
            Self::Figure => "figure",
            Self::Footer => "footer",
            Self::Unknown => "unknown",
        }
    }
}

/// Utility functions for DeepSeekOCR
pub mod ocr_utils {
    use super::*;
    
    /// Calculate optimal tile configuration for image size
    pub fn optimal_tile_config(width: u32, height: u32) -> (u32, u32) {
        let aspect = width as f32 / height as f32;
        
        // Common tile configurations
        let configs = [
            (1, 1), (1, 2), (2, 1), (2, 2),
            (1, 3), (3, 1), (2, 3), (3, 2),
            (3, 3), (1, 4), (4, 1), (2, 4), (4, 2),
        ];
        
        let mut best = (1, 1);
        let mut best_diff = f32::MAX;
        
        for (tx, ty) in configs {
            let target_aspect = tx as f32 / ty as f32;
            let diff = (aspect - target_aspect).abs();
            
            if diff < best_diff && tx * ty <= 9 {
                best_diff = diff;
                best = (tx, ty);
            }
        }
        
        best
    }
    
    /// Estimate memory requirement for OCR processing
    pub fn estimate_memory(width: u32, height: u32, config: &DeepSeekOcrConfig) -> u64 {
        let (tx, ty) = optimal_tile_config(width, height);
        let num_tiles = tx * ty;
        
        // Tiles memory
        let tiles_mem = (num_tiles * config.tile_size * config.tile_size * 3 * 4) as u64;
        
        // Thumbnail memory
        let thumb_mem = (config.base_size * config.base_size * 3 * 4) as u64;
        
        tiles_mem + thumb_mem
    }
    
    /// Compute 3D spatial hash for image region
    pub fn compute_spatial_hash(x: u32, y: u32, z: u8) -> u64 {
        let mut hash = 0u64;
        hash |= (x as u64) << 32;
        hash |= (y as u64) << 16;
        hash |= z as u64;
        hash
    }
    
    /// Validate image for OCR processing
    pub fn validate_image(image_data: &[u8]) -> Result<(u32, u32), DeepSeekOcrError> {
        let img = image::load_from_memory(image_data)
            .map_err(|e| DeepSeekOcrError::InvalidImage(format!("Invalid image: {}", e)))?;
        
        let (w, h) = (img.width(), img.height());
        
        if w < 32 || h < 32 {
            return Err(DeepSeekOcrError::InvalidImage(
                "Image too small for OCR".to_string()
            ));
        }
        
        Ok((w, h))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tile_aspect_ratio() {
        let ratio = TileAspectRatio::new(2, 3);
        assert_eq!(ratio.total_tiles(), 6);
        assert!((ratio.as_f32() - 0.6667).abs() < 0.01);
    }
    
    #[test]
    fn test_config() {
        let config = DeepSeekOcrConfig::default();
        let ratios = config.generate_target_ratios();
        
        assert!(!ratios.is_empty());
        assert!(ratios.iter().all(|r| r.total_tiles() <= config.max_tiles));
    }
    
    #[test]
    fn test_best_aspect_ratio() {
        let config = DeepSeekOcrConfig::default();
        let ratios = config.generate_target_ratios();
        
        let best = config.find_best_aspect_ratio(&ratios, 1920, 1080);
        
        // 1920x1080 has aspect ratio ~1.77
        assert!(best.as_f32() > 1.0);
    }
    
    #[test]
    fn test_optimal_tile_config() {
        let (tx, ty) = ocr_utils::optimal_tile_config(1920, 1080);
        let aspect = tx as f32 / ty as f32;
        
        assert!(tx * ty <= 9);
        assert!(aspect > 1.0);
    }
    
    #[test]
    fn test_region_type() {
        assert_eq!(RegionType::Header.as_str(), "header");
        assert_eq!(RegionType::Paragraph.as_str(), "paragraph");
    }
    
    #[test]
    fn test_spatial_hash() {
        let hash1 = ocr_utils::compute_spatial_hash(100, 200, 5);
        let hash2 = ocr_utils::compute_spatial_hash(100, 200, 5);
        let hash3 = ocr_utils::compute_spatial_hash(101, 200, 5);
        
        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }
    
    #[test]
    fn test_estimate_memory() {
        let config = DeepSeekOcrConfig::default();
        let mem = ocr_utils::estimate_memory(1920, 1080, &config);
        
        assert!(mem > 0);
        // Should be roughly: tiles (6 * 640 * 640 * 3 * 4) + thumbnail (1024 * 1024 * 3 * 4)
        assert!(mem > 10_000_000);
    }
}

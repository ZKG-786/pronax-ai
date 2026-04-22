use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;

use crate::fs::ggml::pronax_ggml_types::SpatialTensorMetadata;

/// Default buffer size for neural streaming (32KB for optimal performance)
pub const TITAN_DEFAULT_BUFFER_SIZE: usize = 32 * 1024;

/// Minimum buffer size for zero-copy operations
pub const TITAN_MIN_BUFFER_SIZE: usize = 4 * 1024;

/// Maximum buffer size for large tensor operations
pub const TITAN_MAX_BUFFER_SIZE: usize = 256 * 1024;

/// 3D spatial read context for tensor streaming
#[derive(Debug, Clone, Copy)]
pub struct NeuralReadContext {
    /// Current 3D position in the tensor
    pub position: SpatialTensorMetadata,
    /// Total tensor dimensions
    pub total_dimensions: SpatialTensorMetadata,
    /// Bytes per element
    pub element_size: usize,
    /// Preferred read alignment
    pub alignment: usize,
    /// Whether to use memory mapping
    pub use_mmap: bool,
}

impl NeuralReadContext {
    /// Create new read context for 3D tensor
    pub fn new(width: u32, height: u32, depth: u32, element_size: usize) -> Self {
        Self {
            position: SpatialTensorMetadata::new(0, 0, 0),
            total_dimensions: SpatialTensorMetadata::new(width, height, depth),
            element_size,
            alignment: 64,
            use_mmap: true,
        }
    }

    /// Calculate remaining bytes to read
    #[inline]
    pub fn remaining_bytes(&self) -> u64 {
        let total_volume = self.total_dimensions.volume();
        let current_volume = self.position.volume();
        let remaining = total_volume.saturating_sub(current_volume);
        remaining * self.element_size as u64
    }

    /// Calculate progress (0.0 to 1.0)
    #[inline]
    pub fn progress(&self) -> f64 {
        let total = self.total_dimensions.volume();
        if total == 0 {
            return 1.0;
        }
        let current = self.position.volume();
        current as f64 / total as f64
    }

    /// Check if reading is complete
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.position.volume() >= self.total_dimensions.volume()
    }

    /// Advance position by bytes
    pub fn advance(&mut self, bytes: u64) {
        let elements = bytes / self.element_size as u64;
        let width = self.total_dimensions.width as u64;
        let height = self.total_dimensions.height as u64;
        
        let new_volume = self.position.volume() + elements;
        
        // Calculate new 3D position
        self.position.depth = (new_volume / (width * height)) as u32;
        let remainder = new_volume % (width * height);
        self.position.height = (remainder / width) as u32;
        self.position.width = (remainder % width) as u32;
    }

    /// With memory mapping preference
    pub fn with_mmap(mut self, use_mmap: bool) -> Self {
        self.use_mmap = use_mmap;
        self
    }

    /// With custom alignment
    pub fn with_alignment(mut self, alignment: usize) -> Self {
        self.alignment = alignment;
        self
    }
}

impl Default for NeuralReadContext {
    fn default() -> Self {
        Self::new(1, 1, 1, 4)
    }
}

/// Titan-level neural buffered reader with 3D spatial tracking
pub struct TitanNeuralReader<R: Read> {
    /// Inner reader
    inner: R,
    /// Buffer for zero-copy operations
    buffer: Box<[u8]>,
    /// Current position in buffer
    buffer_pos: usize,
    /// Filled bytes in buffer
    buffer_filled: usize,
    /// Total bytes read from source
    stream_offset: u64,
    /// 3D spatial read context
    spatial_context: NeuralReadContext,
    /// Whether buffer is dirty (needs refill)
    buffer_dirty: bool,
}

impl<R: Read> TitanNeuralReader<R> {
    /// Create new neural reader with default buffer size
    pub fn new(inner: R) -> Self {
        Self::with_capacity(inner, TITAN_DEFAULT_BUFFER_SIZE)
    }

    /// Create with custom buffer capacity
    pub fn with_capacity(inner: R, capacity: usize) -> Self {
        let capacity = capacity.clamp(TITAN_MIN_BUFFER_SIZE, TITAN_MAX_BUFFER_SIZE);
        
        Self {
            inner,
            buffer: vec![0u8; capacity].into_boxed_slice(),
            buffer_pos: 0,
            buffer_filled: 0,
            stream_offset: 0,
            spatial_context: NeuralReadContext::default(),
            buffer_dirty: true,
        }
    }

    /// With 3D spatial context
    pub fn with_spatial_context(mut self, context: NeuralReadContext) -> Self {
        self.spatial_context = context;
        self
    }

    /// Get current stream offset
    #[inline]
    pub fn stream_offset(&self) -> u64 {
        self.stream_offset
    }

    /// Get spatial context
    #[inline]
    pub fn spatial_context(&self) -> &NeuralReadContext {
        &self.spatial_context
    }

    /// Get mutable spatial context
    #[inline]
    pub fn spatial_context_mut(&mut self) -> &mut NeuralReadContext {
        &mut self.spatial_context
    }

    /// Get buffer capacity
    #[inline]
    pub fn buffer_capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Get available bytes in buffer
    #[inline]
    pub fn buffer_available(&self) -> usize {
        self.buffer_filled.saturating_sub(self.buffer_pos)
    }

    /// Check if buffer needs refill
    #[inline]
    pub fn needs_refill(&self) -> bool {
        self.buffer_dirty || self.buffer_available() == 0
    }

    /// Refill buffer from source
    pub fn refill(&mut self) -> io::Result<usize> {
        // Shift remaining data to front if any
        if self.buffer_pos < self.buffer_filled {
            let remaining = self.buffer_filled - self.buffer_pos;
            self.buffer.copy_within(self.buffer_pos..self.buffer_filled, 0);
            self.buffer_filled = remaining;
        } else {
            self.buffer_filled = 0;
        }
        self.buffer_pos = 0;

        // Read new data
        let bytes_read = self.inner.read(&mut self.buffer[self.buffer_filled..])?;
        self.buffer_filled += bytes_read;
        self.buffer_dirty = false;

        Ok(bytes_read)
    }

    /// Read exact bytes with zero-copy optimization
    pub fn read_exact_neural(&mut self, buf: &mut [u8]) -> io::Result<()> {
        let mut total_read = 0;

        while total_read < buf.len() {
            // Refill if needed
            if self.buffer_available() == 0 {
                let bytes_read = self.refill()?;
                if bytes_read == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "Unexpected end of neural stream",
                    ));
                }
            }

            // Copy from buffer
            let available = self.buffer_available();
            let needed = buf.len() - total_read;
            let to_copy = available.min(needed);

            buf[total_read..total_read + to_copy]
                .copy_from_slice(&self.buffer[self.buffer_pos..self.buffer_pos + to_copy]);

            self.buffer_pos += to_copy;
            total_read += to_copy;
        }

        // Update stream offset and spatial context
        self.stream_offset += total_read as u64;
        self.spatial_context.advance(total_read as u64);

        Ok(())
    }

    /// Read up to buf.len() bytes
    pub fn read_neural(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let mut total_read = 0;

        while total_read < buf.len() {
            // Refill if needed
            if self.buffer_available() == 0 {
                let bytes_read = self.refill()?;
                if bytes_read == 0 {
                    break; // End of stream
                }
            }

            // Copy from buffer
            let available = self.buffer_available();
            let needed = buf.len() - total_read;
            let to_copy = available.min(needed);

            buf[total_read..total_read + to_copy]
                .copy_from_slice(&self.buffer[self.buffer_pos..self.buffer_pos + to_copy]);

            self.buffer_pos += to_copy;
            total_read += to_copy;
        }

        // Update stream offset and spatial context
        self.stream_offset += total_read as u64;
        self.spatial_context.advance(total_read as u64);

        Ok(total_read)
    }

    /// Read little-endian value
    #[inline]
    pub fn read_neural_le<T: NeuralLittleEndian>(&mut self) -> io::Result<T> {
        T::read_neural_le(self)
    }

    /// Skip bytes (efficiently)
    pub fn skip(&mut self, bytes: u64) -> io::Result<u64> {
        let mut remaining = bytes;

        // Skip from buffer first
        let buffer_available = self.buffer_available() as u64;
        if remaining <= buffer_available {
            self.buffer_pos += remaining as usize;
            self.stream_offset += remaining;
            self.spatial_context.advance(remaining);
            return Ok(bytes);
        }

        // Skip buffered bytes
        remaining -= buffer_available as u64;
        self.buffer_pos = self.buffer_filled;

        // For large skips, just read and discard in chunks
        let chunk_size = self.buffer.len();
        let mut skipped = buffer_available;

        while remaining > 0 {
            let to_skip = remaining.min(chunk_size as u64) as usize;
            let mut temp = vec![0u8; to_skip];
            let bytes_read = self.read_neural(&mut temp)?;
            
            if bytes_read == 0 {
                break; // End of stream
            }
            
            skipped += bytes_read as u64;
            remaining -= bytes_read as u64;
        }

        Ok(skipped)
    }

    /// Peek at next byte without consuming
    pub fn peek(&mut self) -> io::Result<Option<u8>> {
        if self.buffer_available() == 0 {
            self.refill()?;
        }

        if self.buffer_available() > 0 {
            Ok(Some(self.buffer[self.buffer_pos]))
        } else {
            Ok(None)
        }
    }

    /// Get a view of available buffered data (zero-copy)
    pub fn buffered_data(&self) -> &[u8] {
        &self.buffer[self.buffer_pos..self.buffer_filled]
    }

    /// Consume into inner reader
    pub fn into_inner(self) -> R {
        self.inner
    }

    /// Get reference to inner reader
    pub fn get_ref(&self) -> &R {
        &self.inner
    }

    /// Get mutable reference to inner reader
    pub fn get_mut(&mut self) -> &mut R {
        &mut self.inner
    }
}

/// Trait for little-endian neural data reading
pub trait NeuralLittleEndian: Sized {
    /// Read from neural reader in little-endian format
    fn read_neural_le<R: Read>(reader: &mut TitanNeuralReader<R>) -> io::Result<Self>;
}

macro_rules! impl_neural_le {
    ($type:ty, $size:expr) => {
        impl NeuralLittleEndian for $type {
            fn read_neural_le<R: Read>(reader: &mut TitanNeuralReader<R>) -> io::Result<Self> {
                let mut bytes = [0u8; $size];
                reader.read_exact_neural(&mut bytes)?;
                Ok(<$type>::from_le_bytes(bytes))
            }
        }
    };
}

impl_neural_le!(u8, 1);
impl_neural_le!(i8, 1);
impl_neural_le!(u16, 2);
impl_neural_le!(i16, 2);
impl_neural_le!(u32, 4);
impl_neural_le!(i32, 4);
impl_neural_le!(f32, 4);
impl_neural_le!(u64, 8);
impl_neural_le!(i64, 8);
impl_neural_le!(f64, 8);

/// Seekable neural reader for file operations
pub struct TitanNeuralFileReader {
    /// Inner buffered reader
    reader: BufReader<File>,
    /// Stream offset tracking
    stream_offset: u64,
    /// 3D spatial context
    spatial_context: NeuralReadContext,
    /// File size (if known)
    file_size: Option<u64>,
}

impl TitanNeuralFileReader {
    /// Open file for neural reading
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        let file_size = if metadata.is_file() {
            Some(metadata.len())
        } else {
            None
        };

        Ok(Self {
            reader: BufReader::with_capacity(TITAN_DEFAULT_BUFFER_SIZE, file),
            stream_offset: 0,
            spatial_context: NeuralReadContext::default(),
            file_size,
        })
    }

    /// With spatial context
    pub fn with_spatial_context(mut self, context: NeuralReadContext) -> Self {
        self.spatial_context = context;
        self
    }

    /// Get file size if known
    #[inline]
    pub fn file_size(&self) -> Option<u64> {
        self.file_size
    }

    /// Get current position
    #[inline]
    pub fn position(&self) -> u64 {
        self.stream_offset
    }

    /// Get spatial context
    #[inline]
    pub fn spatial_context(&self) -> &NeuralReadContext {
        &self.spatial_context
    }

    /// Seek to absolute position
    pub fn seek(&mut self, pos: u64) -> io::Result<u64> {
        self.reader.seek(SeekFrom::Start(pos))?;
        self.stream_offset = pos;
        
        // Reset spatial context position
        self.spatial_context.position = SpatialTensorMetadata::new(0, 0, 0);
        self.spatial_context.advance(pos);
        
        Ok(pos)
    }

    /// Read exact bytes
    pub fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        self.reader.read_exact(buf)?;
        self.stream_offset += buf.len() as u64;
        self.spatial_context.advance(buf.len() as u64);
        Ok(())
    }

    /// Read bytes
    pub fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let bytes_read = self.reader.read(buf)?;
        self.stream_offset += bytes_read as u64;
        self.spatial_context.advance(bytes_read as u64);
        Ok(bytes_read)
    }

    /// Read little-endian value
    #[inline]
    pub fn read_le<T: NeuralLittleEndianFile>(&mut self) -> io::Result<T> {
        T::read_le(self)
    }

    /// Read string with length prefix
    pub fn read_string(&mut self) -> io::Result<String> {
        let len = self.read_le::<u64>()?;
        let mut bytes = vec![0u8; len as usize];
        self.read_exact(&mut bytes)?;
        
        String::from_utf8(bytes).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, e)
        })
    }

    /// Check if at end of file
    pub fn is_eof(&self) -> bool {
        match self.file_size {
            Some(size) => self.stream_offset >= size,
            None => false,
        }
    }

    /// Get progress (0.0 to 1.0) if file size known
    pub fn progress(&self) -> Option<f64> {
        self.file_size.map(|size| {
            if size == 0 {
                1.0
            } else {
                self.stream_offset as f64 / size as f64
            }
        })
    }

    /// Into inner file
    pub fn into_inner(self) -> File {
        self.reader.into_inner()
    }
}

/// Trait for little-endian file reading
pub trait NeuralLittleEndianFile: Sized {
    fn read_le(reader: &mut TitanNeuralFileReader) -> io::Result<Self>;
}

macro_rules! impl_file_le {
    ($type:ty, $size:expr) => {
        impl NeuralLittleEndianFile for $type {
            fn read_le(reader: &mut TitanNeuralFileReader) -> io::Result<Self> {
                let mut bytes = [0u8; $size];
                reader.read_exact(&mut bytes)?;
                Ok(<$type>::from_le_bytes(bytes))
            }
        }
    };
}

impl_file_le!(u8, 1);
impl_file_le!(i8, 1);
impl_file_le!(u16, 2);
impl_file_le!(i16, 2);
impl_file_le!(u32, 4);
impl_file_le!(i32, 4);
impl_file_le!(f32, 4);
impl_file_le!(u64, 8);
impl_file_le!(i64, 8);
impl_file_le!(f64, 8);

/// Memory-mapped neural reader for zero-copy file access
pub struct TitanNeuralMmapReader {
    /// Memory-mapped data
    data: Arc<memmap2::Mmap>,
    /// Current position
    position: usize,
    /// 3D spatial context
    spatial_context: NeuralReadContext,
}

#[cfg(feature = "mmap")]
impl TitanNeuralMmapReader {
    /// Open file with memory mapping
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        
        Ok(Self {
            data: Arc::new(mmap),
            position: 0,
            spatial_context: NeuralReadContext::default(),
        })
    }

    /// With spatial context
    pub fn with_spatial_context(mut self, context: NeuralReadContext) -> Self {
        self.spatial_context = context;
        self
    }

    /// Get data length
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get remaining bytes
    #[inline]
    pub fn remaining(&self) -> usize {
        self.len().saturating_sub(self.position)
    }

    /// Read bytes (zero-copy slice)
    pub fn read_bytes(&mut self, count: usize) -> Option<&[u8]> {
        let end = (self.position + count).min(self.data.len());
        if self.position >= end {
            return None;
        }
        
        let slice = &self.data[self.position..end];
        self.position = end;
        self.spatial_context.advance((end - self.position) as u64);
        
        Some(slice)
    }

    /// Get zero-copy slice of all data
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Get zero-copy slice from position
    pub fn slice_from(&self, offset: usize) -> &[u8] {
        &self.data[offset..]
    }

    /// Seek to position
    pub fn seek(&mut self, pos: usize) {
        self.position = pos.min(self.data.len());
        self.spatial_context.position = SpatialTensorMetadata::new(0, 0, 0);
        self.spatial_context.advance(pos as u64);
    }
}

/// Reader factory for creating appropriate reader type
pub struct TitanNeuralReaderFactory;

impl TitanNeuralReaderFactory {
    /// Create reader for file path (auto-detect best type)
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<TitanNeuralFileReader> {
        TitanNeuralFileReader::open(path)
    }

    /// Create reader with specific buffer size
    pub fn open_buffered<P: AsRef<Path>>(path: P, buffer_size: usize) -> io::Result<TitanNeuralFileReader> {
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        let file_size = if metadata.is_file() {
            Some(metadata.len())
        } else {
            None
        };

        Ok(TitanNeuralFileReader {
            reader: BufReader::with_capacity(buffer_size, file),
            stream_offset: 0,
            spatial_context: NeuralReadContext::default(),
            file_size,
        })
    }

    /// Create memory-mapped reader if feature enabled
    #[cfg(feature = "mmap")]
    pub fn open_mmap<P: AsRef<Path>>(path: P) -> io::Result<TitanNeuralMmapReader> {
        TitanNeuralMmapReader::open(path)
    }
}

/// Type aliases for convenience
pub type NeuralReader<R> = TitanNeuralReader<R>;
pub type NeuralFileReader = TitanNeuralFileReader;
pub type NeuralReaderFactory = TitanNeuralReaderFactory;
pub type ReadContext = NeuralReadContext;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_neural_reader_basic() {
        let data: Vec<u8> = (0..100).map(|i| i as u8).collect();
        let cursor = Cursor::new(data);
        let mut reader = NeuralReader::new(cursor);

        let mut buf = [0u8; 10];
        reader.read_exact_neural(&mut buf).unwrap();
        
        assert_eq!(buf, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(reader.stream_offset(), 10);
    }

    #[test]
    fn test_read_le_values() {
        let data: Vec<u8> = vec![
            0x01, 0x00, // u16: 1
            0x02, 0x00, 0x00, 0x00, // u32: 2
            0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // u64: 3
        ];
        let cursor = Cursor::new(data);
        let mut reader = NeuralReader::new(cursor);

        assert_eq!(reader.read_neural_le::<u16>().unwrap(), 1);
        assert_eq!(reader.read_neural_le::<u32>().unwrap(), 2);
        assert_eq!(reader.read_neural_le::<u64>().unwrap(), 3);
    }

    #[test]
    fn test_spatial_context() {
        let mut ctx = ReadContext::new(256, 256, 128, 4);
        
        assert_eq!(ctx.progress(), 0.0);
        assert!(!ctx.is_complete());
        
        // Advance by one "slice"
        ctx.advance(256 * 256 * 4); // width * height * element_size
        
        assert_eq!(ctx.position.depth, 1);
        assert_eq!(ctx.position.height, 0);
        assert_eq!(ctx.position.width, 0);
    }

    #[test]
    fn test_buffer_management() {
        let data = vec![0u8; 1000];
        let cursor = Cursor::new(data);
        let mut reader = NeuralReader::with_capacity(cursor, 256);

        assert_eq!(reader.buffer_capacity(), 256);
        assert!(reader.needs_refill());

        let mut buf = [0u8; 10];
        reader.read_neural(&mut buf).unwrap();
        
        assert_eq!(reader.buffer_available(), 246); // 256 - 10
        assert!(!reader.needs_refill());
    }

    #[test]
    fn test_skip_operation() {
        let data: Vec<u8> = (0..100).map(|i| i as u8).collect();
        let cursor = Cursor::new(data);
        let mut reader = NeuralReader::new(cursor);

        let skipped = reader.skip(50).unwrap();
        assert_eq!(skipped, 50);
        assert_eq!(reader.stream_offset(), 50);

        let mut buf = [0u8; 10];
        reader.read_exact_neural(&mut buf).unwrap();
        assert_eq!(buf[0], 50);
    }

    #[test]
    fn test_peek_operation() {
        let data = vec![42u8, 43, 44];
        let cursor = Cursor::new(data);
        let mut reader = NeuralReader::new(cursor);

        assert_eq!(reader.peek().unwrap(), Some(42));
        assert_eq!(reader.peek().unwrap(), Some(42)); // Peek doesn't consume
        
        let mut buf = [0u8; 1];
        reader.read_exact_neural(&mut buf).unwrap();
        assert_eq!(buf[0], 42);
        
        assert_eq!(reader.peek().unwrap(), Some(43));
    }
}
"""
Multimodal processor for handling text and images from documents.
Optimized for Windows CPU-only operation.
"""

import os
import base64
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import logging
from dataclasses import dataclass
from PIL import Image
import io
import numpy as np
from datetime import datetime
import tempfile
import fitz  # PyMuPDF for PDF processing

logger = logging.getLogger(__name__)

@dataclass
class MultimodalChunk:
    """Container for multimodal content"""
    text: str
    images: List[Dict[str, Any]]  # List of image metadata
    page_number: int
    chunk_id: str
    metadata: Dict[str, Any]
    embeddings: Optional[Dict[str, np.ndarray]] = None

class MultimodalProcessor:
    """
    Processes documents to extract both text and images.
    CPU-optimized for Windows without CUDA.
    """
    
    def __init__(self, 
                 extract_images: bool = True,
                 image_quality: int = 85,
                 max_image_size: Tuple[int, int] = (1024, 1024),
                 cache_dir: Optional[str] = None):
        """
        Initialize multimodal processor.
        
        Args:
            extract_images: Whether to extract images from documents
            image_quality: JPEG quality for image compression (1-100)
            max_image_size: Maximum image dimensions (width, height)
            cache_dir: Directory for caching processed images
        """
        self.extract_images = extract_images
        self.image_quality = image_quality
        self.max_image_size = max_image_size
        
        # Setup cache directory for Windows
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            # Use Windows temp directory
            self.cache_dir = Path(tempfile.gettempdir()) / "rag_multimodal_cache"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Multimodal cache directory: {self.cache_dir}")
        
        # CPU-optimized settings
        self.use_cpu = True
        self.batch_size = 1  # Small batch for CPU
        
    def process_pdf(self, pdf_path: Union[str, Path]) -> List[MultimodalChunk]:
        """
        Extract text and images from PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of multimodal chunks
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
            
        chunks = []
        
        try:
            # Open PDF with PyMuPDF
            doc = fitz.open(str(pdf_path))
            
            for page_num, page in enumerate(doc):
                # Extract text
                text = page.get_text()
                
                # Extract images if enabled
                images = []
                if self.extract_images:
                    images = self._extract_page_images(page, pdf_path.stem, page_num)
                
                # Create chunk for this page
                chunk_id = self._generate_chunk_id(pdf_path.stem, page_num)
                
                chunk = MultimodalChunk(
                    text=text,
                    images=images,
                    page_number=page_num + 1,
                    chunk_id=chunk_id,
                    metadata={
                        "source": str(pdf_path),
                        "page": page_num + 1,
                        "total_pages": len(doc),
                        "extracted_at": datetime.now().isoformat()
                    }
                )
                chunks.append(chunk)
                
            doc.close()
            logger.info(f"Processed {len(chunks)} pages from {pdf_path.name}")
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise
            
        return chunks
    
    def _extract_page_images(self, page, doc_name: str, page_num: int) -> List[Dict[str, Any]]:
        """
        Extract images from a PDF page.
        
        Args:
            page: PyMuPDF page object
            doc_name: Document name for caching
            page_num: Page number
            
        Returns:
            List of image metadata
        """
        images = []
        
        try:
            # Get list of images on the page
            image_list = page.get_images()
            
            for img_index, img_info in enumerate(image_list):
                try:
                    # Extract image
                    xref = img_info[0]
                    pix = fitz.Pixmap(page.parent, xref)
                    
                    # Convert to PIL Image
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                    else:  # CMYK
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                        img_data = pix.tobytes("png")
                    
                    img = Image.open(io.BytesIO(img_data))
                    
                    # Resize if needed
                    img = self._resize_image(img)
                    
                    # Save to cache
                    image_path = self._cache_image(img, doc_name, page_num, img_index)
                    
                    # Create metadata
                    images.append({
                        "path": str(image_path),
                        "width": img.width,
                        "height": img.height,
                        "format": img.format or "PNG",
                        "page": page_num + 1,
                        "index": img_index
                    })
                    
                    pix = None  # Free memory
                    
                except Exception as e:
                    logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Failed to extract images from page {page_num}: {e}")
            
        return images
    
    def _resize_image(self, img: Image.Image) -> Image.Image:
        """
        Resize image if it exceeds maximum dimensions.
        
        Args:
            img: PIL Image
            
        Returns:
            Resized image
        """
        if img.width > self.max_image_size[0] or img.height > self.max_image_size[1]:
            img.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
        return img
    
    def _cache_image(self, img: Image.Image, doc_name: str, page_num: int, img_index: int) -> Path:
        """
        Save image to cache directory.
        
        Args:
            img: PIL Image
            doc_name: Document name
            page_num: Page number
            img_index: Image index on page
            
        Returns:
            Path to cached image
        """
        # Create subdirectory for document
        doc_dir = self.cache_dir / doc_name
        doc_dir.mkdir(exist_ok=True)
        
        # Generate filename
        filename = f"page_{page_num:03d}_img_{img_index:02d}.jpg"
        image_path = doc_dir / filename
        
        # Save image
        img.save(image_path, "JPEG", quality=self.image_quality, optimize=True)
        
        return image_path
    
    def _generate_chunk_id(self, doc_name: str, page_num: int) -> str:
        """
        Generate unique chunk ID.
        
        Args:
            doc_name: Document name
            page_num: Page number
            
        Returns:
            Unique chunk ID
        """
        content = f"{doc_name}_page_{page_num}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def process_image_file(self, image_path: Union[str, Path]) -> MultimodalChunk:
        """
        Process a standalone image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            MultimodalChunk containing the image
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        try:
            # Open and process image
            img = Image.open(image_path)
            img = self._resize_image(img)
            
            # Cache the image
            cached_path = self._cache_image(
                img, 
                image_path.stem, 
                0, 
                0
            )
            
            # Create chunk
            chunk = MultimodalChunk(
                text="",  # No text for standalone image
                images=[{
                    "path": str(cached_path),
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                    "original_path": str(image_path)
                }],
                page_number=1,
                chunk_id=self._generate_chunk_id(image_path.stem, 0),
                metadata={
                    "source": str(image_path),
                    "type": "image",
                    "extracted_at": datetime.now().isoformat()
                }
            )
            
            return chunk
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise
    
    def encode_image_base64(self, image_path: Union[str, Path]) -> str:
        """
        Encode image to base64 string for API transmission.
        
        Args:
            image_path: Path to image
            
        Returns:
            Base64 encoded string
        """
        image_path = Path(image_path)
        
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    
    def clear_cache(self, doc_name: Optional[str] = None):
        """
        Clear cached images.
        
        Args:
            doc_name: Specific document to clear, or None for all
        """
        if doc_name:
            doc_dir = self.cache_dir / doc_name
            if doc_dir.exists():
                import shutil
                shutil.rmtree(doc_dir)
                logger.info(f"Cleared cache for {doc_name}")
        else:
            # Clear all cache
            import shutil
            for item in self.cache_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
            logger.info("Cleared all multimodal cache")


class MultimodalEmbedder:
    """
    Generate embeddings for multimodal content using CPU-optimized models.
    """
    
    def __init__(self, 
                 text_model: Optional[str] = None,
                 vision_model: Optional[str] = None,
                 device: str = "cpu"):
        """
        Initialize embedder with CPU-optimized settings.
        
        Args:
            text_model: Name of text embedding model
            vision_model: Name of vision embedding model  
            device: Device to use (always 'cpu' for Windows without CUDA)
        """
        self.device = "cpu"  # Force CPU
        self.text_model = text_model
        self.vision_model = vision_model
        
        # Lazy loading of models
        self._text_embedder = None
        self._vision_embedder = None
        
        logger.info(f"Multimodal embedder initialized for CPU")
    
    def embed_chunk(self, chunk: MultimodalChunk) -> MultimodalChunk:
        """
        Generate embeddings for a multimodal chunk.
        
        Args:
            chunk: MultimodalChunk to embed
            
        Returns:
            Chunk with embeddings added
        """
        embeddings = {}
        
        # Embed text if present
        if chunk.text:
            embeddings["text"] = self._embed_text(chunk.text)
        
        # Embed images if present
        if chunk.images:
            image_embeddings = []
            for img_meta in chunk.images:
                img_emb = self._embed_image(img_meta["path"])
                image_embeddings.append(img_emb)
            
            # Average image embeddings
            if image_embeddings:
                embeddings["image"] = np.mean(image_embeddings, axis=0)
        
        chunk.embeddings = embeddings
        return chunk
    
    def _embed_text(self, text: str) -> np.ndarray:
        """
        Generate text embedding.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Placeholder - integrate with your existing embedding system
        # For CPU, use a lightweight model
        if not self._text_embedder:
            logger.info("Loading text embedder for CPU...")
            # Initialize your text embedding model here
            pass
        
        # Generate embedding
        # return self._text_embedder.encode(text)
        
        # Dummy embedding for now
        return np.random.randn(384).astype(np.float32)
    
    def _embed_image(self, image_path: str) -> np.ndarray:
        """
        Generate image embedding.
        
        Args:
            image_path: Path to image
            
        Returns:
            Embedding vector
        """
        # Placeholder - integrate with vision model
        # For CPU, use a lightweight vision model
        if not self._vision_embedder:
            logger.info("Loading vision embedder for CPU...")
            # Initialize your vision embedding model here
            pass
        
        # Generate embedding
        # img = Image.open(image_path)
        # return self._vision_embedder.encode(img)
        
        # Dummy embedding for now
        return np.random.randn(384).astype(np.float32)
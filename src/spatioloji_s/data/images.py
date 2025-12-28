"""
images.py - Efficient image loading and management
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import cv2
import re
from .config import ImageMetadata


class ImageHandler:
    """
    Efficient image loading and management for spatioloji.
    
    Features:
    - Lazy loading: Load images on-demand
    - LRU cache: Automatic memory management
    - FOV association: Link images to FOVs
    - Flexible formats: Support multiple image formats
    """
    
    def __init__(self, 
                 lazy_load: bool = True,
                 cache_size: Optional[int] = None):
        """
        Initialize ImageHandler.
        
        Parameters
        ----------
        lazy_load : bool
            If True, images loaded only when accessed
        cache_size : int, optional
            Max number of images to keep in memory (LRU cache)
        """
        self.lazy_load = lazy_load
        self.cache_size = cache_size
        
        # Storage
        self._images: Dict[str, np.ndarray] = {}
        self._metadata: Dict[str, ImageMetadata] = {}
        self._access_order: List[str] = []
    
    def load_from_folder(self,
                        folder_path: Union[str, Path],
                        fov_ids: Optional[List[str]] = None,
                        pattern: str = "CellComposite_F{fov_id}.{ext}",
                        extensions: List[str] = ['jpg', 'png', 'tif', 'tiff'],
                        load_now: bool = None) -> Dict[str, ImageMetadata]:
        """
        Load images from folder.
        
        Parameters
        ----------
        folder_path : str or Path
            Path to folder containing images
        fov_ids : list of str, optional
            Specific FOV IDs to load. If None, auto-detect.
        pattern : str
            Filename pattern with {fov_id} and {ext} placeholders
        extensions : list of str
            Image file extensions to try
        load_now : bool, optional
            If True, load all images immediately.
        
        Returns
        -------
        dict
            Metadata for found images
        """
        folder = Path(folder_path)
        
        if not folder.exists():
            raise FileNotFoundError(f"Image folder not found: {folder_path}")
        
        load_immediately = not self.lazy_load if load_now is None else load_now
        
        if fov_ids is None:
            fov_ids = self._detect_fovs_from_folder(folder, pattern, extensions)
            print(f"  → Auto-detected {len(fov_ids)} FOVs from images")
        
        loaded_count = 0
        for fov_id in fov_ids:
            metadata = self._load_single_image(
                folder, fov_id, pattern, extensions, load_immediately
            )
            if metadata is not None:
                self._metadata[fov_id] = metadata
                loaded_count += 1
        
        # Report
        total_size_mb = sum(m.memory_size_mb() for m in self._metadata.values())
        n_loaded = sum(1 for m in self._metadata.values() if m.is_loaded)
        
        print(f"  → Found {loaded_count} images ({total_size_mb:.1f} MB estimated)")
        if load_immediately:
            print(f"  → Loaded {n_loaded} images into memory")
        else:
            print(f"  → Lazy loading enabled")
        
        return self._metadata
    
    def _detect_fovs_from_folder(self,
                                 folder: Path,
                                 pattern: str,
                                 extensions: List[str]) -> List[str]:
        """Auto-detect FOV IDs from image files."""
        # Convert pattern to regex
        pattern_parts = pattern.split('{fov_id}')
        if len(pattern_parts) != 2:
            raise ValueError(f"Pattern must contain {{fov_id}}: {pattern}")
        
        prefix = re.escape(pattern_parts[0])
        suffix = pattern_parts[1].replace('{ext}', f"({'|'.join(extensions)})")
        regex_str = f"{prefix}(.+?){re.escape(suffix.split('.')[0])}\\.({'|'.join(extensions)})"
        regex = re.compile(regex_str)
        
        fov_ids = set()
        for file in folder.iterdir():
            if file.is_file():
                match = regex.match(file.name)
                if match:
                    fov_ids.add(match.group(1))
        
        return sorted(fov_ids)
    
    def _load_single_image(self,
                          folder: Path,
                          fov_id: str,
                          pattern: str,
                          extensions: List[str],
                          load_now: bool) -> Optional[ImageMetadata]:
        """Load or register a single image."""
        # Try variants of FOV ID
        fov_variants = [fov_id, fov_id.zfill(3), fov_id.zfill(4)]
        
        for variant in fov_variants:
            for ext in extensions:
                filename = pattern.format(fov_id=variant, ext=ext)
                filepath = folder / filename
                
                if filepath.exists():
                    if load_now:
                        img = cv2.imread(str(filepath), cv2.IMREAD_COLOR)
                        if img is not None:
                            self._images[fov_id] = img
                            return ImageMetadata(
                                fov_id=fov_id,
                                shape=img.shape,
                                dtype=img.dtype,
                                path=filepath,
                                is_loaded=True
                            )
                    else:
                        # Just read shape (faster)
                        img = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
                        if img is not None:
                            return ImageMetadata(
                                fov_id=fov_id,
                                shape=img.shape,
                                dtype=img.dtype,
                                path=filepath,
                                is_loaded=False
                            )
        
        return None
    
    def get(self, fov_id: str) -> Optional[np.ndarray]:
        """
        Get image for FOV (load if needed).
        
        Parameters
        ----------
        fov_id : str
            FOV identifier
        
        Returns
        -------
        np.ndarray or None
            Image array
        """
        fov_id = str(fov_id)
        
        # Check if already loaded
        if fov_id in self._images:
            self._update_access(fov_id)
            return self._images[fov_id]
        
        # Load if we have metadata
        if fov_id in self._metadata:
            metadata = self._metadata[fov_id]
            if metadata.path is not None:
                img = cv2.imread(str(metadata.path), cv2.IMREAD_COLOR)
                if img is not None:
                    self._images[fov_id] = img
                    metadata.is_loaded = True
                    self._update_access(fov_id)
                    self._check_cache_size()
                    return img
        
        return None
    
    def _update_access(self, fov_id: str) -> None:
        """Update access order for LRU cache."""
        if fov_id in self._access_order:
            self._access_order.remove(fov_id)
        self._access_order.append(fov_id)
    
    def _check_cache_size(self) -> None:
        """Evict least recently used images if cache full."""
        if self.cache_size is None:
            return
        
        while len(self._images) > self.cache_size:
            lru_fov = self._access_order.pop(0)
            if lru_fov in self._images:
                del self._images[lru_fov]
                self._metadata[lru_fov].is_loaded = False
    
    def load_all(self) -> None:
        """Load all images into memory."""
        for fov_id in self._metadata.keys():
            if not self._metadata[fov_id].is_loaded:
                self.get(fov_id)
    
    def unload_all(self) -> None:
        """Unload all images from memory."""
        self._images.clear()
        self._access_order.clear()
        for metadata in self._metadata.values():
            metadata.is_loaded = False
    
    def subset(self, fov_ids: List[str]) -> 'ImageHandler':
        """Create new ImageHandler with subset of FOVs."""
        new_handler = ImageHandler(
            lazy_load=self.lazy_load,
            cache_size=self.cache_size
        )
        
        fov_ids_set = set(str(f) for f in fov_ids)
        
        new_handler._metadata = {
            fid: meta for fid, meta in self._metadata.items()
            if fid in fov_ids_set
        }
        
        new_handler._images = {
            fid: img for fid, img in self._images.items()
            if fid in fov_ids_set
        }
        
        return new_handler
    
    def get_metadata(self, fov_id: str) -> Optional[ImageMetadata]:
        """Get metadata for a FOV."""
        return self._metadata.get(str(fov_id))
    
    @property
    def fov_ids(self) -> List[str]:
        """Get list of all FOV IDs."""
        return list(self._metadata.keys())
    
    @property
    def n_loaded(self) -> int:
        """Get number of loaded images."""
        return len(self._images)
    
    @property
    def n_total(self) -> int:
        """Get total number of images."""
        return len(self._metadata)
    
    def summary(self) -> Dict[str, any]:
        """Get summary of image handler state."""
        total_size = sum(m.memory_size_mb() for m in self._metadata.values())
        loaded_size = sum(
            m.memory_size_mb() for m in self._metadata.values() if m.is_loaded
        )
        
        return {
            'n_total': self.n_total,
            'n_loaded': self.n_loaded,
            'total_size_mb': total_size,
            'loaded_size_mb': loaded_size,
            'lazy_load': self.lazy_load,
            'cache_size': self.cache_size
        }


def load_fov_positions_from_images(image_handler: ImageHandler) -> 'pd.DataFrame':
    """
    Create FOV positions DataFrame from image metadata.
    
    Assumes FOVs are arranged in a grid.
    """
    import pandas as pd
    
    fov_data = []
    fov_ids = sorted(image_handler.fov_ids, 
                    key=lambda x: int(x) if x.isdigit() else x)
    
    n_fovs = len(fov_ids)
    n_cols = int(np.ceil(np.sqrt(n_fovs)))
    
    for idx, fov_id in enumerate(fov_ids):
        metadata = image_handler.get_metadata(fov_id)
        
        if metadata is not None:
            row = idx // n_cols
            col = idx % n_cols
            
            x_pos = col * metadata.width
            y_pos = row * metadata.height
            
            fov_data.append({
                'fov': fov_id,
                'x_global_px': x_pos,
                'y_global_px': y_pos,
                'width': metadata.width,
                'height': metadata.height
            })
    
    df = pd.DataFrame(fov_data)
    return df.set_index('fov')
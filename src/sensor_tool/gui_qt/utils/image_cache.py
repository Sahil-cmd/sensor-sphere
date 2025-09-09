"""
Lazy Image Loading and Caching System
High-performance image loading with memory management and lazy loading for sensor images.
"""

import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtCore import QMutex, QMutexLocker, QObject, QSize, Qt, QThread, Signal
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import QLabel

logger = logging.getLogger(__name__)


class ImageCache:
    """High-performance image cache with memory management and lazy loading."""

    def __init__(self, max_memory_mb: int = 30, max_age_hours: int = 24):
        self.max_memory_mb = max_memory_mb
        self.max_age_hours = max_age_hours
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._mutex = QMutex()
        self._default_pixmap = None

    def _generate_cache_key(self, image_path: str, size: QSize = None) -> str:
        """Generate unique cache key for image and size."""
        size_key = f"{size.width()}x{size.height()}" if size else "original"
        key_string = f"{image_path}_{size_key}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, image_path: str, size: QSize = None) -> Optional[QPixmap]:
        """Get cached image if available and not expired."""
        cache_key = self._generate_cache_key(image_path, size)

        with QMutexLocker(self._mutex):
            if cache_key not in self._cache:
                return None

            entry = self._cache[cache_key]

            # Check if expired
            created_time = entry["created_time"]
            if datetime.now() - created_time > timedelta(hours=self.max_age_hours):
                self._remove_entry(cache_key)
                return None

            # Update access statistics
            entry["access_count"] += 1
            entry["last_access"] = datetime.now()

            size_key = f"{size.width()}x{size.height()}" if size else "original"
            logger.debug(f"Image cache hit: {image_path} ({size_key})")
            return entry["pixmap"]

    def put(self, image_path: str, pixmap: QPixmap, size: QSize = None):
        """Store image in cache with memory management."""
        cache_key = self._generate_cache_key(image_path, size)

        with QMutexLocker(self._mutex):
            # Estimate memory usage (rough approximation)
            memory_mb = (pixmap.width() * pixmap.height() * 4) / (
                1024 * 1024
            )  # Assume 4 bytes per pixel

            # Clean up expired entries first
            self._cleanup_expired()

            # Ensure we don't exceed memory limit
            while self._get_total_memory_mb() + memory_mb > self.max_memory_mb:
                if not self._remove_least_recently_used():
                    break  # No more entries to remove

            self._cache[cache_key] = {
                "pixmap": pixmap,
                "image_path": image_path,
                "memory_mb": memory_mb,
                "created_time": datetime.now(),
                "last_access": datetime.now(),
                "access_count": 1,
            }

            logger.debug(f"Image cached: {image_path} ({memory_mb:.2f}MB)")

    def _remove_entry(self, cache_key: str):
        """Remove entry from cache."""
        if cache_key in self._cache:
            del self._cache[cache_key]

    def _cleanup_expired(self):
        """Remove expired entries."""
        now = datetime.now()
        expired_keys = [
            key
            for key, entry in self._cache.items()
            if now - entry["created_time"] > timedelta(hours=self.max_age_hours)
        ]
        for key in expired_keys:
            self._remove_entry(key)

    def _remove_least_recently_used(self) -> bool:
        """Remove least recently used entry. Returns True if an entry was removed."""
        if not self._cache:
            return False

        lru_key = min(self._cache.keys(), key=lambda k: self._cache[k]["last_access"])
        self._remove_entry(lru_key)
        return True

    def _get_total_memory_mb(self) -> float:
        """Calculate total memory usage."""
        return sum(entry["memory_mb"] for entry in self._cache.values())

    def get_default_pixmap(self, size: QSize = None) -> QPixmap:
        """Get default placeholder pixmap with proper centering."""
        if self._default_pixmap is None or size:
            # Create a properly sized placeholder image
            if size and size.isValid():
                pixmap = QPixmap(size)
            else:
                pixmap = QPixmap(100, 100)

            # Use a subtle background with centered placeholder text/icon
            pixmap.fill(Qt.transparent)

            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

            # Draw a subtle background
            painter.fillRect(pixmap.rect(), Qt.lightGray)

            # Draw a centered placeholder indicator
            painter.setPen(Qt.darkGray)
            rect = pixmap.rect()
            center_x = rect.width() // 2
            center_y = rect.height() // 2

            # Draw a simple centered rectangle as placeholder
            placeholder_width = min(rect.width() * 0.6, 60)
            placeholder_height = min(rect.height() * 0.6, 45)
            placeholder_rect = painter.fontMetrics().boundingRect(
                center_x - placeholder_width // 2,
                center_y - placeholder_height // 2,
                int(placeholder_width),
                int(placeholder_height),
                Qt.AlignCenter,
                "No Image",
            )

            painter.drawRect(placeholder_rect)
            painter.drawText(placeholder_rect, Qt.AlignCenter, "No Image")
            painter.end()

            if size is None:
                self._default_pixmap = pixmap

            return pixmap

        return self._default_pixmap

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with QMutexLocker(self._mutex):
            total_memory = self._get_total_memory_mb()
            return {
                "entries": len(self._cache),
                "memory_mb": total_memory,
                "max_memory_mb": self.max_memory_mb,
                "memory_usage_percent": (
                    (total_memory / self.max_memory_mb) * 100
                    if self.max_memory_mb > 0
                    else 0
                ),
                "oldest_entry": min(
                    (entry["created_time"] for entry in self._cache.values()),
                    default=None,
                ),
                "total_access_count": sum(
                    entry["access_count"] for entry in self._cache.values()
                ),
            }


class ImageLoaderWorker(QThread):
    """Background worker for image loading."""

    # Signals
    image_loaded = Signal(str, QPixmap, QSize)  # (image_path, pixmap, requested_size)
    loading_failed = Signal(str, str)  # (image_path, error_message)

    def __init__(self, cache: ImageCache):
        super().__init__()
        self.cache = cache
        self.pending_requests = []
        self._mutex = QMutex()

    def add_request(self, image_path: str, requested_size: QSize = None):
        """Add image loading request to queue."""
        with QMutexLocker(self._mutex):
            # Check cache first
            cached_pixmap = self.cache.get(image_path, requested_size)
            if cached_pixmap:
                self.image_loaded.emit(
                    image_path, cached_pixmap, requested_size or QSize()
                )
                return

            # Add to loading queue (avoid duplicates)
            request_key = f"{image_path}_{requested_size.width() if requested_size else 0}x{requested_size.height() if requested_size else 0}"
            if not any(req["key"] == request_key for req in self.pending_requests):
                self.pending_requests.append(
                    {
                        "key": request_key,
                        "image_path": image_path,
                        "requested_size": requested_size,
                    }
                )

                # Start processing if not already running
                if not self.isRunning():
                    self.start()

    def run(self):
        """Process image loading requests."""
        while True:
            with QMutexLocker(self._mutex):
                if not self.pending_requests:
                    break
                request = self.pending_requests.pop(0)

            try:
                self._process_request(request)
            except Exception as e:
                logger.error(f"Image loading failed: {e}")
                self.loading_failed.emit(request["image_path"], str(e))

    def _process_request(self, request: Dict[str, Any]):
        """Process individual image loading request."""
        image_path = request["image_path"]
        requested_size = request["requested_size"]

        try:
            # Validate input parameters
            if not image_path:
                raise ValueError("Empty image path provided")

            # Load the image with comprehensive error handling
            image_file = Path(image_path)
            if not image_file.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            if not image_file.is_file():
                raise ValueError(f"Path is not a file: {image_path}")

            # Check file size to prevent loading extremely large files
            try:
                file_size_mb = image_file.stat().st_size / (1024 * 1024)
                if file_size_mb > 50:  # 50MB limit
                    raise ValueError(
                        f"Image file too large ({file_size_mb:.1f}MB): {image_path}"
                    )
            except OSError as e:
                logger.warning(f"Could not check file size for {image_path}: {e}")

            # Load the image
            try:
                pixmap = QPixmap(image_path)
            except Exception as e:
                raise IOError(f"Qt failed to load image {image_path}: {e}") from e

            if pixmap.isNull():
                raise ValueError(f"Invalid or corrupted image file: {image_path}")

            # Validate image dimensions
            if pixmap.width() <= 0 or pixmap.height() <= 0:
                raise ValueError(
                    f"Invalid image dimensions ({pixmap.width()}x{pixmap.height()}): {image_path}"
                )

            # Resize if requested with improved centering approach
            if requested_size and requested_size.isValid():
                try:
                    # Validate requested size
                    if requested_size.width() <= 0 or requested_size.height() <= 0:
                        logger.warning(
                            f"Invalid requested size {requested_size.width()}x{requested_size.height()}, using original size"
                        )
                    else:
                        # Use KeepAspectRatio for better space utilization
                        pixmap = pixmap.scaled(
                            requested_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
                        )

                        # If the scaled image is smaller than requested size in one dimension,
                        # create a centered version on a transparent background
                        if (
                            pixmap.width() < requested_size.width()
                            or pixmap.height() < requested_size.height()
                        ):
                            try:
                                centered_pixmap = QPixmap(requested_size)
                                centered_pixmap.fill(Qt.transparent)

                                painter = QPainter(centered_pixmap)
                                if not painter.isActive():
                                    raise RuntimeError(
                                        "Failed to initialize painter for image centering"
                                    )

                                painter.setRenderHint(QPainter.Antialiasing, True)
                                painter.setRenderHint(
                                    QPainter.SmoothPixmapTransform, True
                                )

                                # Calculate position to center the image
                                x = (requested_size.width() - pixmap.width()) // 2
                                y = (requested_size.height() - pixmap.height()) // 2

                                painter.drawPixmap(x, y, pixmap)
                                painter.end()

                                pixmap = centered_pixmap
                            except Exception as e:
                                logger.warning(
                                    f"Failed to center image, using scaled version: {e}"
                                )
                                # Continue with scaled pixmap

                except Exception as e:
                    logger.warning(
                        f"Failed to resize image {image_path}: {e}, using original size"
                    )
                    # Continue with original pixmap

            # Cache the result
            try:
                self.cache.put(image_path, pixmap, requested_size)
            except Exception as e:
                logger.warning(f"Failed to cache image {image_path}: {e}")
                # Continue without caching

            # Emit success signal
            self.image_loaded.emit(image_path, pixmap, requested_size or QSize())
            logger.debug(f"Successfully loaded image: {image_path}")

        except FileNotFoundError as e:
            logger.warning(f"Image file not found: {e}")
            self.loading_failed.emit(image_path, f"File not found: {image_path}")
        except PermissionError as e:
            logger.error(f"Permission denied loading image {image_path}: {e}")
            self.loading_failed.emit(image_path, f"Permission denied: {image_path}")
        except (IOError, OSError) as e:
            logger.error(f"I/O error loading image {image_path}: {e}")
            self.loading_failed.emit(image_path, f"I/O error loading image: {str(e)}")
        except ValueError as e:
            logger.error(f"Invalid image data {image_path}: {e}")
            self.loading_failed.emit(image_path, f"Invalid image: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error loading image {image_path}: {e}")
            self.loading_failed.emit(image_path, f"Failed to load image: {str(e)}")


class LazyImageLabel(QLabel):
    """QLabel that loads images lazily with caching."""

    _instance_counter = 0

    def __init__(self, parent=None, cache_manager=None):
        super().__init__(parent)
        self.cache_manager = cache_manager or get_image_cache_manager()
        self.image_path = None
        self.requested_size = None
        self.is_loading = False

        # Unique instance identifier to prevent signal confusion
        LazyImageLabel._instance_counter += 1
        self.instance_id = LazyImageLabel._instance_counter

        # Connect to cache manager signals
        self.cache_manager.image_ready.connect(self._on_image_ready)
        self.cache_manager.loading_failed.connect(self._on_loading_failed)

        # Set default placeholder
        self._set_placeholder()

    def set_image_path(self, image_path: str, size: QSize = None):
        """Set image path and start lazy loading."""
        try:
            self.image_path = image_path
            self.requested_size = size

            if not image_path or not image_path.strip():
                self._set_placeholder()
                return

            # Validate cache manager
            if not self.cache_manager:
                logger.error("Cache manager not initialized")
                self._set_error_placeholder()
                return

            # Check if already cached
            try:
                cached_pixmap = self.cache_manager.cache.get(image_path, size)
                if cached_pixmap:
                    self.setPixmap(cached_pixmap)
                    self.is_loading = False
                    return
            except Exception as e:
                logger.warning(f"Error checking cache for {image_path}: {e}")

            # Start lazy loading
            try:
                self.is_loading = True
                self._set_loading_placeholder()
                self.cache_manager.load_image(image_path, size)
            except Exception as e:
                logger.error(f"Failed to start image loading for {image_path}: {e}")
                self.is_loading = False
                self._set_error_placeholder()

        except Exception as e:
            logger.error(f"Unexpected error setting image path {image_path}: {e}")
            self.is_loading = False
            self._set_error_placeholder()

    def _on_image_ready(self, image_path: str, pixmap: QPixmap, size: QSize):
        """Handle image ready signal."""
        # More robust size comparison to fix image disappearing issue
        expected_size = self.requested_size or QSize()
        sizes_match = (size is None and expected_size is None) or (
            size is not None
            and expected_size is not None
            and size.width() == expected_size.width()
            and size.height() == expected_size.height()
        )

        if image_path == self.image_path and sizes_match:
            self.setPixmap(pixmap)
            self.is_loading = False

    def _on_loading_failed(self, image_path: str, error_message: str):
        """Handle loading failed signal."""
        if image_path == self.image_path:
            logger.warning(f"Failed to load image {image_path}: {error_message}")
            self._set_error_placeholder()
            self.is_loading = False

    def _set_placeholder(self):
        """Set default placeholder."""
        placeholder = self.cache_manager.cache.get_default_pixmap(self.requested_size)
        self.setPixmap(placeholder)

    def _set_loading_placeholder(self):
        """Set loading placeholder."""
        # Could be enhanced with a loading animation
        self._set_placeholder()

    def _set_error_placeholder(self):
        """Set error placeholder."""
        # Could be enhanced with an error icon
        self._set_placeholder()


class ImageCacheManager(QObject):
    """High-level manager for lazy image loading and caching."""

    # Signals
    image_ready = Signal(str, QPixmap, QSize)  # (image_path, pixmap, size)
    loading_failed = Signal(str, str)  # (image_path, error_message)

    def __init__(self, max_cache_memory_mb: int = 30):
        super().__init__()

        self.cache = ImageCache(max_memory_mb=max_cache_memory_mb)
        self.worker = ImageLoaderWorker(self.cache)

        # Connect worker signals
        self.worker.image_loaded.connect(self.image_ready.emit)
        self.worker.loading_failed.connect(self.loading_failed.emit)

        logger.info(f"Image cache manager initialized (cache: {max_cache_memory_mb}MB)")

    def load_image(self, image_path: str, size: QSize = None):
        """Request image loading (lazy loading)."""
        try:
            if not image_path:
                raise ValueError("Empty image path provided")
            if not self.worker:
                raise RuntimeError("Image worker not initialized")

            self.worker.add_request(image_path, size)
        except Exception as e:
            logger.error(f"Failed to queue image loading request for {image_path}: {e}")
            # Emit failure signal directly
            self.loading_failed.emit(image_path, f"Failed to queue loading: {str(e)}")

    def get_cached_image(
        self, image_path: str, size: QSize = None
    ) -> Optional[QPixmap]:
        """Get image from cache if available."""
        return self.cache.get(image_path, size)

    def create_lazy_label(self, parent=None) -> LazyImageLabel:
        """Create a new lazy image label connected to this manager."""
        return LazyImageLabel(parent, self)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        return self.cache.get_stats()

    def clear_cache(self):
        """Clear all cached images."""
        # Note: This would need proper implementation in ImageCache
        pass


# Global instance for easy access
_global_image_manager = None


def get_image_cache_manager(max_cache_memory_mb: int = 30) -> ImageCacheManager:
    """Get global image cache manager instance."""
    global _global_image_manager
    if _global_image_manager is None:
        _global_image_manager = ImageCacheManager(max_cache_memory_mb)
    return _global_image_manager

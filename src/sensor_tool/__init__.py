# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# sensor_tool/__init__.py

from . import utils

# Import core modules (no GUI dependencies)
from .data_loader import DataLoader
from .validate_sensors import validate_sensors_main


def qt_gui_main():
    """Lazy import Qt GUI main to avoid circular imports and missing dependencies."""
    try:
        from .gui_qt.main_window import main as _qt_gui_main

        return _qt_gui_main()
    except ImportError as e:
        raise ImportError(f"Qt GUI dependencies not available: {e}") from e


__all__ = [
    "qt_gui_main",
    "DataLoader",
    "validate_sensors_main",
    "utils",
]

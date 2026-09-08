"""Process-local native planning owners.

The package deliberately exposes the geometry context as the Python seam;
ctypes handles and C ABI details remain implementation details of the module.
"""

from .geometry import NativeGeometryContext, NativeGeometryError

__all__ = ["NativeGeometryContext", "NativeGeometryError"]

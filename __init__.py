"""Point Cloud Editor Plugin.

Provides tools for cleaning and editing point clouds by removing isolated points
with few neighbors within a specified voxel distance.
"""

import lichtfeld as lf

from .panels.main_panel import PointCloudEditorPanel

_classes = [PointCloudEditorPanel]


def on_load():
    """Register all plugin classes when the plugin loads."""
    for cls in _classes:
        lf.register_class(cls)


def on_unload():
    """Unregister all plugin classes when the plugin unloads."""
    for cls in reversed(_classes):
        lf.unregister_class(cls)

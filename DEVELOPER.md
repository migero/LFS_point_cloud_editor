# Point Cloud Editor Plugin - Developer Notes

## Plugin Structure

```
point_cloud_editor/
├── pyproject.toml          # Plugin manifest and dependencies
├── __init__.py             # Plugin registration and lifecycle
├── README.md               # User documentation
├── example_usage.py        # Programmatic usage examples
└── panels/
    ├── __init__.py         # Panel module init
    └── main_panel.py       # Main UI panel with filtering logic
```

## Key Components

### 1. Plugin Manifest (`pyproject.toml`)

Defines the plugin metadata and dependencies:
- Plugin name: `point_cloud_editor`
- Version: 0.1.0
- Requires: LichtFeld API v1, scipy (optional but recommended)
- Hot reload enabled for development

### 2. Plugin Lifecycle (`__init__.py`)

Handles plugin loading/unloading:
- `on_load()`: Registers all panel classes with LichtFeld
- `on_unload()`: Unregisters classes when plugin is disabled

### 3. Main Panel (`panels/main_panel.py`)

The UI and core functionality:

**Panel Configuration:**
- ID: `point_cloud_editor.main_panel`
- Space: `MAIN_PANEL_TAB` (appears as a tab like Rendering/Training)
- Order: 150 (positioned after core tabs)
- Poll: Only visible when scene has point cloud nodes

**UI Elements:**
- Voxel size slider (0.001 - 0.5)
- Neighbor threshold slider (1 - 20)
- Point cloud info display
- "Remove Isolated Points" button
- "Save Point Cloud" button
- Operation result display

**Core Methods:**

1. `_remove_isolated_points()`:
   - Gets point cloud data from scene
   - Calls filtering algorithm
   - Applies mask to remove points
   - Updates scene and notifies renderer

2. `_find_isolated_points_kdtree()`:
   - Builds KD-tree from point positions
   - Queries neighbors within voxel_size radius
   - Returns boolean mask (True=keep, False=remove)
   - O(n log n) complexity

3. `_find_isolated_points_bruteforce()`:
   - Fallback when scipy unavailable
   - Nested loop comparison
   - O(n²) complexity (slow for large clouds)

4. `_save_point_cloud()`:
   - Exports to PLY format
   - Saves to ~/Downloads/ with timestamp
   - Includes positions and colors

5. `_write_ply_file()`:
   - Binary PLY writer
   - Format: binary_little_endian
   - Properties: x, y, z, red, green, blue

## API Usage

### Accessing Point Cloud Data

```python
import lichtfeld as lf

# Get scene and find point cloud nodes
scene = lf.get_scene()
for node in scene.get_nodes():
    pc = node.point_cloud()
    if pc is not None:
        # Access data
        positions = pc.means      # Tensor [N, 3]
        colors = pc.colors        # Tensor [N, 3]
        count = pc.size           # int
        
        # Data is on GPU by default
        positions_cpu = positions.cpu().numpy()
```

### Filtering Points

```python
import numpy as np

# Create boolean mask (True = keep, False = remove)
keep_mask = np.array([True, False, True, ...])

# Convert to LichtFeld Tensor
keep_tensor = lf.Tensor.from_numpy(keep_mask)

# Move to same device as point cloud
keep_tensor = keep_tensor.to(pc.means.device)

# Apply filter
removed_count = pc.filter(keep_tensor)

# Notify scene of changes
scene.is_point_cloud_modified = True
scene.notify_changed()
```

### KD-Tree Neighbor Search

```python
from scipy.spatial import cKDTree

# Build tree (CPU operation)
points = pc.means.cpu().numpy()  # Nx3
tree = cKDTree(points)

# Find neighbors within radius for all points
radius = 0.01
neighbors_list = tree.query_ball_tree(tree, radius)

# neighbors_list[i] contains indices of neighbors for point i
for i, neighbors in enumerate(neighbors_list):
    count = len(neighbors) - 1  # Subtract self
    print(f"Point {i} has {count} neighbors")
```

## Algorithm Details

### Isolation Detection

A point is considered "isolated" if:
1. It has few neighbors within the voxel_size distance
2. The neighbor count ≤ neighbor_threshold

**Example:**
- voxel_size = 0.01 (1cm search radius)
- neighbor_threshold = 2
- Point has 2 neighbors → REMOVE (≤ threshold)
- Point has 3 neighbors → KEEP (> threshold)

### Optimization Strategy

1. **Spatial Indexing**: KD-tree enables fast spatial queries
2. **Single Pass**: Each point queried once, no redundant searches
3. **Batch Processing**: All points filtered in one operation
4. **GPU Integration**: Filter applied directly to GPU tensors

### Performance Characteristics

| Point Count | KD-Tree Build | Query Time | Total Time |
|-------------|---------------|------------|------------|
| 10K         | <0.1s         | <0.1s      | ~0.2s      |
| 100K        | ~0.5s         | ~0.5s      | ~1.0s      |
| 1M          | ~5s           | ~5s        | ~10s       |
| 10M         | ~60s          | ~60s       | ~120s      |

*Times are approximate and depend on hardware and point density*

## Extension Ideas

### Additional Filters

1. **Statistical Outlier Removal**:
   - Compute mean distance to K nearest neighbors
   - Remove points beyond N standard deviations

2. **Radius Outlier Removal**:
   - Similar to current, but with different logic
   - Requires minimum absolute neighbor count (not threshold-based)

3. **Density-Based Clustering**:
   - Use DBSCAN to identify clusters
   - Remove small clusters or isolated points

4. **Surface Normal Filtering**:
   - Compute normals from neighbors
   - Remove points with inconsistent normals

### UI Enhancements

1. **Preview Mode**:
   - Highlight points to be removed (red) vs kept (green)
   - Allow user to see result before applying

2. **Progress Bar**:
   - Show processing progress for large point clouds
   - Cancel operation support

3. **Undo/Redo**:
   - Store operation history
   - Allow reverting changes

4. **Batch Processing**:
   - Process multiple point cloud nodes
   - Apply same parameters to all

5. **Parameter Presets**:
   - Save/load common parameter combinations
   - "Light", "Medium", "Aggressive" presets

### Performance Improvements

1. **GPU Acceleration**:
   - Implement neighbor search on GPU using CUDA
   - Use spatial hashing or octrees

2. **Incremental Processing**:
   - Process point cloud in chunks
   - Update UI progressively

3. **Multi-threading**:
   - Parallel neighbor searches
   - Utilize multiple CPU cores

## Testing

### Manual Testing Steps

1. Load a dataset with point cloud
2. Open Point Cloud Editor tab
3. Try different parameter combinations:
   - Small voxel_size (0.001) with low threshold (1)
   - Large voxel_size (0.1) with high threshold (10)
4. Verify points are removed correctly
5. Save and verify PLY file

### Validation

1. **Count Verification**:
   - Before count + removed = after count

2. **Spatial Verification**:
   - Removed points should have few neighbors
   - Kept points should have many neighbors

3. **File Integrity**:
   - Saved PLY should load correctly
   - Position and color data preserved

## Dependencies

### Required
- `lichtfeld` - Core LichtFeld Studio API
- `numpy` - Array operations

### Optional
- `scipy` - KD-tree (highly recommended)

### Installing scipy

```bash
# In plugin directory
cd ~/.lichtfeld/plugins/point_cloud_editor
uv pip install scipy
```

Or using the plugin's virtual environment:
```bash
cd ~/.lichtfeld/plugins/point_cloud_editor
.venv/bin/pip install scipy
```

## Troubleshooting

### Plugin Not Appearing

1. Check plugin is in correct directory: `~/.lichtfeld/plugins/point_cloud_editor/`
2. Verify `pyproject.toml` has `[tool.lichtfeld]` section
3. Restart LichtFeld Studio
4. Check logs for loading errors

### Slow Performance

1. Install scipy for KD-tree optimization
2. Reduce point cloud size before processing
3. Increase voxel_size to reduce neighbor count

### Out of Memory

1. Process on CPU instead of GPU
2. Use smaller point clouds
3. Increase threshold to remove fewer points

## Debugging

Enable debug logging:

```python
import lichtfeld as lf

lf.log.set_level("DEBUG")
```

Check plugin state:

```python
state = lf.plugins.get_state("point_cloud_editor")
error = lf.plugins.get_error("point_cloud_editor")
print(f"State: {state}")
print(f"Error: {error}")
```

## License

Follows LichtFeld Studio license.

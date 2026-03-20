# Point Cloud Editor Plugin - Quick Start

## What This Plugin Does

Removes isolated points from point clouds - points that have very few neighbors within a specified distance. This is useful for cleaning up noisy scans, removing outliers, and improving point cloud quality.

## Installation

1. **The plugin is already created** in `/run/media/migero/base/LichtFeld-Studio/plugins/point_cloud_editor/`

2. **Copy to your plugins directory** (if different):
   ```bash
   mkdir -p ~/.lichtfeld/plugins/
   cp -r /run/media/migero/base/LichtFeld-Studio/plugins/point_cloud_editor ~/.lichtfeld/plugins/
   ```

3. **Install scipy** (optional but highly recommended for performance):
   ```bash
   cd ~/.lichtfeld/plugins/point_cloud_editor
   uv pip install scipy
   ```

4. **Restart LichtFeld Studio** - the plugin will appear as a new tab called "Point Cloud Editor"

## Quick Usage

1. **Load a dataset** with a point cloud (Tools > Load Dataset)
2. **Open the "Point Cloud Editor" tab** in the main panel
3. **Adjust parameters**:
   - **Voxel Size**: Distance to search for neighbors (default: 0.01)
   - **Neighbor Threshold**: Points with ≤ this many neighbors will be removed (default: 2)
4. **Click "Remove Isolated Points"** to clean the point cloud
5. **Click "Save Point Cloud"** to export to `~/Downloads/`

## Files Created

```
plugins/point_cloud_editor/
├── pyproject.toml              # Plugin configuration
├── __init__.py                 # Plugin registration
├── README.md                   # User documentation
├── DEVELOPER.md                # Technical documentation
├── example_usage.py            # Python API examples
└── panels/
    ├── __init__.py
    └── main_panel.py           # Main UI and logic (400+ lines)
```

## Key Features

✅ **Efficient**: Uses KD-tree for fast neighbor search (O(n log n))  
✅ **Non-destructive**: Never overwrites original files  
✅ **Real-time**: See point counts and results immediately  
✅ **Flexible**: Adjustable parameters for different use cases  
✅ **Export**: Save cleaned point clouds as PLY files  

## How It Works

1. **Build KD-Tree** from point positions
2. **Query neighbors** within voxel_size radius for each point
3. **Count neighbors** (excluding the point itself)
4. **Remove points** where neighbor count ≤ threshold
5. **Update scene** and notify renderer

## Example Parameters

| Use Case | Voxel Size | Threshold | Effect |
|----------|-----------|-----------|--------|
| Light cleaning | 0.01 | 1 | Remove only very isolated points |
| Medium cleaning | 0.01 | 2 | Default - good balance |
| Aggressive | 0.05 | 5 | Remove sparse regions |
| Fine detail | 0.001 | 1 | Preserve detail, remove noise |

## API Usage

You can also use the functionality programmatically:

```python
import lichtfeld as lf
from scipy.spatial import cKDTree
import numpy as np

# Get point cloud
scene = lf.get_scene()
node = next(n for n in scene.get_nodes() if n.point_cloud())
pc = node.point_cloud()

# Get positions
points = pc.means.cpu().numpy()

# Find isolated points
tree = cKDTree(points)
neighbors = tree.query_ball_tree(tree, 0.01)  # voxel_size
keep_mask = np.array([len(n) - 1 > 2 for n in neighbors])  # threshold

# Apply filter
keep_tensor = lf.Tensor.from_numpy(keep_mask).to(pc.means.device)
pc.filter(keep_tensor)
scene.notify_changed()
```

## Testing the Plugin

1. **Build LichtFeld Studio** (if not already built):
   ```bash
   cd /run/media/migero/base/LichtFeld-Studio/build
   cmake --build .
   ```

2. **Run the application**:
   ```bash
   ./build/LichtFeld-Studio
   ```

3. **Load a dataset** with point cloud data

4. **Check the plugin appears** in the main panel tabs

5. **Test the functionality** with different parameters

## Troubleshooting

### Plugin doesn't appear
- Check it's in the right directory: `~/.lichtfeld/plugins/point_cloud_editor/`
- Restart LichtFeld Studio
- Check logs in the console

### Slow performance
- Install scipy: `uv pip install scipy`
- The fallback brute-force method is O(n²) and very slow

### No point cloud nodes
- The tab only appears when a point cloud is loaded
- Load a dataset first (Tools > Load Dataset)

## Understanding the Code

The main logic is in `panels/main_panel.py`:

- **`poll()`**: Shows panel only when point clouds exist
- **`draw()`**: Renders the UI with sliders and buttons
- **`_remove_isolated_points()`**: Main processing function
- **`_find_isolated_points_kdtree()`**: Fast neighbor search using KD-tree
- **`_save_point_cloud()`**: Export to PLY format

## Next Steps

1. **Test the plugin** with your data
2. **Adjust parameters** to find what works best
3. **Save results** and compare before/after
4. **Extend functionality** (see DEVELOPER.md for ideas)

## Documentation

- **README.md**: User guide and features
- **DEVELOPER.md**: Technical details and API reference
- **example_usage.py**: Programmatic usage examples

## Support

The plugin uses LichtFeld Studio's official plugin API. Refer to:
- `/docs/plugin-system.md` - Plugin system overview
- `/docs/plugin-dev-workflow.md` - Development workflow
- `/docs/plugins/api-reference.md` - Complete API reference

Enjoy cleaning your point clouds! 🎉

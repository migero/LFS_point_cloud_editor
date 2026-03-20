# Point Cloud Editor Plugin

A LichtFeld Studio plugin for cleaning and editing point clouds by removing isolated points with few neighbors.

## Features

- **Isolated Point Removal**: Automatically detects and removes points that have few neighbors within a specified distance
- **KD-Tree Optimization**: Uses scipy's KD-tree for efficient spatial queries on large point clouds
- **Adjustable Parameters**:
  - **Voxel Size**: Controls the search radius for finding neighbors (0.001 - 0.5)
  - **Neighbor Threshold**: Minimum number of neighbors required to keep a point (1-20)
- **Non-Destructive**: Original point cloud files are never overwritten
- **Save Functionality**: Export cleaned point clouds to PLY format

## Installation

1. Copy this folder to `~/.lichtfeld/plugins/point_cloud_editor/` (or your plugins directory)
2. The plugin will be automatically discovered when you restart LichtFeld Studio
3. Required dependency: `scipy` (for KD-tree implementation)

To install scipy in the plugin environment:
```bash
cd ~/.lichtfeld/plugins/point_cloud_editor
uv pip install scipy
```

## Usage

1. Load a dataset with a point cloud in LichtFeld Studio (Tools > Load Dataset)
2. Open the "Point Cloud Editor" tab in the main panel
3. Adjust the parameters:
   - **Voxel Size**: Smaller values = stricter neighbor requirements
   - **Neighbor Threshold**: Higher values = more aggressive removal
4. Click "Remove Isolated Points" to process the point cloud
5. Click "Save Point Cloud" to export the cleaned result to `~/Downloads/`

## How It Works

### Algorithm

The plugin uses a spatial indexing approach for efficient neighbor searching:

1. **Build KD-Tree**: Constructs a KD-tree from all point positions
2. **Neighbor Query**: For each point, queries all neighbors within the voxel_size radius
3. **Filter Points**: Removes points with neighbor count ≤ threshold
4. **Update Scene**: Applies the filter mask to the point cloud and notifies the renderer

### Performance

- **Fast**: KD-tree enables O(log n) neighbor queries
- **Optimization**: Each point is processed only once
- **Fallback**: Includes a slower brute-force method if scipy is unavailable

### Parameters

- **Voxel Size (Search Radius)**:
  - Range: 0.001 to 0.5
  - Default: 0.01
  - Smaller values detect fine-grained noise
  - Larger values remove broader isolated regions

- **Neighbor Threshold**:
  - Range: 1 to 20
  - Default: 2
  - A point is removed if it has ≤ this many neighbors
  - Higher values = more aggressive filtering

## Example Use Cases

1. **Remove Scanner Noise**: Set voxel_size=0.01, threshold=2 to remove isolated scan artifacts
2. **Clean Sparse Regions**: Set voxel_size=0.05, threshold=5 to remove sparse outliers
3. **Preserve Detail**: Set voxel_size=0.001, threshold=1 for conservative cleaning

## Output

Cleaned point clouds are saved to `~/Downloads/` with the format:
```
{original_name}_cleaned_{timestamp}.ply
```

Example: `points_cleaned_20260320_143022.ply`

## API Reference

The plugin works with LichtFeld's scene API:

```python
# Access point cloud data
scene = lf.get_scene()
for node in scene.get_nodes():
    pc = node.point_cloud()
    if pc:
        means = pc.means  # Nx3 positions
        colors = pc.colors  # Nx3 colors
        
        # Filter points
        keep_mask = lf.Tensor.from_numpy(mask)
        pc.filter(keep_mask)
        
        # Mark as modified
        scene.is_point_cloud_modified = True
        scene.notify_changed()
```

## Technical Details

- **Language**: Python 3.10+
- **Dependencies**: lichtfeld, numpy, scipy (optional)
- **Panel Space**: MAIN_PANEL_TAB (appears alongside Rendering and Training)
- **Hot Reload**: Supported for rapid development

## Known Limitations

- Processes only the first point cloud node in the scene
- Cannot undo operations (use "Save" to preserve intermediate results)
- Large point clouds (>1M points) may take several seconds to process

## Future Enhancements

Possible improvements:
- [ ] Multi-node support (process multiple point clouds)
- [ ] Undo/redo functionality
- [ ] Progressive processing with progress bar
- [ ] Additional filters (statistical outlier removal, radius outlier removal)
- [ ] Visualization of removed points before applying
- [ ] GPU-accelerated neighbor search

## License

This plugin follows the same license as LichtFeld Studio.

## Contributing

To modify this plugin:

1. Edit files in this directory
2. Enable hot reload in `pyproject.toml`
3. Changes will be reflected immediately in LichtFeld Studio

## Support

For issues or questions, refer to the LichtFeld Studio documentation or create an issue in the main repository.

# Point Cloud Editor Plugin

A LichtFeld Studio plugin for cleaning, simplifying, and editing point clouds with undo support.

## Features

- **Isolated Point Removal**: Automatically detects and removes points that have few neighbors within a specified distance
- **Point Cloud Simplification**: Merge nearby points by averaging positions and colors to reduce point count while preserving structure
- **Undo/Redo Support**: Full undo/redo functionality for all operations
- **KD-Tree Optimization**: Uses scipy's KD-tree for efficient spatial queries on large point clouds
- **Adjustable Parameters**:
  - **Removal**: Search radius (0.001 - 0.5) and neighbor threshold (1-20)
  - **Simplification**: Merge distance (0.001 - 1.0) and points per cluster (2-50)
- **Non-Destructive**: Original point cloud files are never overwritten
- **Save Functionality**: Export modified point clouds to PLY format
- **Auto-Dependency Installation**: scipy is automatically installed when the plugin loads

## Installation

1. Copy this folder to `~/.lichtfeld/plugins/point_cloud_editor/` (or your plugins directory)
2. Restart LichtFeld Studio - the plugin will be automatically discovered
3. **Dependencies are installed automatically** - scipy will be installed via the plugin's virtual environment on first load

No manual installation needed! The plugin's `pyproject.toml` declares scipy as a dependency and LichtFeld installs it automatically.

## Usage

### Isolated Point Removal

1. Load a dataset with a point cloud in LichtFeld Studio (Tools > Load Dataset)
2. Open the "Point Cloud Editor" tab in the main panel
3. Expand the "Isolated Point Removal" section
4. Adjust the parameters:
   - **Search Radius**: Distance to search for neighbors (smaller = stricter)
   - **Min Neighbors**: Points with ≤ this many neighbors will be removed
5. Click "Remove Isolated Points" to process the point cloud
6. Use "Undo" button to revert if needed

### Point Cloud Simplification

1. Open the "Point Cloud Simplification" section
2. Adjust the parameters:
   - **Merge Distance**: Maximum distance for points to be merged together
   - **Points per Cluster**: Target number of points to average into one
3. Click "Simplify Point Cloud" to merge nearby points
4. The algorithm averages positions and colors of clustered points
5. Use "Undo" to restore the original if needed

### Saving Results

- Click "Save Point Cloud" to export the current state to `~/Downloads/`
- Files are saved with timestamp: `{name}_cleaned_{timestamp}.ply`
- Original files are never overwritten

### Undo/Redo

- **Undo** button appears next to Save when operations can be undone
- Full undo/redo support for both removal and simplification
- Undo stack persists across plugin usage
- Use Ctrl+Z (or Edit > Undo) for standard undo in LichtFeld Studio

## How It Works

### Isolated Point Removal Algorithm

The plugin uses a spatial indexing approach for efficient neighbor searching:

1. **Build KD-Tree**: Constructs a KD-tree from all point positions
2. **Neighbor Query**: For each point, queries all neighbors within the search radius
3. **Filter Points**: Removes points with neighbor count ≤ threshold
4. **Update Scene**: Applies the filter mask to the point cloud and notifies the renderer
5. **Push Undo**: Saves the original state for undo/redo support

### Point Cloud Simplification Algorithm

Uses a greedy clustering approach to merge nearby points:

1. **Build KD-Tree**: Constructs a spatial index for fast neighbor queries
2. **Greedy Clustering**: 
   - For each unprocessed point, find neighbors within merge distance
   - Select up to N neighbors (points per cluster)
   - Average their positions and colors
   - Mark all clustered points as processed to avoid reuse
3. **Replace Data**: Updates the point cloud with simplified data
4. **Push Undo**: Saves the original state for restoration

This approach ensures:
- No point is merged into multiple clusters
- Even distribution of simplification across the cloud
- O(n log n) time complexity with KD-tree
- Preservation of overall structure and appearance

### Performance

- **Fast**: KD-tree enables O(log n) neighbor queries
- **Optimization**: Each point is processed only once
- **Fallback**: Includes a slower brute-force method if scipy is unavailable

### Parameters

#### Isolated Point Removal

- **Search Radius**:
  - Range: 0.001 to 0.5
  - Default: 0.01
  - Smaller values detect fine-grained noise
  - Larger values remove broader isolated regions

- **Min Neighbors**:
  - Range: 1 to 20
  - Default: 2
  - A point is removed if it has ≤ this many neighbors
  - Higher values = more aggressive filtering

#### Point Cloud Simplification

- **Merge Distance**:
  - Range: 0.001 to 1.0
  - Default: 0.05
  - Maximum distance for points to be grouped together
  - Larger values = more aggressive simplification

- **Points per Cluster**:
  - Range: 2 to 50
  - Default: 5
  - Target number of points to average into one
  - Higher values = greater reduction in point count

## Example Use Cases

### Isolated Point Removal

1. **Remove Scanner Noise**: Search radius=0.01, Min neighbors=2 to remove isolated scan artifacts
2. **Clean Sparse Regions**: Search radius=0.05, Min neighbors=5 to remove sparse outliers
3. **Preserve Detail**: Search radius=0.001, Min neighbors=1 for conservative cleaning

### Point Cloud Simplification

1. **Gentle Simplification**: Merge distance=0.02, Points per cluster=3 for ~30% reduction
2. **Aggressive Reduction**: Merge distance=0.1, Points per cluster=10 for ~90% reduction
3. **Uniform Downsampling**: Merge distance=0.05, Points per cluster=5 for balanced simplification

### Combined Workflow

1. **Clean then Simplify**: First remove isolated points, then simplify for optimal results
2. **Iterative Simplification**: Apply simplification multiple times with undo between steps
3. **Save Checkpoints**: Use "Save Point Cloud" to preserve intermediate results

## Output

Cleaned point clouds are saved to `~/Downloads/` with the format:
```
{original_name}_cleaned_{timestamp}.ply
```

Example: `points_cleaned_20260320_143022.ply`

## API Reference

The plugin works with LichtFeld's scene and undo APIs:

### Point Cloud Access

```python
import lichtfeld as lf

# Access point cloud data
scene = lf.get_scene()
for node in scene.get_nodes():
    pc = node.point_cloud()
    if pc:
        means = pc.means  # Nx3 positions
        colors = pc.colors  # Nx3 colors
        
        # Filter points (removal)
        keep_mask = lf.Tensor.from_numpy(mask)
        pc.filter(keep_mask)
        
        # Replace data (simplification)
        new_means = lf.Tensor.from_numpy(new_positions).cuda()
        new_colors = lf.Tensor.from_numpy(new_colors).cuda()
        pc.set_data(new_means, new_colors)
        
        # Mark as modified
        scene.is_point_cloud_modified = True
        scene.notify_changed()
```

### Undo/Redo Support

```python
# Push an undo step
def undo_operation():
    # Restore original state
    pc.set_data(original_means, original_colors)
    scene.notify_changed()

def redo_operation():
    # Reapply the operation
    pc.set_data(modified_means, modified_colors)
    scene.notify_changed()

lf.undo.push(
    "Operation Name",
    undo_operation,
    redo_operation
)

# Check undo availability
if lf.undo.can_undo():
    lf.undo.undo()

if lf.undo.can_redo():
    lf.undo.redo()
```

## Technical Details

- **Language**: Python 3.10+
- **Dependencies**: lichtfeld, numpy, scipy (optional)
- **Panel Space**: MAIN_PANEL_TAB (appears alongside Rendering and Training)
- **Hot Reload**: Supported for rapid development

## Known Limitations

- Processes only the first point cloud node in the scene
- Large point clouds (>1M points) may take several seconds to process
- Undo steps consume memory (original data is kept in RAM/VRAM)
- Simplification is non-reversible through re-simplification (use undo instead)

## Future Enhancements

Possible improvements:
- [ ] Multi-node support (process multiple point clouds)
- [ ] Progressive processing with progress bar and cancellation
- [ ] Additional filters (statistical outlier removal, radius outlier removal)
- [ ] Visualization of points to be removed/merged before applying
- [ ] GPU-accelerated neighbor search
- [ ] Voxel grid downsampling as alternative to clustering
- [ ] Preview mode with color-coded visualization
- [ ] Batch processing with parameter presets

## License

This plugin follows the same license as LichtFeld Studio.

## Contributing

To modify this plugin:

1. Edit files in this directory
2. Enable hot reload in `pyproject.toml`
3. Changes will be reflected immediately in LichtFeld Studio

## Support

For issues or questions, refer to the LichtFeld Studio documentation or create an issue in the main repository.

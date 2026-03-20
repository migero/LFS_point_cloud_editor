"""Point Cloud Editor Panel.

Provides UI and functionality for removing isolated points from point clouds.
"""

from pathlib import Path
import numpy as np
from typing import Optional
import time

import lichtfeld as lf


class PointCloudEditorPanel(lf.ui.Panel):
    """Panel for editing point clouds - remove isolated points with few neighbors."""
    
    id = "point_cloud_editor.main_panel"
    label = "Point Cloud Editor"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 150  # Place after Rendering and Training tabs

    def __init__(self):
        """Initialize the panel with default parameters."""
        self._voxel_size = 0.01  # Default voxel size for neighbor search
        self._neighbor_threshold = 2  # Minimum number of neighbors to keep a point
        self._processing = False
        self._last_result = None
        self._backup_data = None  # Store backup before modification
        
    @classmethod
    def poll(cls, context) -> bool:
        """Show panel only when a scene with point cloud is loaded."""
        if not lf.has_scene():
            return False
        scene = lf.get_scene()
        if not scene.has_nodes():
            return False
        # Check if there's at least one point cloud node
        for node in scene.get_nodes():
            pc = node.point_cloud()
            if pc is not None:
                return True
        return False

    def draw(self, ui):
        """Draw the immediate-mode UI."""
        try:
            ui.heading("Point Cloud Isolation Removal")
            ui.text_disabled("Remove points with few neighbors within a voxel distance")
            
            ui.separator()
            
            # Voxel size slider for neighbor search radius
            ui.text("Voxel Size (Search Radius)")
            changed, self._voxel_size = ui.slider_float(
                "##voxel_size", 
                self._voxel_size, 
                0.001, 
                0.5,
                format="%.4f"
            )
            ui.text_disabled("Distance to search for neighboring points")
            
            ui.spacing()
            
            # Neighbor threshold input
            ui.text("Neighbor Threshold")
            changed, self._neighbor_threshold = ui.slider_int(
                "##neighbor_threshold",
                self._neighbor_threshold,
                1,
                20
            )
            ui.text_disabled(f"Remove points with <= {self._neighbor_threshold} neighbors")
            
            ui.separator()
            
            # Show current scene info - with error handling
            try:
                scene = lf.get_scene()
                if scene and scene.has_nodes():
                    point_cloud_nodes = []
                    for node in scene.get_nodes():
                        try:
                            pc = node.point_cloud()
                            if pc is not None:
                                point_cloud_nodes.append((node.name, pc.size))
                        except Exception as e:
                            lf.log.warn(f"Error accessing node point cloud: {e}")
                            continue
                    
                    if point_cloud_nodes:
                        ui.text(f"Point Cloud Nodes: {len(point_cloud_nodes)}")
                        for name, count in point_cloud_nodes:
                            ui.text_disabled(f"  • {name}: {count:,} points")
                    else:
                        ui.text("No point cloud nodes found")
                else:
                    ui.text("Scene not loaded")
            except Exception as e:
                ui.text(f"Error accessing scene: {str(e)}")
                lf.log.error(f"Error in draw() getting scene info: {e}")
            
            ui.separator()
            
            # Action buttons
            if self._processing:
                ui.text("Processing...")
            else:
                if ui.button_styled("Remove Isolated Points", "primary", (250, 0)):
                    self._remove_isolated_points()
                
                ui.spacing()
                
                # Save button
                if ui.button_styled("Save Point Cloud", "secondary", (250, 0)):
                    self._save_point_cloud()
                
                # Show result from last operation
                if self._last_result:
                    ui.spacing()
                    ui.separator()
                    ui.text("Last Operation:")
                    for line in self._last_result.split('\n'):
                        ui.text_disabled(line)
        
        except Exception as e:
            ui.text(f"UI Error: {str(e)}")
            lf.log.error(f"Error in draw(): {e}")
            import traceback
            traceback.print_exc()

    def _remove_isolated_points(self):
        """Remove isolated points from the point cloud."""
        self._processing = True
        self._last_result = None
        
        try:
            scene = lf.get_scene()
            
            # Find the first point cloud node
            target_node = None
            target_pc = None
            for node in scene.get_nodes():
                pc = node.point_cloud()
                if pc is not None:
                    target_node = node
                    target_pc = pc
                    break
            
            if target_pc is None:
                self._last_result = "Error: No point cloud found in scene"
                lf.log.error(self._last_result)
                return
            
            original_count = target_pc.size
            lf.log.info(f"Processing point cloud '{target_node.name}' with {original_count:,} points")
            lf.log.info(f"Parameters: voxel_size={self._voxel_size}, neighbor_threshold={self._neighbor_threshold}")
            
            # Get point positions as numpy array
            means_tensor = target_pc.means
            if means_tensor is None:
                self._last_result = "Error: Point cloud has no position data"
                lf.log.error(self._last_result)
                return
            
            # Convert to numpy (CPU)
            means_np = means_tensor.cpu().numpy()
            
            # Backup original data before modification
            if target_pc.colors is not None:
                colors_np = target_pc.colors.cpu().numpy()
                self._backup_data = (means_np.copy(), colors_np.copy())
            else:
                self._backup_data = (means_np.copy(), None)
            
            start_time = time.time()
            
            # Use KD-tree for efficient neighbor search
            keep_mask = self._find_isolated_points_kdtree(means_np)
            
            elapsed = time.time() - start_time
            
            # Count points to remove
            remove_count = np.sum(~keep_mask)
            keep_count = np.sum(keep_mask)
            
            lf.log.info(f"Found {remove_count:,} isolated points to remove")
            lf.log.info(f"Keeping {keep_count:,} points")
            lf.log.info(f"Processing time: {elapsed:.2f}s")
            
            if remove_count == 0:
                self._last_result = f"No isolated points found\n{original_count:,} points remain"
                lf.log.info("No points removed")
            else:
                # Apply filter to point cloud
                # Convert numpy bool array to lichtfeld Tensor
                keep_tensor = lf.Tensor.from_numpy(keep_mask).to(means_tensor.device)
                removed = target_pc.filter(keep_tensor)
                
                # Mark point cloud as modified so it can be saved
                scene.is_point_cloud_modified = True
                
                # Notify the scene that data has changed
                scene.notify_changed()
                
                self._last_result = (
                    f"Removed {remove_count:,} isolated points\n"
                    f"Remaining: {keep_count:,} points\n"
                    f"Time: {elapsed:.2f}s"
                )
                lf.log.info(f"Successfully filtered point cloud: {remove_count:,} points removed")
        
        except Exception as e:
            self._last_result = f"Error: {str(e)}"
            lf.log.error(f"Error removing isolated points: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self._processing = False

    def _find_isolated_points_kdtree(self, points: np.ndarray) -> np.ndarray:
        """
        Find isolated points using KD-tree for efficient neighbor search.
        
        Args:
            points: Nx3 array of point positions
            
        Returns:
            Boolean mask where True = keep point, False = remove point
        """
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            lf.log.error("scipy not available, using slower brute force method")
            return self._find_isolated_points_bruteforce(points)
        
        # Build KD-tree for efficient spatial queries
        lf.log.info("Building KD-tree...")
        tree = cKDTree(points)
        
        # Query for neighbors within voxel_size radius
        # For each point, find all neighbors (including itself)
        lf.log.info(f"Querying neighbors within radius {self._voxel_size}...")
        neighbors_list = tree.query_ball_tree(tree, self._voxel_size)
        
        # Create keep mask based on neighbor count
        # Subtract 1 from neighbor count because each point is its own neighbor
        keep_mask = np.array([
            (len(neighbors) - 1) > self._neighbor_threshold 
            for neighbors in neighbors_list
        ], dtype=bool)
        
        return keep_mask
    
    def _find_isolated_points_bruteforce(self, points: np.ndarray) -> np.ndarray:
        """
        Fallback brute force method for finding isolated points.
        Used when scipy is not available.
        
        Args:
            points: Nx3 array of point positions
            
        Returns:
            Boolean mask where True = keep point, False = remove point
        """
        n = len(points)
        keep_mask = np.zeros(n, dtype=bool)
        voxel_size_sq = self._voxel_size ** 2
        
        lf.log.info("Using brute force neighbor search (this may be slow)...")
        
        # Process in batches to show progress
        batch_size = 1000
        for i in range(0, n, batch_size):
            batch_end = min(i + batch_size, n)
            
            for j in range(i, batch_end):
                point = points[j]
                # Count neighbors within voxel_size
                distances_sq = np.sum((points - point) ** 2, axis=1)
                # Exclude the point itself (distance = 0)
                neighbor_count = np.sum((distances_sq > 0) & (distances_sq <= voxel_size_sq))
                keep_mask[j] = neighbor_count > self._neighbor_threshold
            
            if (i + batch_size) % 10000 == 0:
                lf.log.info(f"Processed {i + batch_size:,} / {n:,} points")
        
        return keep_mask

    def _save_point_cloud(self):
        """Save the current point cloud to a PLY file."""
        try:
            scene = lf.get_scene()
            
            # Find the first point cloud node
            target_node = None
            target_pc = None
            for node in scene.get_nodes():
                pc = node.point_cloud()
                if pc is not None:
                    target_node = node
                    target_pc = pc
                    break
            
            if target_pc is None:
                self._last_result = "Error: No point cloud found to save"
                lf.log.error(self._last_result)
                return
            
            # Generate output filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_name = f"{target_node.name}_cleaned_{timestamp}.ply"
            output_path = Path.home() / "Downloads" / output_name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Get point data
            means_tensor = target_pc.means
            colors_tensor = target_pc.colors
            
            if means_tensor is None:
                self._last_result = "Error: Point cloud has no position data"
                lf.log.error(self._last_result)
                return
            
            # Convert to numpy
            means_np = means_tensor.cpu().numpy()
            
            if colors_tensor is not None:
                colors_np = colors_tensor.cpu().numpy()
            else:
                # Default white color if no colors
                colors_np = np.ones((len(means_np), 3), dtype=np.uint8) * 255
            
            # Save as PLY using the built-in io functionality
            # Note: We'll write a simple PLY file directly
            self._write_ply_file(output_path, means_np, colors_np)
            
            self._last_result = f"Saved to:\n{output_path}\n{len(means_np):,} points"
            lf.log.info(f"Point cloud saved to {output_path}")
        
        except Exception as e:
            self._last_result = f"Error saving: {str(e)}"
            lf.log.error(f"Error saving point cloud: {e}")
            import traceback
            traceback.print_exc()
    
    def _write_ply_file(self, filepath: Path, points: np.ndarray, colors: np.ndarray):
        """
        Write a simple PLY file with points and colors.
        
        Args:
            filepath: Output file path
            points: Nx3 array of positions
            colors: Nx3 array of RGB colors (0-255)
        """
        n_points = len(points)
        
        # Ensure colors are uint8
        if colors.dtype != np.uint8:
            colors = (colors * 255).astype(np.uint8)
        
        with open(filepath, 'wb') as f:
            # Write PLY header
            header = f"""ply
format binary_little_endian 1.0
element vertex {n_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
            f.write(header.encode('ascii'))
            
            # Write vertex data
            # Interleave position (float32) and color (uint8) data
            for i in range(n_points):
                # Position (3 floats)
                f.write(points[i].astype(np.float32).tobytes())
                # Color (3 uint8)
                f.write(colors[i].astype(np.uint8).tobytes())
        
        lf.log.info(f"Wrote {n_points:,} points to {filepath}")

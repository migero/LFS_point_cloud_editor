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
        # Removal parameters
        self._voxel_size = 0.01  # Default voxel size for neighbor search
        self._neighbor_threshold = 2  # Minimum number of neighbors to keep a point
        
        # Simplification parameters
        self._simplify_voxel_size = 0.05  # Voxel size for simplification
        self._points_per_cluster = 5  # Number of points to merge into one
        
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
        ui.heading("Point Cloud Editor")
        
        # ============ ISOLATED POINT REMOVAL ============
        if ui.collapsing_header("Isolated Point Removal", default_open=True):
            ui.text_disabled("Remove points with few neighbors")
            
            ui.spacing()
            
            # Voxel size slider for neighbor search radius
            changed, self._voxel_size = ui.slider_float(
                "Search Radius", 
                self._voxel_size, 
                0.001, 
                0.5
            )
            
            # Neighbor threshold input
            changed, self._neighbor_threshold = ui.slider_int(
                "Min Neighbors",
                self._neighbor_threshold,
                1,
                20
            )
            
            ui.spacing()
            
            if ui.button("Remove Isolated Points", (-1, 0)):
                self._remove_isolated_points()
        
        ui.spacing()
        
        # ============ POINT CLOUD SIMPLIFICATION ============
        if ui.collapsing_header("Point Cloud Simplification", default_open=True):
            ui.text_disabled("Merge nearby points by averaging")
            
            ui.spacing()
            
            # Simplification voxel size
            changed, self._simplify_voxel_size = ui.slider_float(
                "Merge Distance",
                self._simplify_voxel_size,
                0.001,
                1.0
            )
            
            # Points per cluster
            changed, self._points_per_cluster = ui.slider_int(
                "Points per Cluster",
                self._points_per_cluster,
                2,
                50
            )
            
            ui.spacing()
            
            if ui.button("Simplify Point Cloud", (-1, 0)):
                self._simplify_point_cloud()
        
        ui.separator()
        
        # ============ SAVE & UNDO ============
        if ui.button("Save Point Cloud", (-1, 0)):
            self._save_point_cloud()
        
        ui.same_line()
        
        if lf.undo.can_undo():
            if ui.button("Undo", (100, 0)):
                lf.undo.undo()
        else:
            ui.begin_disabled(True)
            ui.button("Undo", (100, 0))
            ui.end_disabled()
        
        # Show result from last operation
        if self._last_result:
            ui.separator()
            ui.label("Last Operation:")
            for line in self._last_result.split('\n'):
                ui.text_disabled(line)

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
                # Backup current state for undo
                original_means = means_np.copy()
                original_colors = colors_np.copy() if target_pc.colors is not None else None
                
                # Apply filter to point cloud
                # Convert numpy bool array to lichtfeld Tensor
                # The filter() method handles device conversion internally
                keep_tensor = lf.Tensor.from_numpy(keep_mask)
                removed = target_pc.filter(keep_tensor)
                
                # Mark point cloud as modified so it can be saved
                scene.is_point_cloud_modified = True
                
                # Notify the scene that data has changed
                scene.notify_changed()
                
                # Push undo step
                def undo_removal():
                    """Restore the original point cloud."""
                    means_restore = lf.Tensor.from_numpy(original_means).cuda()
                    if original_colors is not None:
                        colors_restore = lf.Tensor.from_numpy(original_colors).cuda()
                    else:
                        colors_restore = lf.Tensor.from_numpy(np.ones((len(original_means), 3), dtype=np.uint8) * 255).cuda()
                    target_pc.set_data(means_restore, colors_restore)
                    scene.notify_changed()
                
                def redo_removal():
                    """Reapply the filter."""
                    keep_tensor_redo = lf.Tensor.from_numpy(keep_mask)
                    target_pc.filter(keep_tensor_redo)
                    scene.notify_changed()
                
                lf.undo.push(
                    f"Remove Isolated Points ({remove_count:,})",
                    undo_removal,
                    redo_removal
                )
                
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

    def _simplify_point_cloud(self):
        """Simplify point cloud by merging nearby points through averaging."""
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
            lf.log.info(f"Simplifying point cloud '{target_node.name}' with {original_count:,} points")
            lf.log.info(f"Parameters: merge_distance={self._simplify_voxel_size}, points_per_cluster={self._points_per_cluster}")
            
            # Get point data
            means_tensor = target_pc.means
            colors_tensor = target_pc.colors
            
            if means_tensor is None:
                self._last_result = "Error: Point cloud has no position data"
                lf.log.error(self._last_result)
                return
            
            # Convert to numpy
            means_np = means_tensor.cpu().numpy()
            colors_np = colors_tensor.cpu().numpy() if colors_tensor is not None else None
            
            # Backup original data for undo
            original_means = means_np.copy()
            original_colors = colors_np.copy() if colors_np is not None else None
            
            start_time = time.time()
            
            # Perform simplification
            new_means, new_colors = self._cluster_and_average(
                means_np, 
                colors_np,
                self._simplify_voxel_size,
                self._points_per_cluster
            )
            
            elapsed = time.time() - start_time
            
            new_count = len(new_means)
            removed_count = original_count - new_count
            reduction_pct = (removed_count / original_count) * 100 if original_count > 0 else 0
            
            lf.log.info(f"Simplified to {new_count:,} points ({removed_count:,} merged, {reduction_pct:.1f}% reduction)")
            lf.log.info(f"Processing time: {elapsed:.2f}s")
            
            if removed_count == 0:
                self._last_result = f"No simplification applied\n{original_count:,} points remain"
                lf.log.info("No points merged")
            else:
                # Convert to tensors
                means_new_tensor = lf.Tensor.from_numpy(new_means.astype(np.float32)).cuda()
                
                if new_colors is not None:
                    if new_colors.dtype != np.uint8:
                        new_colors = (new_colors * 255).astype(np.uint8)
                    colors_new_tensor = lf.Tensor.from_numpy(new_colors).cuda()
                else:
                    colors_new_tensor = lf.Tensor.from_numpy(np.ones((new_count, 3), dtype=np.uint8) * 255).cuda()
                
                # Replace point cloud data
                target_pc.set_data(means_new_tensor, colors_new_tensor)
                
                # Mark as modified
                scene.is_point_cloud_modified = True
                scene.notify_changed()
                
                # Push undo step
                def undo_simplify():
                    """Restore the original point cloud."""
                    means_restore = lf.Tensor.from_numpy(original_means).cuda()
                    if original_colors is not None:
                        colors_restore = lf.Tensor.from_numpy(original_colors).cuda()
                    else:
                        colors_restore = lf.Tensor.from_numpy(np.ones((len(original_means), 3), dtype=np.uint8) * 255).cuda()
                    target_pc.set_data(means_restore, colors_restore)
                    scene.notify_changed()
                
                def redo_simplify():
                    """Reapply the simplification."""
                    means_redo = lf.Tensor.from_numpy(new_means.astype(np.float32)).cuda()
                    colors_redo = lf.Tensor.from_numpy(new_colors if new_colors.dtype == np.uint8 else (new_colors * 255).astype(np.uint8)).cuda()
                    target_pc.set_data(means_redo, colors_redo)
                    scene.notify_changed()
                
                lf.undo.push(
                    f"Simplify Point Cloud ({removed_count:,} merged)",
                    undo_simplify,
                    redo_simplify
                )
                
                self._last_result = (
                    f"Simplified point cloud\n"
                    f"Original: {original_count:,} points\n"
                    f"Result: {new_count:,} points\n"
                    f"Reduction: {reduction_pct:.1f}%\n"
                    f"Time: {elapsed:.2f}s"
                )
                lf.log.info(f"Successfully simplified point cloud")
        
        except Exception as e:
            self._last_result = f"Error: {str(e)}"
            lf.log.error(f"Error simplifying point cloud: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self._processing = False

    def _cluster_and_average(self, points: np.ndarray, colors: Optional[np.ndarray], 
                            merge_distance: float, points_per_cluster: int) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Cluster nearby points and average their positions and colors.
        
        Uses a greedy clustering approach:
        1. For each unprocessed point, find nearby neighbors
        2. Average up to N neighbors together
        3. Mark them as processed to avoid reuse
        
        Args:
            points: Nx3 array of point positions
            colors: Nx3 array of colors (or None)
            merge_distance: Maximum distance for points to be merged
            points_per_cluster: Target number of points to merge into one
            
        Returns:
            (new_points, new_colors) - Simplified arrays
        """
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            lf.log.error("scipy required for simplification")
            return points, colors
        
        n = len(points)
        processed = np.zeros(n, dtype=bool)
        
        new_points_list = []
        new_colors_list = [] if colors is not None else None
        
        lf.log.info(f"Building KD-tree for simplification...")
        tree = cKDTree(points)
        
        lf.log.info(f"Clustering points (target: {points_per_cluster} per cluster)...")
        
        processed_count = 0
        cluster_count = 0
        
        for i in range(n):
            if processed[i]:
                continue
            
            # Find neighbors within merge distance
            neighbor_indices = tree.query_ball_point(points[i], merge_distance)
            
            # Filter out already processed points
            available_neighbors = [idx for idx in neighbor_indices if not processed[idx]]
            
            if len(available_neighbors) == 0:
                continue
            
            # Take up to points_per_cluster neighbors
            cluster_indices = available_neighbors[:points_per_cluster]
            
            # Average positions
            cluster_points = points[cluster_indices]
            avg_point = cluster_points.mean(axis=0)
            new_points_list.append(avg_point)
            
            # Average colors if available
            if colors is not None:
                cluster_colors = colors[cluster_indices]
                avg_color = cluster_colors.mean(axis=0)
                new_colors_list.append(avg_color)
            
            # Mark all points in cluster as processed
            processed[cluster_indices] = True
            processed_count += len(cluster_indices)
            cluster_count += 1
            
            # Log progress every 10000 clusters
            if cluster_count % 10000 == 0:
                lf.log.info(f"  Processed {processed_count:,} / {n:,} points, {cluster_count:,} clusters created")
        
        new_points = np.array(new_points_list, dtype=np.float32)
        new_colors = np.array(new_colors_list, dtype=colors.dtype) if new_colors_list else None
        
        lf.log.info(f"Created {cluster_count:,} clusters from {n:,} points")
        
        return new_points, new_colors

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

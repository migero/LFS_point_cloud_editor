"""
Example script demonstrating Point Cloud Editor usage.

This script can be run from LichtFeld Studio's Python console or editor
to programmatically use the point cloud cleaning functionality.
"""

import lichtfeld as lf
import numpy as np
from scipy.spatial import cKDTree


def clean_point_cloud(voxel_size=0.01, neighbor_threshold=2):
    """
    Clean the current point cloud by removing isolated points.
    
    Args:
        voxel_size: Search radius for neighbors
        neighbor_threshold: Minimum neighbors required to keep a point
    
    Returns:
        Tuple of (removed_count, remaining_count) or None if no point cloud
    """
    scene = lf.get_scene()
    if not scene or not scene.has_nodes():
        lf.log.error("No scene loaded")
        return None
    
    # Find first point cloud node
    target_node = None
    target_pc = None
    for node in scene.get_nodes():
        pc = node.point_cloud()
        if pc is not None:
            target_node = node
            target_pc = pc
            break
    
    if target_pc is None:
        lf.log.error("No point cloud found in scene")
        return None
    
    original_count = target_pc.size
    lf.log.info(f"Processing '{target_node.name}' with {original_count:,} points")
    
    # Get point positions
    means_np = target_pc.means.cpu().numpy()
    
    # Build KD-tree and find neighbors
    tree = cKDTree(means_np)
    neighbors_list = tree.query_ball_tree(tree, voxel_size)
    
    # Create keep mask (subtract 1 because each point is its own neighbor)
    keep_mask = np.array([
        (len(neighbors) - 1) > neighbor_threshold 
        for neighbors in neighbors_list
    ], dtype=bool)
    
    remove_count = np.sum(~keep_mask)
    keep_count = np.sum(keep_mask)
    
    lf.log.info(f"Removing {remove_count:,} isolated points")
    lf.log.info(f"Keeping {keep_count:,} points")
    
    # Apply filter
    keep_tensor = lf.Tensor.from_numpy(keep_mask).to(target_pc.means.device)
    target_pc.filter(keep_tensor)
    
    # Mark as modified
    scene.is_point_cloud_modified = True
    scene.notify_changed()
    
    return (remove_count, keep_count)


def get_point_cloud_stats():
    """
    Get statistics about the current point cloud.
    
    Returns:
        Dictionary with point cloud statistics
    """
    scene = lf.get_scene()
    if not scene or not scene.has_nodes():
        return None
    
    stats = {
        'nodes': [],
        'total_points': 0
    }
    
    for node in scene.get_nodes():
        pc = node.point_cloud()
        if pc is not None:
            means = pc.means.cpu().numpy()
            
            node_stats = {
                'name': node.name,
                'count': pc.size,
                'bounds_min': means.min(axis=0).tolist(),
                'bounds_max': means.max(axis=0).tolist(),
                'center': means.mean(axis=0).tolist(),
            }
            
            stats['nodes'].append(node_stats)
            stats['total_points'] += pc.size
    
    return stats


def analyze_point_density(voxel_size=0.01):
    """
    Analyze the neighbor density distribution of the point cloud.
    
    Args:
        voxel_size: Search radius for neighbors
        
    Returns:
        Dictionary with density statistics
    """
    scene = lf.get_scene()
    if not scene:
        return None
    
    # Find first point cloud
    target_pc = None
    for node in scene.get_nodes():
        pc = node.point_cloud()
        if pc is not None:
            target_pc = pc
            break
    
    if target_pc is None:
        return None
    
    means_np = target_pc.means.cpu().numpy()
    
    # Build KD-tree
    tree = cKDTree(means_np)
    neighbors_list = tree.query_ball_tree(tree, voxel_size)
    
    # Count neighbors for each point (excluding self)
    neighbor_counts = np.array([len(n) - 1 for n in neighbors_list])
    
    return {
        'total_points': len(means_np),
        'voxel_size': voxel_size,
        'min_neighbors': int(neighbor_counts.min()),
        'max_neighbors': int(neighbor_counts.max()),
        'mean_neighbors': float(neighbor_counts.mean()),
        'median_neighbors': float(np.median(neighbor_counts)),
        'isolated_1': int(np.sum(neighbor_counts <= 1)),
        'isolated_2': int(np.sum(neighbor_counts <= 2)),
        'isolated_5': int(np.sum(neighbor_counts <= 5)),
    }


# Example usage:
if __name__ == "__main__":
    # Get current stats
    stats = get_point_cloud_stats()
    if stats:
        print("Point Cloud Statistics:")
        print(f"  Total points: {stats['total_points']:,}")
        for node_stats in stats['nodes']:
            print(f"  Node '{node_stats['name']}':")
            print(f"    Count: {node_stats['count']:,}")
            print(f"    Bounds: {node_stats['bounds_min']} to {node_stats['bounds_max']}")
    
    # Analyze density
    density = analyze_point_density(voxel_size=0.01)
    if density:
        print("\nDensity Analysis (voxel_size=0.01):")
        print(f"  Mean neighbors: {density['mean_neighbors']:.1f}")
        print(f"  Median neighbors: {density['median_neighbors']:.0f}")
        print(f"  Points with ≤1 neighbor: {density['isolated_1']:,}")
        print(f"  Points with ≤2 neighbors: {density['isolated_2']:,}")
        print(f"  Points with ≤5 neighbors: {density['isolated_5']:,}")
    
    # Clean the point cloud
    result = clean_point_cloud(voxel_size=0.01, neighbor_threshold=2)
    if result:
        removed, remaining = result
        print(f"\nCleaning Result:")
        print(f"  Removed: {removed:,} points")
        print(f"  Remaining: {remaining:,} points")

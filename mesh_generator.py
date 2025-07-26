import numpy as np
import open3d as o3d
import cv2
import os

class MeshGenerator:
    def __init__(self):
        self.output_dir = "uploads"
    
    def create_mesh(self, image, depth_map, output_name="output_mesh.ply"):
        """
        Create 3D mesh from RGB image and depth map
        Args:
            image: RGB image (numpy array)
            depth_map: Depth map (numpy array)
            output_name: Output filename
        Returns:
            output_path: Path to generated mesh file
        """
        try:
            # Resize for faster processing
            h, w = image.shape[:2]
            if max(h, w) > 512:
                scale = 512 / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h))
                depth_map = cv2.resize(depth_map, (new_w, new_h))
            
            # Create point cloud
            point_cloud = self._create_point_cloud(image, depth_map)
            
            # Generate mesh
            mesh = self._generate_mesh_from_pointcloud(point_cloud)
            
            # Save mesh
            output_path = os.path.join(self.output_dir, output_name)
            o3d.io.write_triangle_mesh(output_path, mesh)
            
            return output_name
            
        except Exception as e:
            print(f"Mesh generation error: {e}")
            return self._create_simple_mesh(image, depth_map, output_name)
    
    def _create_point_cloud(self, image, depth_map):
        """Create point cloud from image and depth"""
        h, w = image.shape[:2]
        
        # Create coordinate grids
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Camera intrinsics (simplified)
        fx = fy = max(w, h)  # Focal length
        cx, cy = w // 2, h // 2  # Principal point
        
        # Convert depth to 3D coordinates
        z = depth_map.astype(np.float32) / 255.0 * 10  # Scale depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # Create point cloud
        points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        colors = image.reshape(-1, 3) / 255.0
        
        # Filter valid points
        valid_mask = z.flatten() > 0.1
        points = points[valid_mask]
        colors = colors[valid_mask]
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd
    
    def _generate_mesh_from_pointcloud(self, point_cloud):
        """Generate mesh from point cloud using Poisson reconstruction"""
        # Estimate normals
        point_cloud.estimate_normals()
        
        # Poisson surface reconstruction
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            point_cloud, depth=8
        )
        
        # Clean mesh
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        return mesh
    
    def _create_simple_mesh(self, image, depth_map, output_name):
        """Fallback: create simple plane mesh"""
        h, w = image.shape[:2]
        
        # Create vertices
        vertices = []
        colors = []
        faces = []
        
        # Sample points from image
        step = max(1, min(h, w) // 50)  # Adaptive sampling
        
        for i in range(0, h, step):
            for j in range(0, w, step):
                x = (j - w/2) / w * 2
                y = (h/2 - i) / h * 2
                z = depth_map[i, j] / 255.0
                
                vertices.append([x, y, z])
                colors.append(image[i, j] / 255.0)
        
        # Create simple mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        
        # Save mesh
        output_path = os.path.join(self.output_dir, output_name)
        o3d.io.write_triangle_mesh(output_path, mesh)
        
        return output_name
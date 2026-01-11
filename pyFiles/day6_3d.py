import open3d as o3d
import numpy as np

def load_sample_data():
    """Step 1: Get the 2D images (Color + Depth)."""
    dataset = o3d.data.SampleRedwoodRGBDImages()
    color = o3d.io.read_image(dataset.color_paths[0])
    depth = o3d.io.read_image(dataset.depth_paths[0])
    return color, depth

def create_point_cloud(color, depth):
    """Step 2: Turn the 2D pair into 3D dots."""
    # Combine into one object
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, convert_rgb_to_intensity=False
    )
    
    # Project into 3D space using standard camera settings
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
        )
    )
    
    # Flip it upright
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    
    # Calculate which way each point is facing (Normals)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
    
    return pcd

#the ball pass algorithm
def generate_mesh_bpa(pcd):
    """Step 3: Connect the dots into a solid surface."""
    radii = [0.005, 0.01, 0.02, 0.04]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    return mesh


def apply_laplacian_smoothing(mesh, iterations=5):
    """
    Refines the mesh by moving vertices toward the center of their neighbors.
    - mesh: The input O3D TriangleMesh.
    - iterations: How many times to apply the filter. 
      Higher = smoother (but the model may shrink).
    """
    print(f"Applying Laplacian smoothing ({iterations} iterations)...")
    
    #Smoothing Filter
    # filter_smooth_laplacian is the standard for cleaning BPA 'noise'
    mesh_smooth = mesh.filter_smooth_laplacian(number_of_iterations=iterations)
    
    #Recompute Normals
    mesh_smooth.compute_vertex_normals()
    
    return mesh_smooth

#possion is better if the data is noisy.
def generate_mesh_poisson(pcd):
    """
    Method B: Poisson Reconstruction (The 'Bubble Wrap' Method).
    Solves a math equation to create a completely smooth, watertight surface.
    """
    print("Running Poisson Reconstruction...")
    
    # 'depth' determines resolution. Higher (e.g. 10 or 12) = more detail but slower.
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=11
    )
    
    # Cleanup
    # Poisson creates a 'bubble' around the data. We need to pop the low-density parts.
    # We remove vertices that don't have enough original points near them.
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    return mesh

def export_mesh(mesh, file_path):
    """
    Exports the mesh to a file.
    - .obj: Keeps color and texture data.
    - .stl: Standard for 3D printing (geometry only).
    """
    print(f"Exporting mesh to {file_path}...")
    
    # This function returns a Boolean (T/F) if it succeeded
    success = o3d.io.write_triangle_mesh(file_path, mesh)
    
    if success:
        print("Export successful!")
    else:
        print("Export failed. Check your file path or permissions.")

def main():
    #Load
    color, depth = load_sample_data()
    
    #Process
    point_cloud = create_point_cloud(color, depth)
    mesh = generate_mesh_bpa(point_cloud)
    mesh = apply_laplacian_smoothing(mesh)

    #Visualize
    print("Visualizing Color Image...")
    o3d.visualization.draw_geometries([color])
    
    print("Visualizing Depth Map...")
    o3d.visualization.draw_geometries([depth])
    
    print("Visualizing Final 3D Mesh...")
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


    export_mesh(mesh, "meshes/mesh1.obj")

if __name__ == "__main__":
    main()
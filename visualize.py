import numpy as np
import matplotlib.pyplot as plt
from cdd import Matrix, Polyhedron, RepType
from scipy.spatial import ConvexHull

def visualize_polyhedron(poly, sample_points=False, num_points=25):
    vertices = np.array([gen[1:] for gen in poly.get_generators() if gen[0] == 1]) 
    print(vertices)
    dim = len(vertices[0])
    print("dimension is", dim)
    fig = plt.figure(figsize=(10, 8))
    if dim == 2:
        hull = ConvexHull(vertices)

        plt.plot(vertices[hull.vertices, 0], vertices[hull.vertices, 1], 'b-', linewidth=2)

        plt.plot([vertices[hull.vertices[-1], 0], vertices[hull.vertices[0], 0]],
                [vertices[hull.vertices[-1], 1], vertices[hull.vertices[0], 1]], 'b-', linewidth=2)

        plt.scatter(vertices[:, 0], vertices[:, 1], color='red', zorder=2)

        for i, (x, y) in enumerate(vertices):
            plt.text(x, y, f'({x:.2f}, {y:.2f})', fontsize=10, ha='right', color='black')

        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('2D Polyhedron Visualization')
        plt.grid(True)
        plt.axis('equal')  
        plt.savefig("polyhedron_plot.png", dpi=300, bbox_inches="tight")
    elif dim == 3:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        hull = ConvexHull(vertices)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                   color='red', s=50, label="Vertices")
        for i, (x, y, z) in enumerate(vertices):
            ax.text(x, y, z, f'({x:.2f}, {y:.2f}, {z:.2f})', color='black')
        
        faces = [vertices[simplex] for simplex in hull.simplices]
        poly3d = Poly3DCollection(faces, alpha=0.3, facecolor='cyan', edgecolor='blue', linewidths=1)
        ax.add_collection3d(poly3d)
        
        #  sample points in the bounding box and plot those inside
        if sample_points:
            equations = hull.equations  
            x_min, y_min, z_min = np.min(vertices, axis=0)
            x_max, y_max, z_max = np.max(vertices, axis=0)
            pts = np.random.uniform([x_min, y_min, z_min],
                                    [x_max, y_max, z_max],
                                    size=(num_points, 3))
            pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1)))) # constant term
            inside = np.all(np.dot(pts_hom, equations.T) <= 1e-12, axis=1)
            ax.scatter(pts[inside, 0], pts[inside, 1], pts[inside, 2],
                       color='green', s=10, alpha=0.3, label="Sampled Interior Points")
        
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title('3D Polyhedron Visualization')
        ax.legend()
        plt.savefig("polyhedron_plot_3d.png", dpi=300, bbox_inches="tight")
        plt.show()
    else:
        print(f"ERROR: can only visualize 2d or 3d, this has {dim} dimensions")


def create_example_polyhedron():
    # cube of vol 27
    mat = Matrix([
            [0,  1,  0,  0],  # x >= 0
            [0,  0,  1,  0],  # y >= 0
            [0,  0,  0,  1],  # z >= 0
            [3, -1,  0,  0],  # x <= 3
            [4,  0, -1,  0],  # y <= 3
            [5,  0,  0, -1],  # z <= 3
        ], number_type='float')

    poly = Polyhedron(mat)
    visualize_polyhedron(poly)

    # square
    h_rep_matrix = Matrix([
        [0,  1,  0],  # x >= 0
        [0,  0,  1],  # y >= 0
        [1, -1,  0],  # x <= 1
        [1,  0, -1],  # y <= 1
    ], number_type='float')
    poly = Polyhedron(h_rep_matrix)
    visualize_polyhedron(poly)

create_example_polyhedron()

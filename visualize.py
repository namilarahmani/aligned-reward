import numpy as np
import matplotlib.pyplot as plt
from cdd import Matrix, Polyhedron, RepType
from scipy.spatial import ConvexHull

def visualize_polyhedron(poly, sample_points=False, num_points=25):
    vertices = np.array([list(gen[1:]) for gen in poly.get_generators() if gen[0] == 1])
    rays = np.array([list(gen[1:]) for gen in poly.get_generators() if gen[0] == 0])
    
    print("vertices:")
    print(vertices)
    print("rays:")
    print(rays)
    
    if vertices.size == 0 and rays.size == 0:
        print("No vertices / rays found!")
        return
        
    dim = len(vertices[0])
    print("dimension is", dim)
    
    if dim == 2:
        if len(vertices) >= 3:
            hull = ConvexHull(vertices)
            plt.plot(vertices[hull.vertices, 0], vertices[hull.vertices, 1],
                     'b-', linewidth=2, label="Convex Hull")
            plt.plot([vertices[hull.vertices[-1], 0], vertices[hull.vertices[0], 0]],
                     [vertices[hull.vertices[-1], 1], vertices[hull.vertices[0], 1]],
                     'b-', linewidth=2)
        else:
            plt.plot(vertices[:, 0], vertices[:, 1], 'b-', linewidth=2, label="Vertices")
        
        plt.scatter(vertices[:, 0], vertices[:, 1], color='red', zorder=2, label="Vertices")
        for i, (x, y) in enumerate(vertices):
            plt.text(x, y, f'({x:.2f}, {y:.2f})', fontsize=10, ha='right', color='black')
        
        # plot rays
        if rays.size > 0:
            base = np.mean(vertices, axis=0)
            bbox = np.ptp(vertices, axis=0)  
            scale = 0.5 * np.linalg.norm(bbox)
            for i, ray in enumerate(rays):
                # draw an arrow from the base point in the ray direction
                dx, dy = ray * scale
                plt.arrow(base[0], base[1], dx, dy, head_width=scale*0.1,
                          head_length=scale*0.1, fc='green', ec='green',
                          length_includes_head=True, label="Ray" if i == 0 else "")
        
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('2D Polyhedron Visualization')
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        plt.savefig("polyhedron_plot.png", dpi=300, bbox_inches="tight")
        plt.show()

    elif dim == 3:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # plot vertices
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                   color='red', s=50, label="Vertices")
        for i, (x, y, z) in enumerate(vertices):
            ax.text(x, y, z, f'({x:.2f}, {y:.2f}, {z:.2f})', color='black')
        
        if len(vertices) >= 4:
            hull = ConvexHull(vertices)
            faces = [vertices[simplex] for simplex in hull.simplices]
            poly3d = Poly3DCollection(faces, alpha=0.3, facecolor='cyan', edgecolor='blue', linewidths=1)
            ax.add_collection3d(poly3d)
        
        # plot rays if any exist.
        if rays.size > 0:
            base = np.mean(vertices, axis=0)
            bbox = np.ptp(vertices, axis=0)
            scale = 0.5 * np.linalg.norm(bbox)
            for ray in rays:
                # ax.quiver(x, y, z, u, v, w, length=scale, normalize=True)
                ax.quiver(base[0], base[1], base[2],
                          ray[0], ray[1], ray[2],
                          length=scale, normalize=True, color='green', label="Ray")
        
        if sample_points and len(vertices) >= 4:
            equations = hull.equations  # each facet: a*x + b*y + c*z + d = 0
            x_min, y_min, z_min = np.min(vertices, axis=0)
            x_max, y_max, z_max = np.max(vertices, axis=0)
            pts = np.random.uniform([x_min, y_min, z_min],
                                    [x_max, y_max, z_max],
                                    size=(num_points, 3))
            pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1))))
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

    # custom shape - define vertices and rays (first val is flag)
    data = [
        [1, 0, 0, 1],
        [1, 1, 0, 2],
        [1, 0, 1, 2],
        [1, 1, 3, 2],
        [0, 0, 1, 0]
    ]

    # make cdd matrix in GENERATOR representation:
    mat = Matrix(data, number_type='fraction')
    mat.rep_type = RepType.GENERATOR
    poly = Polyhedron(mat)
    visualize_polyhedron(poly, sample_points=False, num_points=50)

create_example_polyhedron()

import numpy as np
import matplotlib.pyplot as plt
from cdd import Matrix, Polyhedron, RepType
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import linprog

def find_feasible_weights(pairs, preferences):
    '''
    Represent each preference as an inequality constraint and use scipy.linprog
    to find a set of feasible weights (or determine that none exist)
    '''
    A_neq = []
    b_neq = []
    A_eq = []
    b_eq = []

    epsilon = 1e-5  # small margin

    for (features_0, features_1), pref in zip(pairs, preferences):
        delta_f = features_1 - features_0

        if pref == 0:
            A_neq.append(delta_f)
            b_neq.append(-epsilon)
        elif pref == 1:
            A_neq.append(-delta_f)
            b_neq.append(-epsilon)
        elif pref == -1:
            A_eq.append(delta_f)
            b_eq.append(0)

    A_neq = np.array(A_neq) if A_neq else None
    b_neq = np.array(b_neq) if b_neq else None
    A_eq = np.array(A_eq) if A_eq else None
    b_eq = np.array(b_eq) if b_eq else None

    n_features = len(pairs[0][0])  # number of features
    result = linprog(c=np.zeros(n_features), A_ub=A_neq, b_ub=b_neq,
                     A_eq=A_eq, b_eq=b_eq, bounds=[(None, None)]*n_features)
    return result

def visualize_polyhedron(poly, file_name="polyhedron_plot.png", sample_points=True, num_points=45,
                         preference_plane=None, lp_solution=None):
    # Extract vertices (flag==1) and rays (flag==0)
    vertices = np.array([[float(x) for x in gen[1:]] 
                         for gen in poly.get_generators() if gen[0] == 1])
    rays = np.array([[float(x) for x in gen[1:]] 
                     for gen in poly.get_generators() if gen[0] == 0])
    
    print("vertices:")
    print(vertices)
    print("rays:")
    print(rays)
    
    if vertices.size == 0 and rays.size == 0:
        print("No vertices / rays found!")
        return
        
    dim = len(vertices[0])
    print("dimension is", dim)
    
    # If no LP solution was passed and if the polyhedron is in H-representation,
    # try to compute one using the inequalities.
    if lp_solution is None:
        try:
            # poly.get_inequalities() returns a Matrix with each row [b, c1, c2, ...]
            ineq = poly.get_inequalities()
            ineq_array = np.array(ineq)
            # Each inequality is of the form: b + c*x >= 0.
            # Convert to standard form: -c*x <= b.
            b_vec = ineq_array[:, 0]
            A = ineq_array[:, 1:]
            A_ub = -A
            b_ub = b_vec
            n_dim = A.shape[1]
            lp_solution = linprog(c=np.zeros(n_dim), A_ub=A_ub, b_ub=b_ub,
                                  bounds=[(None, None)]*n_dim)
            if lp_solution.success:
                print("LP solution computed:", lp_solution.x)
            else:
                print("LP solution not found:", lp_solution.message)
        except Exception as e:
            print("Could not compute LP solution from inequalities:", e)
    
    fig = plt.figure(figsize=(10, 8))
    
    if dim == 2:
        # 2D Plotting
        if len(vertices) >= 3:
            hull = ConvexHull(vertices)
            plt.plot(vertices[hull.vertices, 0], vertices[hull.vertices, 1],
                     'b-', linewidth=2, label="Convex Hull")
            # Close the loop
            plt.plot([vertices[hull.vertices[-1], 0], vertices[hull.vertices[0], 0]],
                     [vertices[hull.vertices[-1], 1], vertices[hull.vertices[0], 1]],
                     'b-', linewidth=2)
        else:
            plt.plot(vertices[:, 0], vertices[:, 1], 'b-', linewidth=2, label="Vertices")
        
        plt.scatter(vertices[:, 0], vertices[:, 1], color='red', zorder=2, label="Vertices")
        for i, (x, y) in enumerate(vertices):
            plt.text(x, y, f'({x:.2f}, {y:.2f})', fontsize=10, ha='right', color='black')
        
        # Draw rays as arrows
        if rays.size > 0:
            base = np.mean(vertices, axis=0)
            norm_bbox = np.linalg.norm(np.ptp(vertices, axis=0))
            scale = 0.5 * norm_bbox if (norm_bbox != 0) else 1
            arrow_width = scale * 0.05  
            for i, ray in enumerate(rays):
                dx, dy = ray * scale
                plt.arrow(base[0], base[1], dx, dy,
                          head_width=scale*0.2,
                          head_length=scale*0.2,
                          width=arrow_width,
                          fc='green', ec='green',
                          length_includes_head=True,
                          label="Ray" if i == 0 else "")
                plt.text(base[0]+dx, base[1]+dy, f"ray{i}", color='green')
        
        # Preference boundary and shading in 2D
        if preference_plane is not None:
            normal, offset = preference_plane  # normal: 2D vector, offset: scalar
            x_vals = np.linspace(np.min(vertices[:, 0]) - 1, np.max(vertices[:, 0]) + 1, 100)
            if abs(normal[1]) > 1e-6:
                y_vals = (-normal[0]*x_vals - offset) / normal[1]
            else:
                x_vals = np.full(100, -offset/normal[0])
                y_vals = np.linspace(np.min(vertices[:, 1]) - 1, np.max(vertices[:, 1]) + 1, 100)
            plt.plot(x_vals, y_vals, 'm--', linewidth=2, label="Preference Boundary")
            
            # Shade regions by evaluating the preference function on a grid
            grid_x, grid_y = np.meshgrid(np.linspace(np.min(vertices[:, 0]), np.max(vertices[:, 0]), 50),
                                          np.linspace(np.min(vertices[:, 1]), np.max(vertices[:, 1]), 50))
            grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
            f_vals = grid_points.dot(normal) + offset
            region1 = grid_points[f_vals >= 0]
            region2 = grid_points[f_vals < 0]
            plt.scatter(region1[:, 0], region1[:, 1], color='orange', alpha=0.2, label="Pref. region (>=0)")
            plt.scatter(region2[:, 0], region2[:, 1], color='purple', alpha=0.2, label="Pref. region (<0)")
        
        # Plot the LP solution (if found)
        if lp_solution is not None and lp_solution.success:
            sol = lp_solution.x
            plt.scatter(sol[0], sol[1], color='black', s=100, marker='o', label='LP Solution')
            plt.text(sol[0], sol[1], 'LP Sol', fontsize=10, ha='left', color='black')
        
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('2D Polyhedron Visualization')
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
        plt.show()

    elif dim == 3:
        # 3D Plotting
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                   color='red', s=50, label="Vertices")
        for i, (x, y, z) in enumerate(vertices):
            ax.text(x, y, z, f'({x:.2f}, {y:.2f}, {z:.2f})', color='black')
        
        # Draw original polyhedron via convex hull
        if len(vertices) >= 4:
            hull = ConvexHull(vertices)
            faces = [vertices[simplex] for simplex in hull.simplices]
            poly3d = Poly3DCollection(faces, alpha=0.1, facecolor='cyan', edgecolor='blue', linewidths=1)
            ax.add_collection3d(poly3d)
        
        # Plot rays
        if rays.size > 0:
            base = np.mean(vertices, axis=0)
            norm_bbox = np.linalg.norm(np.ptp(vertices, axis=0))
            scale = 0.5 * norm_bbox if (norm_bbox != 0) else 1
            for i, ray in enumerate(rays):
                ax.quiver(base[0], base[1], base[2],
                          ray[0], ray[1], ray[2],
                          length=scale, normalize=True, color='green', label="Ray" if i == 0 else "")
                tip = base + ray * scale
                ax.text(tip[0], tip[1], tip[2], f"ray{i}", color='green')
        
        # Shading for preference plane in 3D
        if preference_plane is not None:
            normal, offset = preference_plane  # normal: 3D, offset: scalar
            x_min, y_min, z_min = np.min(vertices, axis=0)
            x_max, y_max, z_max = np.max(vertices, axis=0)
            num_samples_large = max(500, num_points * 10)
            pts = np.random.uniform([x_min, y_min, z_min],
                                    [x_max, y_max, z_max],
                                    size=(num_samples_large, 3))
            pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1))))
            if len(vertices) >= 4:
                inside = np.all(np.dot(pts_hom, hull.equations.T) <= 1e-6, axis=1)
            else:
                inside = np.full(pts.shape[0], True)
            pts_inside = pts[inside]
            
            f_vals = pts_inside.dot(normal) + offset
            pos_pts = pts_inside[f_vals >= 0]
            neg_pts = pts_inside[f_vals < 0]
            
            if len(pos_pts) >= 4:
                try:
                    hull_pos = ConvexHull(pos_pts)
                    faces_pos = [pos_pts[simplex] for simplex in hull_pos.simplices]
                    poly3d_pos = Poly3DCollection(faces_pos, alpha=0.3, facecolor='orange', edgecolor='none')
                    ax.add_collection3d(poly3d_pos)
                except Exception as e:
                    print("Could not compute convex hull for positive region:", e)
            
            if len(neg_pts) >= 4:
                try:
                    hull_neg = ConvexHull(neg_pts)
                    faces_neg = [neg_pts[simplex] for simplex in hull_neg.simplices]
                    poly3d_neg = Poly3DCollection(faces_neg, alpha=0.3, facecolor='purple', edgecolor='none')
                    ax.add_collection3d(poly3d_neg)
                except Exception as e:
                    print("Could not compute convex hull for negative region:", e)
            
            # Also plot the hyperplane itself
            x0 = -offset / np.dot(normal, normal) * np.array(normal)
            if abs(normal[0]) < abs(normal[1]):
                v = np.array([1, 0, 0])
            else:
                v = np.array([0, 1, 0])
            v1 = np.cross(normal, v)
            if np.linalg.norm(v1) < 1e-6:
                v = np.array([0, 0, 1])
                v1 = np.cross(normal, v)
            v1 /= np.linalg.norm(v1)
            v2 = np.cross(normal, v1)
            v2 /= np.linalg.norm(v2)
            L = np.max(np.ptp(vertices, axis=0))
            grid_range = np.linspace(-L, L, 10)
            U, V = np.meshgrid(grid_range, grid_range)
            plane_points = x0[np.newaxis, np.newaxis, :] + U[..., np.newaxis]*v1 + V[..., np.newaxis]*v2
            ax.plot_surface(plane_points[:, :, 0], plane_points[:, :, 1], plane_points[:, :, 2],
                            color='magenta', alpha=0.2)
        
        # Plot the LP solution (if found)
        if lp_solution is not None and lp_solution.success:
            sol = lp_solution.x
            ax.scatter(sol[0], sol[1], sol[2], color='black', s=100, marker='o', label='LP Solution')
            ax.text(sol[0], sol[1], sol[2], 'LP Sol', color='black')
        
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title('3D Polyhedron Visualization')
        ax.view_init(elev=20, azim=30)
        ax.legend()
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
        plt.show()
        
    else:
        print(f"ERROR: can only visualize 2d or 3d, this has {dim} dimensions")

def main():
    # 3D ex: cube (H-representation)
    mat = Matrix([
            [0,  1,  0,  0],  # x >= 0
            [0,  0,  1,  0],  # y >= 0
            [0,  0,  0,  1],  # z >= 0
            [3, -1,  0,  0],  # x <= 3  (1 - x >= 0)
            [4,  0, -1,  0],  # y <= 3
            [5,  0,  0, -1],  # z <= 3
        ], number_type='float')
    poly = Polyhedron(mat)
    preference_plane = (np.array([1, -1, 0], dtype=float), -0.5)
    visualize_polyhedron(poly, file_name="polyhedron_3dcube.png", preference_plane=preference_plane)

    # 2D ex: square (H-representation)
    h_rep_matrix = Matrix([
        [0,  1,  0],  # x >= 0
        [0,  0,  1],  # y >= 0
        [1, -1,  0],  # x <= 1
        [1,  0, -1],  # y <= 1
    ], number_type='float')
    poly = Polyhedron(h_rep_matrix)
    preference_plane_2d = (np.array([1, -1], dtype=float), 0)
    visualize_polyhedron(poly, file_name="square.png", preference_plane=preference_plane_2d)

    # custom shape: defined via generator representation (LP solution not computed in this case)
    data = [
        [1, 0, 0, 1],
        [1, 1, 0, 2],
        [1, 0, 1, 2],
        [1, 1, 3, 2],
        [0, 0, 1, 0]
    ]
    mat = Matrix(data, number_type='fraction')
    mat.rep_type = RepType.GENERATOR
    poly = Polyhedron(mat)
    visualize_polyhedron(poly, sample_points=False, num_points=50)

if __name__ == "__main__":
    main()

import time
import numpy as np
import pyvista as pv


def generate_perfect_rotated_hcp(radius=500.0, box_x=7000.0, box_y=7000.0, box_z=3000.0):
    """Generates a perfect, mathematically rigid rotated HCP crystal lattice inside the 7x7x3km box."""
    dx = 2.0 * radius
    dy = np.sqrt(3.0) * radius
    dz = 2.0 * np.sqrt(2.0 / 3.0) * radius

    max_dim = max(box_x, box_y) * 1.5
    search_limit_xy = int(max_dim / min(dx, dy)) + 3
    search_limit_z = int(box_z / dz) + 3

    sphere_data = []
    rot_angle = np.radians(15.0)
    cos_r, sin_r = np.cos(rot_angle), np.sin(rot_angle)

    for k in range(-5, search_limit_z):
        z_raw = k * dz
        for j in range(-search_limit_xy, search_limit_xy):
            y_raw = j * dy
            for i in range(-search_limit_xy, search_limit_xy):
                x_raw = i * dx

                if j % 2 == 1:
                    x_raw += radius
                if k % 2 == 1:
                    x_raw += radius / 3.0
                    y_raw += radius / np.sqrt(3.0)

                x_rot = x_raw * cos_r - y_raw * sin_r
                y_rot = x_raw * sin_r + y_raw * cos_r
                z_rot = z_raw

                if 0 <= x_rot <= box_x and 0 <= y_rot <= box_y and 0 <= z_rot <= box_z:
                    sphere_data.append(([x_rot, y_rot, z_rot], k))

    return sphere_data


def check_local_point_inside_sector(x, y, z, r_min, r_max, theta_max_deg, z_max):
    """Checks if a single Cartesian point is inside the base sector envelope."""
    if z < 0 or z > z_max:
        return False
    r = np.sqrt(x ** 2 + y ** 2)
    if r < r_min or r > r_max:
        return False
    theta = np.degrees(np.arctan2(y, x))
    return 0 <= theta <= theta_max_deg


def evaluate_sphere_intersection(center, radius, offset_x, offset_y, offset_z,
                                 r_min, r_max, theta_max_deg, z_max):
    """
    Samples internal points of a sphere to estimate containment metrics.
    Returns: (is_any_inside, is_majority_inside)
    """
    # Shift center to the moving sector frame of reference
    cx = center[0] - offset_x
    cy = center[1] - offset_y
    cz = center[2] - offset_z

    # Fast check: if the center is nowhere near the sector boundaries, skip sampling
    # Sector bounding radius is r_max, max Z is z_max
    dist_from_origin = np.sqrt(cx ** 2 + cy ** 2)
    if dist_from_origin > (r_max + radius) or dist_from_origin < (r_min - radius):
        return False, False
    if cz > (z_max + radius) or cz < -radius:
        return False, False

    # Define a low-overhead, symmetric 7-point sampling cloud inside the sphere
    # (Center + 6 cardinal direction surface anchors)
    sample_offsets = np.array([
        [0, 0, 0],
        [radius * 0.8, 0, 0], [-radius * 0.8, 0, 0],
        [0, radius * 0.8, 0], [0, -radius * 0.8, 0],
        [0, 0, radius * 0.8], [0, 0, -radius * 0.8]
    ])

    inside_count = 0
    for sx, sy, sz in sample_offsets:
        if check_local_point_inside_sector(cx + sx, cy + sy, cz + sz, r_min, r_max, theta_max_deg, z_max):
            inside_count += 1

    any_inside = inside_count > 0
    majority_inside = (inside_count / len(sample_offsets)) > 0.5

    return any_inside, majority_inside


# ==========================================================
# Execution & Marching Simulation Setup
# ==========================================================
if __name__ == "__main__":
    SPHERE_R = 500.0
    BOX_X, BOX_Y, BOX_Z = 7000.0, 7000.0, 3000.0  # Scaled up to 7x7x3km

    SECTOR_R_MIN = 2000.0
    SECTOR_R_MAX = 5000.0
    SECTOR_HEIGHT = 1000.0
    SECTOR_ANGLE = 90.0

    V_sphere = (4.0 / 3.0) * np.pi * (SPHERE_R ** 3)
    V_sector = (SECTOR_ANGLE / 360.0) * np.pi * (SECTOR_R_MAX ** 2 - SECTOR_R_MIN ** 2) * SECTOR_HEIGHT

    print(f"Populating static {int(BOX_X / 1000)}x{int(BOX_Y / 1000)}x{int(BOX_Z / 1000)}km HCP Crystal Block...")
    static_hcp_block = generate_perfect_rotated_hcp(SPHERE_R, BOX_X, BOX_Y, BOX_Z)
    print(f"Lattice initialized with {len(static_hcp_block)} total baseline spheres.\n")

    # Marching resolution sweeps (Step every 1000 meters across the larger space)
    x_steps = np.arange(0, 2500, 1000)
    y_steps = np.arange(0, 2500, 1000)
    z_steps = np.arange(0, 2500, 1000)

    print("--- Starting Marching Annular Simulation ---")
    best_void_fraction = 1.0
    optimal_offset = (0, 0, 0)
    optimal_any_list = []
    optimal_majority_count = 0

    for sz in z_steps:
        for sy in y_steps:
            for sx in x_steps:
                any_inside_spheres = []
                majority_inside_count = 0

                for item in static_hcp_block:
                    pos, layer_k = item
                    any_in, maj_in = evaluate_sphere_intersection(
                        pos, SPHERE_R, sx, sy, sz,
                        SECTOR_R_MIN, SECTOR_R_MAX, SECTOR_ANGLE, SECTOR_HEIGHT
                    )

                    if any_in:
                        any_inside_spheres.append(item)
                    if maj_in:
                        majority_inside_count += 1

                # Base void metrics on spheres that have stable majority presence
                total_sphere_vol = majority_inside_count * V_sphere
                packing_fraction = total_sphere_vol / V_sector
                void_fraction = max(0.0, 1.0 - packing_fraction)

                print(f"Sector Origin -> [X: {sx:4.0f}m, Y: {sy:4.0f}m, Z: {sz:4.0f}m] | "
                      f"Highlighted (Any): {len(any_inside_spheres):2d} | "
                      f"Majority (>50%): {majority_inside_count:2d} | "
                      f"Practical Void Fraction: {void_fraction * 100:5.2f}%")

                if void_fraction < best_void_fraction and packing_fraction <= 0.7405:
                    best_void_fraction = void_fraction
                    optimal_offset = (sx, sy, sz)
                    optimal_any_list = any_inside_spheres
                    optimal_majority_count = majority_inside_count

                time.sleep(0.02)

    print("\n--- Simulation Complete ---")
    ox, oy, oz = optimal_offset
    print(f"Optimal Location Matrix: [X: {ox}m, Y: {oy}m, Z: {oz}m]")
    print(f"Total Highlighted Spheres (Any part inside): {len(optimal_any_list)}")
    print(f"Spheres with >50% volume inside: {optimal_majority_count}")
    print(f"Minimum Practical Void Fraction (based on >50% criteria): {best_void_fraction * 100:.2f}%")

    # ==========================================================
    # Rendering the Scene
    # ==========================================================
    print("\nPreparing PyVista Scene...")
    plotter = pv.Plotter()
    plotter.add_axes()
    plotter.show_grid(xtitle='X (m)', ytitle='Y (m)', ztitle='Z (m)')

    # Generate sector mesh at the optimal offset location
    r_coords = np.linspace(SECTOR_R_MIN, SECTOR_R_MAX, 50)
    theta_coords = np.linspace(0, np.radians(SECTOR_ANGLE), 50)
    z_coords = np.linspace(0, SECTOR_HEIGHT, 50)
    r_mat, t_mat, z_mat = np.meshgrid(r_coords, theta_coords, z_coords, indexing='ij')

    sector_x = r_mat * np.cos(t_mat) + ox
    sector_y = r_mat * np.sin(t_mat) + oy
    sector_z = z_mat + oz
    sector_mesh = pv.StructuredGrid(sector_x, sector_y, sector_z)

    plotter.add_mesh(sector_mesh, color='lime', opacity=0.15, show_edges=True, edge_color='darkgreen')

    # Convert the optimal highlighted group into a fast-lookup set
    highlighted_positions = {tuple(pos) for pos, _ in optimal_any_list}

    # Render out all spheres inside the 7x7x3km box space
    for center, layer_k in static_hcp_block:
        sphere = pv.Sphere(radius=SPHERE_R, center=center, theta_resolution=16, phi_resolution=16)

        if tuple(center) in highlighted_positions:
            # Highlight spheres that touch the sector space
            s_color = 'cyan' if layer_k % 2 == 0 else 'lightcoral'
            s_opacity = 0.90
            e_color = 'blue' if layer_k % 2 == 0 else 'darkred'
        else:
            # Fade out background spheres outside the sector footprint
            s_color = 'lightgray'
            s_opacity = 0.05
            e_color = 'silver'

        plotter.add_mesh(sphere, color=s_color, opacity=s_opacity, show_edges=True, edge_color=e_color)

    plotter.camera_position = [
        (14000.0, 14000.0, 10000.0),
        (3500.0, 3500.0, 1500.0),
        (0.0, 0.0, 1.0)
    ]
    plotter.show()
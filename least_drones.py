import numpy as np
import pyvista as pv


def generate_perfect_rotated_hcp(radius=500.0, box_x=7000.0, box_y=7000.0, box_z=7000.0):
    """Generates a perfect, mathematically rigid rotated HCP crystal lattice inside the 7x7x7km box."""
    dx = 2.0 * radius
    dy = np.sqrt(3.0) * radius
    dz = 2.0 * np.sqrt(2.0 / 3.0) * radius

    max_dim = max(box_x, box_y, box_z) * 1.5
    search_limit = int(max_dim / min(dx, dy, dz)) + 3

    sphere_data = []
    rot_angle = np.radians(15.0)
    cos_r, sin_r = np.cos(rot_angle), np.sin(rot_angle)

    for k in range(-5, search_limit):
        z_raw = k * dz
        for j in range(-search_limit, search_limit):
            y_raw = j * dy
            for i in range(-search_limit, search_limit):
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


def get_rotation_matrix(yaw_deg, pitch_deg, roll_deg):
    """Generates a standard 3D rotation matrix from Euler angles."""
    y = np.radians(yaw_deg)
    p = np.radians(pitch_deg)
    r = np.radians(roll_deg)

    R_y = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    R_p = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    R_r = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])

    return R_y @ R_p @ R_r


def is_sector_completely_inside_box(origin, R_matrix, r_min=2000.0, r_max=5000.0,
                                    theta_max_deg=90.0, z_max=1000.0, box_dim=7000.0):
    """Verifies if the 8 extreme corners of the sector's bounding box fit inside the 7x7x7km cube."""
    t_max = np.radians(theta_max_deg)

    local_corners = np.array([
        [r_min, 0, 0], [r_max, 0, 0],
        [r_min * np.cos(t_max), r_min * np.sin(t_max), 0],
        [r_max * np.cos(t_max), r_max * np.sin(t_max), 0],
        [r_min, 0, z_max], [r_max, 0, z_max],
        [r_min * np.cos(t_max), r_min * np.sin(t_max), z_max],
        [r_max * np.cos(t_max), r_max * np.sin(t_max), z_max]
    ])

    global_corners = (R_matrix @ local_corners.T).T + origin
    return np.all((global_corners >= 0) & (global_corners <= box_dim))


def check_transformed_point_inside_sector(pt, origin, R_matrix_inv, r_min, r_max, theta_max_deg, z_max):
    """Transforms a global point back to the sector's local space to run containment checks."""
    local_pt = R_matrix_inv @ (pt - origin)
    lx, ly, lz = local_pt

    if lz < 0 or lz > z_max:
        return False
    r = np.sqrt(lx ** 2 + ly ** 2)
    if r < r_min or r > r_max:
        return False
    theta = np.degrees(np.arctan2(ly, lx))
    return 0 <= theta <= theta_max_deg


def evaluate_sphere_containment_ratio(center, radius, origin, R_matrix_inv, r_min, r_max, theta_max_deg, z_max):
    """Samples internal points of a sphere to check for section containment."""
    sample_offsets = np.array([
        [0, 0, 0],
        [radius * 0.8, 0, 0], [-radius * 0.8, 0, 0],
        [0, radius * 0.8, 0], [0, -radius * 0.8, 0],
        [0, 0, radius * 0.8], [0, 0, -radius * 0.8]
    ])

    inside_count = 0
    for sx, sy, sz in sample_offsets:
        if check_transformed_point_inside_sector(center + np.array([sx, sy, sz]), origin,
                                                 R_matrix_inv, r_min, r_max, theta_max_deg, z_max):
            inside_count += 1

    return inside_count / len(sample_offsets)


class MarchingAnnularManager:
    def __init__(self, sphere_radius, box_dim, search_iterations=20):
        self.radius = sphere_radius
        self.box_dim = box_dim
        self.iterations = search_iterations

        self.s_rmin, self.s_rmax = 2000.0, 5000.0
        self.s_height, self.s_angle = 1000.0, 90.0
        self.V_sector = (self.s_angle / 360.0) * np.pi * (self.s_rmax ** 2 - self.s_rmin ** 2) * self.s_height

        print(
            f"Populating static {int(box_dim / 1000)}x{int(box_dim / 1000)}x{int(box_dim / 1000)}km HCP Crystal Block...")
        self.static_hcp = generate_perfect_rotated_hcp(self.radius, box_dim, box_dim, box_dim)

        self.origin = None
        self.R_matrix = None
        self.highlighted_spheres = []

        # Find the layout that MINIMIZES >75% sphere containment
        self.find_minimal_immersion_sector()
        self.run_containment_analysis(self.radius, label="Initial State (Minimized Layout)")

    def find_minimal_immersion_sector(self):
        """Evaluates multiple valid configurations to isolate the one with the lowest >75% collision matrix."""
        print(
            f"Running Monte Carlo loop over {self.iterations} unique valid orientations to minimize >75% immersion...")

        min_75_count = float('inf')
        best_origin = None
        best_R = None

        valid_found = 0
        while valid_found < self.iterations:
            rand_origin = np.random.uniform(0, self.box_dim, 3)
            yaw = np.random.uniform(0, 360)
            pitch = np.random.uniform(0, 360)
            roll = np.random.uniform(0, 360)

            R = get_rotation_matrix(yaw, pitch, roll)

            if is_sector_completely_inside_box(rand_origin, R, self.s_rmin, self.s_rmax,
                                               self.s_angle, self.s_height, self.box_dim):
                valid_found += 1

                # Evaluate >75% count for this valid orientation instance
                R_inv = R.T
                count_75 = 0
                for item in self.static_hcp:
                    pos, _ = item
                    ratio = evaluate_sphere_containment_ratio(
                        np.array(pos), self.radius, rand_origin, R_inv,
                        self.s_rmin, self.s_rmax, self.s_angle, self.s_height
                    )
                    if ratio > 0.75:
                        count_75 += 1

                print(f"  Valid Iteration {valid_found}/{self.iterations} | Spheres (>75%): {count_75}")

                # Update tracking if this configuration has fewer heavily immersed spheres
                if count_75 < min_75_count:
                    min_75_count = count_75
                    best_origin = rand_origin
                    best_R = R

        self.origin = best_origin
        self.R_matrix = best_R
        print(f"\nOptimal Minimized Configuration Fixed!")
        print(f"Minimum Found Spheres (>75%): {min_75_count}")

    def run_containment_analysis(self, target_radius, label="Current State"):
        """Analyzes and lists the exact enclosure distribution counts for the given sphere radius."""
        R_inv = self.R_matrix.T

        count_0 = 0
        count_25 = 0
        count_50 = 0
        count_75 = 0

        temp_highlighted = []

        for item in self.static_hcp:
            pos, layer_k = item
            ratio = evaluate_sphere_containment_ratio(
                np.array(pos), target_radius, self.origin, R_inv,
                self.s_rmin, self.s_rmax, self.s_angle, self.s_height
            )

            if ratio > 0.0:
                count_0 += 1
                temp_highlighted.append(item)
            if ratio > 0.25:
                count_25 += 1
            if ratio > 0.50:
                count_50 += 1
            if ratio > 0.75:
                count_75 += 1

        if label == "Initial State (Minimized Layout)":
            self.highlighted_spheres = temp_highlighted

        print(f"\n==========================================")
        print(f" ENCLOSURE METRICS BREAKDOWN ({label})")
        print(f"==========================================")
        print(f" Spheres Enclosed > 75%:  {count_75}")
        print(f" Spheres Enclosed > 50%:  {count_50}")
        print(f" Spheres Enclosed > 25%:  {count_25}")
        print(f" Spheres Enclosed >  0%:  {count_0} (Total Highlighted)")
        print(f"==========================================\n")

        return count_50

    def expi(self, expansion_offset):
        new_radius = self.radius + expansion_offset
        print(f"Executing expi({expansion_offset}). Expanding active spheres to {new_radius}m...")

        majority_count = self.run_containment_analysis(new_radius, label=f"Post-expi({expansion_offset})")

        V_new_sphere = (4.0 / 3.0) * np.pi * (new_radius ** 3)
        total_expanded_vol = majority_count * V_new_sphere
        packing_fraction = min(0.7405, total_expanded_vol / self.V_sector)
        new_void_fraction = max(0.0, 1.0 - packing_fraction)

        print(f"Post-Expansion Void Fraction (Based on >50%): {new_void_fraction * 100:.2f}%")
        return new_radius

    def render_scene(self, active_radius):
        print("Assembling 3D PyVista Canvas...")
        plotter = pv.Plotter()
        plotter.add_axes()
        plotter.show_grid(xtitle='X (m)', ytitle='Y (m)', ztitle='Z (m)')

        r_coords = np.linspace(self.s_rmin, self.s_rmax, 40)
        theta_coords = np.linspace(0, np.radians(self.s_angle), 40)
        z_coords = np.linspace(0, self.s_height, 20)
        r_m, t_m, z_m = np.meshgrid(r_coords, theta_coords, z_coords, indexing='ij')

        local_x = r_m * np.cos(t_m)
        local_y = r_m * np.sin(t_m)
        local_z = z_m

        pts = np.vstack([local_x.ravel(), local_y.ravel(), local_z.ravel()])
        transformed_pts = (self.R_matrix @ pts).T + self.origin

        sector_mesh = pv.StructuredGrid(
            transformed_pts[:, 0].reshape(local_x.shape),
            transformed_pts[:, 1].reshape(local_y.shape),
            transformed_pts[:, 2].reshape(local_z.shape)
        )

        plotter.add_mesh(
            sector_mesh,
            color='green',
            opacity=0.45,
            show_edges=True,
            edge_color='black',
            line_width=3.5
        )

        highlighted_set = {tuple(pos) for pos, _ in self.highlighted_spheres}

        for center, layer_k in self.static_hcp:
            is_highlighted = tuple(center) in highlighted_set
            r_render = active_radius if is_highlighted else self.radius
            sphere = pv.Sphere(radius=r_render, center=center, theta_resolution=12, phi_resolution=12)

            if is_highlighted:
                s_color = 'cyan' if layer_k % 2 == 0 else 'lightcoral'
                s_opacity = 0.85
                e_color = 'blue' if layer_k % 2 == 0 else 'darkred'
            else:
                s_color = 'lightgray'
                s_opacity = 0.02
                e_color = 'silver'

            plotter.add_mesh(sphere, color=s_color, opacity=s_opacity, show_edges=is_highlighted, edge_color=e_color)

        plotter.camera_position = [
            (15000.0, 15000.0, 12000.0),
            (3500.0, 3500.0, 3500.0),
            (0.0, 0.0, 1.0)
        ]
        plotter.show()


if __name__ == "__main__":
    BASE_RADIUS = 500.0
    BOX_SIZE = 7000.0

    # search_iterations defines how many valid random locations to check
    # to find the minimal immersion point. Increase for a deeper search.
    manager = MarchingAnnularManager(sphere_radius=BASE_RADIUS, box_dim=BOX_SIZE, search_iterations=30)
    final_radius = manager.expi(expansion_offset=150)
    manager.render_scene(active_radius=final_radius)
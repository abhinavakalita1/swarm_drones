import numpy as np
import pyvista as pv


def create_radar_sector(inner_radius=2000.0, outer_radius=5000.0,
                        angle_deg=90.0, height=1000.0, resolution=100):
    """
    Creates a 3D sector wedge (hollow cylinder segment) using a structured grid topology.
    """
    r_coords = np.linspace(inner_radius, outer_radius, resolution)
    theta_coords = np.linspace(0, np.radians(angle_deg), resolution)
    z_coords = np.linspace(0, height, resolution)

    r, theta, z = np.meshgrid(r_coords, theta_coords, z_coords, indexing='ij')

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    grid = pv.StructuredGrid(x, y, z)
    return grid


class RadarVisualizer:
    def __init__(self):
        self.plotter = pv.Plotter()
        self.plotter.add_axes()
        self.plotter.show_grid(xtitle='X (m)', ytitle='Y (m)', ztitle='Z (m)')

        self.sector_mesh = create_radar_sector()
        self.plotter.add_mesh(
            self.sector_mesh,
            color='lime',
            opacity=0.25,
            show_edges=True,
            edge_color='darkgreen',
            label='Radar Scanning Zone'
        )

        self.sphere_actors = []

    def spawn_target_sphere(self, x, y, z, radius=500.0, color='red'):
        """Spawns a sphere using standard Cartesian coordinates."""
        sphere_geom = pv.Sphere(radius=radius, center=(x, y, z))
        actor = self.plotter.add_mesh(
            sphere_geom,
            color=color,
            opacity=0.90,
            show_edges=False,
        )
        self.sphere_actors.append(actor)
        return actor

    def spawn_target_cylindrical(self, r, theta_deg, z, radius=500.0, color='red'):
        """
        Spawns a sphere using cylindrical coordinates.

        Parameters:
        - r: Radius / Range from origin (meters)
        - theta_deg: Bearing angle (degrees, where 0 is along the +X axis)
        - z: Elevation / Altitude (meters)
        """
        # Convert theta from degrees to radians
        theta_rad = np.radians(theta_deg)

        # --- Mathematical Conversion to Cartesian ---
        x = r * np.cos(theta_rad)
        y = r * np.sin(theta_rad)

        # Forward the calculated positions to our standard spawning mechanism
        print(
            f"Spawning target at Cylindrical(r={r}m, θ={theta_deg}°, z={z}m) -> Cartesian(X={x:.1f}m, Y={y:.1f}m, Z={z:.1f}m)")
        return self.spawn_target_sphere(x, y, z, radius, color)

    def render_scene(self):
        self.plotter.camera_position = [
            (8000.0, 8000.0, 6000.0),
            (2500.0, 2500.0, 500.0),
            (0.0, 0.0, 1.0)
        ]
        self.plotter.show()


# ==========================================================
# Scenario Simulation Setup
# ==========================================================
if __name__ == "__main__":
    visualizer = RadarVisualizer()

    # --- Spawning Targets Using Cylindrical Coordinates (r, theta, z) ---


    # Target A: Right on the inner boundary (2000m), at a 45-degree bearing, 500m up
    for i in range(10, 96, 23):
        visualizer.spawn_target_cylindrical(r=2500.0, theta_deg=i, z=500.0, color='crimson')

    # Target B: Dead center of the radar envelope (3500m out, 30-degree bearing, 200m up)
    for i in range(7, 96, 90//6):
        visualizer.spawn_target_cylindrical(r=3500.0, theta_deg=i, z=500.0, color='orange')

    # Target C: Outer boundary edge limit (5000m out, right at the 90-degree edge, 800m up)
    for i in range(5, 90, 90//8):
        visualizer.spawn_target_cylindrical(r=4500.0, theta_deg=i, z=500.0, color='blue')

    visualizer.render_scene()

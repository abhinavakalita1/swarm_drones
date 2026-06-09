"""
hcp_drone_swarm.py

Two-phase script:

PHASE 1 — HCP crystal + marching sector simulation
  Runs the full HCP lattice and marching annular sector sweep.
  At the end, saves ALL highlighted sphere positions (optimal_any_list)
  to  hcp_sphere_positions.json  so Phase 2 can load them.

PHASE 2 — PyBullet drone swarm
  Reads hcp_sphere_positions.json.
  Scales all positions by 0.01  (so 500 m → 5.0 PyBullet units).
  Spawns one Drone at each sphere position.
  Radar range:
    base range = 500 m  →  5.0 units  (after ×0.01)
    if the optimal sector offset had any axis ≥ 150 m, use 650 m → 6.5 units
  Assigns a vertical rotating radar to every 3rd drone (index 0, 3, 6 …).
  Camera framed to see the whole swarm.

Toggle PHASE = 1 or 2 at the top, or set PHASE = "both" to run sequentially.
"""

import time
import numpy as np
import json
import os
import math
import pybullet as p
import pybullet_data

# ── optionally import pyvista only when needed ────────────────
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    print("[WARN] pyvista not installed — Phase 1 visualisation skipped")


SCALE        = 0.01                      # metres → PyBullet units
SAVE_PATH    = "hcp_sphere_positions.json"
PHASE        = "both"                    # 1 | 2 | "both"


# ══════════════════════════════════════════════════════════════
# PHASE 1 FUNCTIONS  (HCP + marching sector)
# ══════════════════════════════════════════════════════════════

def generate_perfect_rotated_hcp(radius=500.0, box_x=7000.0,
                                  box_y=7000.0, box_z=3000.0):
    dx = 2.0 * radius
    dy = np.sqrt(3.0) * radius
    dz = 2.0 * np.sqrt(2.0 / 3.0) * radius

    max_dim          = max(box_x, box_y) * 1.5
    search_limit_xy  = int(max_dim / min(dx, dy)) + 3
    search_limit_z   = int(box_z / dz) + 3

    sphere_data = []
    rot_angle   = np.radians(15.0)
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


def check_local_point_inside_sector(x, y, z, r_min, r_max,
                                    theta_max_deg, z_max):
    if z < 0 or z > z_max:
        return False
    r = np.sqrt(x**2 + y**2)
    if r < r_min or r > r_max:
        return False
    theta = np.degrees(np.arctan2(y, x))
    return 0 <= theta <= theta_max_deg


def evaluate_sphere_intersection(center, radius, offset_x, offset_y, offset_z,
                                 r_min, r_max, theta_max_deg, z_max):
    cx = center[0] - offset_x
    cy = center[1] - offset_y
    cz = center[2] - offset_z

    dist_from_origin = np.sqrt(cx**2 + cy**2)
    if (dist_from_origin > (r_max + radius) or
            dist_from_origin < (r_min - radius)):
        return False, False
    if cz > (z_max + radius) or cz < -radius:
        return False, False

    sample_offsets = np.array([
        [0, 0, 0],
        [radius*0.8, 0, 0], [-radius*0.8, 0, 0],
        [0, radius*0.8, 0], [0, -radius*0.8, 0],
        [0, 0, radius*0.8], [0, 0, -radius*0.8]
    ])

    inside_count = 0
    for sx, sy, sz in sample_offsets:
        if check_local_point_inside_sector(
                cx+sx, cy+sy, cz+sz,
                r_min, r_max, theta_max_deg, z_max):
            inside_count += 1

    return inside_count > 0, (inside_count / len(sample_offsets)) > 0.5


def run_phase1():
    SPHERE_R       = 500.0
    BOX_X, BOX_Y, BOX_Z = 7000.0, 7000.0, 3000.0
    SECTOR_R_MIN   = 2000.0
    SECTOR_R_MAX   = 5000.0
    SECTOR_HEIGHT  = 1000.0
    SECTOR_ANGLE   = 90.0

    V_sphere = (4.0/3.0) * np.pi * (SPHERE_R**3)
    V_sector = ((SECTOR_ANGLE/360.0) * np.pi *
                (SECTOR_R_MAX**2 - SECTOR_R_MIN**2) * SECTOR_HEIGHT)

    print(f"Populating HCP block ({int(BOX_X/1000)}×"
          f"{int(BOX_Y/1000)}×{int(BOX_Z/1000)} km)…")
    static_hcp_block = generate_perfect_rotated_hcp(SPHERE_R, BOX_X, BOX_Y, BOX_Z)
    print(f"  {len(static_hcp_block)} spheres in lattice\n")

    x_steps = np.arange(0, 2500, 1000)
    y_steps = np.arange(0, 2500, 1000)
    z_steps = np.arange(0, 2500, 1000)

    best_void       = 1.0
    optimal_offset  = (0, 0, 0)
    optimal_any     = []
    optimal_maj_cnt = 0

    print("--- Marching annular sector sweep ---")
    for sz in z_steps:
        for sy in y_steps:
            for sx in x_steps:
                any_list  = []
                maj_count = 0
                for item in static_hcp_block:
                    pos, layer_k = item
                    any_in, maj_in = evaluate_sphere_intersection(
                        pos, SPHERE_R, sx, sy, sz,
                        SECTOR_R_MIN, SECTOR_R_MAX, SECTOR_ANGLE, SECTOR_HEIGHT
                    )
                    if any_in:
                        any_list.append(item)
                    if maj_in:
                        maj_count += 1

                packing  = (maj_count * V_sphere) / V_sector
                void_frac = max(0.0, 1.0 - packing)

                print(f"  [{sx:4.0f}, {sy:4.0f}, {sz:4.0f}] m  "
                      f"any={len(any_list):2d}  maj={maj_count:2d}  "
                      f"void={void_frac*100:5.2f}%")

                if void_frac < best_void and packing <= 0.7405:
                    best_void      = void_frac
                    optimal_offset = (sx, sy, sz)
                    optimal_any    = any_list
                    optimal_maj_cnt = maj_count

                time.sleep(0.02)

    ox, oy, oz = optimal_offset
    print(f"\n--- Phase 1 complete ---")
    print(f"  Optimal offset      : [{ox}, {oy}, {oz}] m")
    print(f"  Highlighted spheres : {len(optimal_any)}")
    print(f"  Majority (>50%)     : {optimal_maj_cnt}")
    print(f"  Best void fraction  : {best_void*100:.2f}%")

    # ── Save sphere positions ─────────────────────────────────
    data = {
        "optimal_offset":  [float(v) for v in optimal_offset],
        "sphere_radius_m": float(SPHERE_R),
        "positions": [
            {"pos": [float(v) for v in pos], "layer_k": int(lk)}
            for pos, lk in optimal_any
        ]
    }
    with open(SAVE_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {len(optimal_any)} positions → {SAVE_PATH}\n")

    # ── PyVista visualisation ─────────────────────────────────
    if HAS_PYVISTA:
        plotter = pv.Plotter()
        plotter.add_axes()
        plotter.show_grid(xtitle="X (m)", ytitle="Y (m)", ztitle="Z (m)")

        r_c  = np.linspace(SECTOR_R_MIN, SECTOR_R_MAX, 50)
        th_c = np.linspace(0, np.radians(SECTOR_ANGLE), 50)
        z_c  = np.linspace(0, SECTOR_HEIGHT, 50)
        r_m, t_m, z_m = np.meshgrid(r_c, th_c, z_c, indexing="ij")
        sec_mesh = pv.StructuredGrid(
            r_m*np.cos(t_m)+ox, r_m*np.sin(t_m)+oy, z_m+oz)
        plotter.add_mesh(sec_mesh, color="lime", opacity=0.15,
                         show_edges=True, edge_color="darkgreen")

        highlighted = {tuple(pos) for pos, _ in optimal_any}
        for center, lk in static_hcp_block:
            sph = pv.Sphere(radius=SPHERE_R, center=center,
                            theta_resolution=16, phi_resolution=16)
            if tuple(center) in highlighted:
                c  = "cyan"        if lk % 2 == 0 else "lightcoral"
                op = 0.90
                ec = "blue"        if lk % 2 == 0 else "darkred"
            else:
                c, op, ec = "lightgray", 0.05, "silver"
            plotter.add_mesh(sph, color=c, opacity=op,
                             show_edges=True, edge_color=ec)

        plotter.camera_position = [
            (14000, 14000, 10000), (3500, 3500, 1500), (0, 0, 1)]
        plotter.show()

    return optimal_offset


# ══════════════════════════════════════════════════════════════
# PHASE 2 CLASSES  (Drone + radar, from drone script)
# ══════════════════════════════════════════════════════════════

class Drone:
    def __init__(self, position, orientation=(0, 0, 0)):
        quat     = p.getQuaternionFromEuler(orientation)
        urdf_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "drone.urdf")

        if os.path.exists(urdf_path):
            self.id = p.loadURDF(urdf_path, basePosition=position,
                                 baseOrientation=quat, useFixedBase=False)
        else:
            # fallback
            self.id = p.loadURDF(
                "sphere2.urdf", basePosition=position,
                baseOrientation=quat, useFixedBase=False)

        self.mass = p.getDynamicsInfo(self.id, -1)[0]
        for i in range(p.getNumJoints(self.id)):
            self.mass += p.getDynamicsInfo(self.id, i)[0]

        self._prop_joints = []
        self._prop_dirs   = [1, -1, -1, 1]
        try:
            joint_names = ["prop_fl_joint","prop_fr_joint",
                           "prop_rl_joint","prop_rr_joint"]
            joint_map = {}
            for i in range(p.getNumJoints(self.id)):
                info = p.getJointInfo(self.id, i)
                joint_map[info[1].decode()] = i
            for name in joint_names:
                jid = joint_map[name]
                self._prop_joints.append(jid)
                p.setJointMotorControl2(
                    self.id, jid,
                    controlMode=p.VELOCITY_CONTROL, force=0)
        except Exception:
            pass

        self._radars = []

    def add_rotating_vertical_radar(self, local_pos, range=5.0,
                                    v_fov=120, h_elevation=20,
                                    angular_resolution=5):
        half  = 0.005
        vis   = p.createVisualShape(p.GEOM_BOX,
                                    halfExtents=[half, half, half],
                                    rgbaColor=[0.2, 0.8, 0.2, 1.0])
        cube_id = p.createMultiBody(baseMass=0,
                                    baseVisualShapeIndex=vis,
                                    basePosition=[0, 0, 0])
        radar = dict(
            local_pos=np.array(local_pos, dtype=float),
            base_fwd=np.array([0.0, 1.0, 0.0]),
            range=range,
            v_fov=v_fov,
            h_elevation=h_elevation,
            angular_resolution=angular_resolution,
            cube_id=cube_id,
            current_rotation_deg=0.0,
            has_completed_sweep=False,
            all_persistent_lines=[]
        )
        self._radars.append(radar)
        return radar

    def _build_ray_frame(self, fwd):
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(fwd, world_up)) > 0.99:
            world_up = np.array([1.0, 0.0, 0.0])
        right = np.cross(fwd, world_up)
        right /= np.linalg.norm(right)
        up    = np.cross(right, fwd)
        up   /= np.linalg.norm(up)
        return right, up

    def _get_vertical_ray_directions(self, radar, fwd):
        right, up = self._build_ray_frame(fwd)
        v_half = radar["v_fov"]       / 2
        h_half = radar["h_elevation"] / 2
        step   = radar["angular_resolution"]
        dirs   = []
        for ha in np.arange(-h_half, h_half + step, step):
            for va in np.arange(-v_half, v_half + step, step):
                d  = (fwd
                      + math.tan(math.radians(ha)) * right
                      + math.tan(math.radians(va)) * up)
                d /= np.linalg.norm(d)
                dirs.append(d)
        return dirs

    def update_and_sweep_radar(self):
        pos_pb, orn = p.getBasePositionAndOrientation(self.id)
        rot        = np.array(
            p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        drone_pos  = np.array(pos_pb)

        for radar in self._radars:
            world_pos = drone_pos + rot @ radar["local_pos"]
            p.resetBasePositionAndOrientation(
                radar["cube_id"], world_pos.tolist(), [0,0,0,1])

            if radar["has_completed_sweep"]:
                continue

            yaw_r = math.radians(radar["current_rotation_deg"])
            c, s  = math.cos(yaw_r), math.sin(yaw_r)
            z_rot = np.array([[c,-s,0],[s,c,0],[0,0,1]])
            fwd   = z_rot @ radar["base_fwd"]

            for d in self._get_vertical_ray_directions(radar, fwd):
                tp  = world_pos + d * radar["range"]
                lid = p.addUserDebugLine(
                    world_pos.tolist(), tp.tolist(),
                    lineColorRGB=[1.0, 0.0, 0.0], lineWidth=2.0)
                radar["all_persistent_lines"].append(lid)

            radar["current_rotation_deg"] += 10.0
            if radar["current_rotation_deg"] >= 360.0:
                radar["has_completed_sweep"] = True
                print(f"  Drone @ {drone_pos.round(2)} finished sweep.")

    def hover(self):
        pos, _ = p.getBasePositionAndOrientation(self.id)
        p.applyExternalForce(
            self.id, -1,
            [0, 0, self.mass * 9.81], pos, p.WORLD_FRAME)
        for jid, direction in zip(self._prop_joints, self._prop_dirs):
            p.setJointMotorControl2(
                self.id, jid,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=direction * 20.0, force=0.1)


# ══════════════════════════════════════════════════════════════
# PHASE 2 ENTRY
# ══════════════════════════════════════════════════════════════

def run_phase2(optimal_offset=None):
    # ── Load saved positions ──────────────────────────────────
    if not os.path.exists(SAVE_PATH):
        raise FileNotFoundError(
            f"{SAVE_PATH} not found. Run Phase 1 first "
            f"(set PHASE = 1 or PHASE = 'both').")

    with open(SAVE_PATH) as f:
        data = json.load(f)

    raw_positions    = [(item["pos"], item["layer_k"])
                        for item in data["positions"]]
    saved_offset     = data.get("optimal_offset", [0, 0, 0])
    sphere_radius_m  = data.get("sphere_radius_m", 500.0)

    print(f"[INFO] Loaded {len(raw_positions)} sphere positions from {SAVE_PATH}")
    print(f"[INFO] Saved optimal offset: {saved_offset} m")

    # ── Radar range decision ──────────────────────────────────
    # If any axis of the optimal offset is ≥ 150 m, use 650 m range
    offset_to_check = optimal_offset if optimal_offset else saved_offset
    radar_range_m   = (650.0
                       if any(abs(v) >= 150 for v in offset_to_check)
                       else 500.0)
    radar_range_pb  = radar_range_m * SCALE
    print(f"[INFO] Radar range: {radar_range_m:.0f} m → "
          f"{radar_range_pb:.2f} PyBullet units")

    # ── Scale all positions ───────────────────────────────────
    scaled_positions = [
        ([v * SCALE for v in pos], lk)
        for pos, lk in raw_positions
    ]

    # ── PyBullet scene ────────────────────────────────────────
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    # ── Camera: frame the whole swarm ─────────────────────────
    all_pts   = np.array([pos for pos, _ in scaled_positions])
    centre    = all_pts.mean(axis=0)
    extent    = np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0))
    cam_dist  = max(extent * 1.2, 5.0)

    p.resetDebugVisualizerCamera(
        cameraDistance=cam_dist,
        cameraYaw=35,
        cameraPitch=-30,
        cameraTargetPosition=centre.tolist()
    )
    print(f"[INFO] Camera → centre {centre.round(2)}, "
          f"dist {cam_dist:.1f}")

    # ── Spawn ALL drones — none have radar yet ────────────────
    drones = []
    for idx, (pos, lk) in enumerate(scaled_positions):
        drone = Drone(position=pos, orientation=(0, 0, 0))
        drones.append(drone)

    n_drones = len(drones)
    print(f"[INFO] Spawned {n_drones} drones (0 with radar so far)\n")

    # ── Slider ────────────────────────────────────────────────
    # "Drones with radar" — starts at 1, goes up to total drones.
    # Slider RIGHT = more drones get radar, incrementally.
    # Moving LEFT does NOT remove already-assigned radars
    # (you can't un-attach a radar from a drone mid-sim).
    sid_radar = p.addUserDebugParameter(
        "Drones with radar  (increase to add more)",
        0, n_drones, 1
    )

    # Assign radar to drone 0 immediately (slider starts at 1)
    n_radars_assigned = 0

    def assign_radar_to_drone(drone):
        drone.add_rotating_vertical_radar(
            local_pos=[0, 0.15 * SCALE, 0],
            range=radar_range_pb,
            v_fov=120,
            h_elevation=2,
            angular_resolution=10
        )

    # Give the first drone its radar right away
    assign_radar_to_drone(drones[0])
    n_radars_assigned = 1
    print(f"[INFO] Radar assigned to drone 0  (1/{n_drones})")

    # ── Simulation loop ───────────────────────────────────────
    while True:
        try:
            slider_val = int(round(
                p.readUserDebugParameter(sid_radar)))
        except Exception:
            print("[INFO] PyBullet window closed.")
            break

        # Clamp to valid range
        slider_val = max(0, min(n_drones, slider_val))

        # If slider moved UP: assign radars to the next batch of drones
        if slider_val > n_radars_assigned:
            for i in range(n_radars_assigned, slider_val):
                assign_radar_to_drone(drones[i])
                print(f"[INFO] Radar assigned to drone {i}  "
                      f"({i+1}/{n_drones})")
            n_radars_assigned = slider_val

        # Step all drones
        for i, drone in enumerate(drones):
            drone.hover()
            # Only call radar update on drones that have one
            if i < n_radars_assigned and drone._radars:
                drone.update_and_sweep_radar()

        p.stepSimulation()
        time.sleep(1.0 / 240.0)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    optimal_offset = None

    if PHASE == 1 or PHASE == "both":
        optimal_offset = run_phase1()

    if PHASE == 2 or PHASE == "both":
        run_phase2(optimal_offset=optimal_offset)
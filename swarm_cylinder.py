import pybullet as p
import pybullet_data
import time
import os
import math
import numpy as np


class Drone:
    def __init__(self, position, orientation=(0, 0, 0)):
        """
        position: [x,y,z]
        orientation: [roll,pitch,yaw] in radians
        """
        quat = p.getQuaternionFromEuler(orientation)
        urdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drone.urdf")

        # Fallback handling in case local drone.urdf is missing
        if os.path.exists(urdf_path):
            self.id = p.loadURDF(urdf_path, basePosition=position, baseOrientation=quat, useFixedBase=False)
        else:
            self.id = p.loadURDF("kuka_lwr/kuka_lwr.urdf", basePosition=position, baseOrientation=quat,
                                 useFixedBase=False)

        self.mass = 0
        self.mass += p.getDynamicsInfo(self.id, -1)[0]
        for i in range(p.getNumJoints(self.id)):
            self.mass += p.getDynamicsInfo(self.id, i)[0]
        print("Total mass:", self.mass)

        # cache propeller joint indices
        self._prop_joints = []
        self._prop_dirs = [1, -1, -1, 1]
        try:
            joint_names = ["prop_fl_joint", "prop_fr_joint", "prop_rl_joint", "prop_rr_joint"]
            joint_map = {}
            for i in range(p.getNumJoints(self.id)):
                info = p.getJointInfo(self.id, i)
                joint_map[info[1].decode()] = i
            for name in joint_names:
                jid = joint_map[name]
                self._prop_joints.append(jid)
                p.setJointMotorControl2(self.id, jid, controlMode=p.VELOCITY_CONTROL, force=0)
        except Exception:
            pass  # Fallback doesn't use matching joint names

        self._radars = []  # list of radar dicts

    def _spin_propellers(self):
        for jid, direction in zip(self._prop_joints, self._prop_dirs):
            p.setJointMotorControl2(
                self.id, jid,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=direction * 20.0,
                force=0.1
            )

    # ===================================================
    # Vertical Rotating Radar Implementation
    # ===================================================

    def add_rotating_vertical_radar(self, local_pos, range=5, v_fov=120, h_elevation=20, angular_resolution=5):
        """
        Adds a radar designed to sweep vertically.
        v_fov       : Vertical fan envelope spanning up/down (replacing standard horizontal fov)
        h_elevation : Horizontal thin beam width (replacing standard vertical elevation)
        """
        # Visual mount indicator
        half = 0.005
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half, half, half], rgbaColor=[0.2, 0.8, 0.2, 1.0])
        cube_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=[0, 0, 0])

        radar = dict(
            local_pos=np.array(local_pos, dtype=float),
            base_fwd=np.array([0.0, 1.0, 0.0]),  # Starts pointing at world +Y
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

    def _build_ray_frame(self, forward_vector):
        fwd = forward_vector
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(fwd, world_up)) > 0.99:
            world_up = np.array([1.0, 0.0, 0.0])
        right = np.cross(fwd, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        up /= np.linalg.norm(up)
        return right, up

    def _get_vertical_ray_directions(self, radar, current_fwd):
        """Generates a vertical slice profile rotated to the current tracking angle."""
        right, up = self._build_ray_frame(current_fwd)

        v_half = radar["v_fov"] / 2
        h_half = radar["h_elevation"] / 2
        step = radar["angular_resolution"]

        # Vertically oriented fans loop through vertical angles primarily
        v_angles = np.arange(-v_half, v_half + step, step)
        h_angles = np.arange(-h_half, h_half + step, step)

        dirs = []
        for ha in h_angles:
            for va in v_angles:
                ha_r = math.radians(ha)
                va_r = math.radians(va)
                # Left/Right thinning map to 'right', Up/Down spreading maps to 'up'
                d = current_fwd + math.tan(ha_r) * right + math.tan(va_r) * up
                d /= np.linalg.norm(d)
                dirs.append(d)
        return dirs

    def update_and_sweep_radar(self):
        drone_pos, drone_orn = p.getBasePositionAndOrientation(self.id)
        rot = np.array(p.getMatrixFromQuaternion(drone_orn)).reshape(3, 3)
        drone_pos = np.array(drone_pos)

        for radar in self._radars:
            world_pos = drone_pos + rot @ radar["local_pos"]
            p.resetBasePositionAndOrientation(radar["cube_id"], world_pos.tolist(), [0, 0, 0, 1])

            # Terminate drawing updates once a full 360-degree sweep is registered
            if radar["has_completed_sweep"]:
                continue

            # Calculate current heading vector using Z-axis rotation matrix
            yaw_r = math.radians(radar["current_rotation_deg"])
            c, s = math.cos(yaw_r), math.sin(yaw_r)
            z_rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            current_fwd = z_rot @ radar["base_fwd"]

            # Compute the vertical sector ray vectors
            dirs = self._get_vertical_ray_directions(radar, current_fwd)

            # --- FIX: Draw lines using the actual ray directions 'd' instead of 'current_fwd' ---
            for d in dirs:
                target_point = world_pos + d * radar["range"]  # <-- Changed 'current_fwd' to 'd'
                lid = p.addUserDebugLine(
                    world_pos.tolist(),
                    target_point.tolist(),
                    lineColorRGB=[1.0, 0.0, 0.0],
                    lineWidth=2.0
                )
                radar["all_persistent_lines"].append(lid)

            # Step by 10 degrees per tick
            radar["current_rotation_deg"] += 10.0
            if radar["current_rotation_deg"] >= 360.0:
                radar["has_completed_sweep"] = True
                print(f"Drone at position {drone_pos.tolist()} finished its 360-degree radar sweep.")

    def hover(self):
        pos, _ = p.getBasePositionAndOrientation(self.id)
        p.applyExternalForce(self.id, -1, [0, 0, self.mass * 9.81], pos, p.WORLD_FRAME)
        self._spin_propellers()


# ==========================================================
# Simulation Entry Point
# ==========================================================

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
plane = p.loadURDF("plane.urdf")

swarm_config = [
    # --- Echelon 1 ---
    {"id": 1, "pos": (-150, 2500, 500)},  # Included
    {"id": 2, "pos": (-150, 2500, 850)},
    {"id": 3, "pos": (150, 2500, 500)},
    {"id": 4, "pos": (150, 2500, 850)},
    # --- Echelon 2 ---
    {"id": 5, "pos": (-150, 3500, 500)},  # Included
    {"id": 6, "pos": (-150, 3500, 850)},
    {"id": 7, "pos": (150, 3500, 500)},
    {"id": 8, "pos": (150, 3500, 850)},
    # --- Echelon 3 ---
    {"id": 9, "pos": (-150, 4500, 500)},  # Included
    {"id": 10, "pos": (-150, 4500, 850)},
    {"id": 11, "pos": (150, 4500, 500)},
    {"id": 12, "pos": (150, 4500, 850)},
]

drones_map = {}
for cfg in swarm_config:
    scaled_pos = [x / 100 for x in list(cfg["pos"])]
    d = Drone(position=scaled_pos, orientation=(0, 0, 0))

    # Only assign rotating vertical radars to Drone 1, 5, and 9
    if cfg["id"] in [1, 5, 9]:
        d.add_rotating_vertical_radar(
            local_pos=[0, 0.15, 0],
            range=5,
            v_fov=120,  # Wide up/down sweep envelope
            h_elevation=2,  # Thin horizontal razor profile
            angular_resolution=10
        )
    drones_map[cfg["id"]] = d

# Frame the active scanning column (Drone 1 focus area)
p.resetDebugVisualizerCamera(
    cameraDistance=12.0,
    cameraYaw=35,
    cameraPitch=-25,
    cameraTargetPosition=[-1.5, 25.0, 5.0]
)

while True:
    for idx, d in drones_map.items():
        d.hover()
        # Only step the radar loop logic for the targeted units
        if idx in [1, 5, 9]:
            d.update_and_sweep_radar()

    p.stepSimulation()
    time.sleep(1 / 240)
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



        self.id = p.loadURDF(

            urdf_path,

            basePosition=position,

            baseOrientation=quat,

            useFixedBase=False

        )



        self.mass = 0

        self.mass += p.getDynamicsInfo(self.id, -1)[0]

        for i in range(p.getNumJoints(self.id)):

            self.mass += p.getDynamicsInfo(self.id, i)[0]

        print("Total mass:", self.mass)



        # cache propeller joint indices

        self._prop_joints = []

        self._prop_dirs = [1, -1, -1, 1]

        joint_names = ["prop_fl_joint", "prop_fr_joint", "prop_rl_joint", "prop_rr_joint"]

        joint_map = {}

        for i in range(p.getNumJoints(self.id)):

            info = p.getJointInfo(self.id, i)

            joint_map[info[1].decode()] = i

        for name in joint_names:

            jid = joint_map[name]

            self._prop_joints.append(jid)

            p.setJointMotorControl2(self.id, jid, controlMode=p.VELOCITY_CONTROL, force=0)



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

    # Radar

    # ===================================================



    def add_radar(self, local_pos, ray_vector, range=500, fov=120, elevation=20, angular_resolution=8):

        """

        local_pos   : [x,y,z] position on base_link surface where radar cube sits

        ray_vector  : [x,y,z] direction the radar faces (world-aligned, normalized internally)

        range       : max ray length in metres (default 500)

        fov         : horizontal field of view in degrees, ±fov/2 (default 120)

        elevation   : vertical field of view in degrees, ±elevation/2 (default 20)

        """



        # --- visual cube (surface-connected, no physics) ---

        half = 0.005

        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half, half, half],

                                  rgbaColor=[0.2, 0.8, 0.2, 1.0])

        cube_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,

                                    basePosition=[0, 0, 0])  # moved in update



        # normalise ray vector

        rv = np.array(ray_vector, dtype=float)

        rv /= np.linalg.norm(rv)



        radar = dict(

            local_pos=np.array(local_pos, dtype=float),

            ray_vector=rv,

            range=range,

            fov=fov,

            elevation=elevation,

            angular_resolution=angular_resolution,

            cube_id=cube_id,

            line_ids=[],

        )

        self._radars.append(radar)

        return radar



    def _build_ray_frame(self, ray_vector):

        """Return (right, up) orthonormal vectors for the given forward direction."""

        fwd = ray_vector

        # pick a world-up that isn't parallel to fwd

        world_up = np.array([0.0, 0.0, 1.0])

        if abs(np.dot(fwd, world_up)) > 0.99:

            world_up = np.array([1.0, 0.0, 0.0])

        right = np.cross(fwd, world_up)

        right /= np.linalg.norm(right)

        up = np.cross(right, fwd)

        up /= np.linalg.norm(up)

        return right, up



    def _radar_ray_directions(self, radar):

        """

        Generate ray directions spanning ±fov/2 horizontally and

        ±elevation/2 vertically, sampled every 5 degrees.

        """

        fwd    = radar["ray_vector"]

        h_half = radar["fov"] / 2

        v_half = radar["elevation"] / 2

        step   = radar["angular_resolution"]



        right, up = self._build_ray_frame(fwd)



        h_angles = np.arange(-h_half, h_half + step, step)

        v_angles = np.arange(-v_half, v_half + step, step)



        dirs = []

        for va in v_angles:

            for ha in h_angles:

                ha_r = math.radians(ha)

                va_r = math.radians(va)

                d = fwd + math.tan(ha_r) * right + math.tan(va_r) * up

                d /= np.linalg.norm(d)

                dirs.append(d)



        return dirs



    def _dash_segments(self, origin, direction, range, dash=0.3, gap=0.3):

        """

        Return list of (start, end) pairs forming a dashed line

        from origin along direction for the given range.

        """

        segments = []

        t = 0.0

        while t < range:

            t_end = min(t + dash, range)

            segments.append((

                origin + direction * t,

                origin + direction * t_end

            ))

            t = t_end + gap

        return segments



    def _update_radars(self):

        drone_pos, drone_orn = p.getBasePositionAndOrientation(self.id)

        rot = np.array(p.getMatrixFromQuaternion(drone_orn)).reshape(3, 3)

        drone_pos = np.array(drone_pos)



        for radar in self._radars:

            # world position of radar cube

            world_pos = drone_pos + rot @ radar["local_pos"]



            # move cube

            p.resetBasePositionAndOrientation(

                radar["cube_id"],

                world_pos.tolist(),

                [0, 0, 0, 1]

            )



            # ray directions (fixed in world frame as per spec)

            dirs = self._radar_ray_directions(radar)



            # build flat list of all dash segments across all rays

            all_segs = []

            for d in dirs:

                all_segs.extend(self._dash_segments(world_pos, d, radar["range"]))



            # create or update debug lines (one line per dash segment)

            if not radar["line_ids"]:

                for seg_start, seg_end in all_segs:

                    lid = p.addUserDebugLine(

                        seg_start.tolist(), seg_end.tolist(),

                        lineColorRGB=[1.0, 0.0, 0.0],

                        lineWidth=3.0

                    )

                    radar["line_ids"].append(lid)

            else:

                # number of segments may differ if drone moved; rebuild if needed

                if len(radar["line_ids"]) != len(all_segs):

                    for lid in radar["line_ids"]:

                        p.removeUserDebugItem(lid)

                    radar["line_ids"] = []

                    for seg_start, seg_end in all_segs:

                        lid = p.addUserDebugLine(

                            seg_start.tolist(), seg_end.tolist(),

                            lineColorRGB=[1.0, 0.0, 0.0],

                            lineWidth=3.0

                        )

                        radar["line_ids"].append(lid)

                else:

                    for i, (lid, (seg_start, seg_end)) in enumerate(

                            zip(radar["line_ids"], all_segs)):

                        radar["line_ids"][i] = p.addUserDebugLine(

                            seg_start.tolist(), seg_end.tolist(),

                            lineColorRGB=[1.0, 0.0, 0.0],

                            lineWidth=3.0,

                            replaceItemUniqueId=lid

                        )



    # ===================================================

    # Hover controller

    # ===================================================



    def hover(self):

        pos, _ = p.getBasePositionAndOrientation(self.id)

        p.applyExternalForce(self.id, -1, [0, 0, self.mass * 9.81],

                             pos, p.WORLD_FRAME)

        self._spin_propellers()

        self._update_radars()



    # ===================================================

    # Translational movement

    # ===================================================



    def move_forward(self, force=5):

        pos, _ = p.getBasePositionAndOrientation(self.id)

        p.applyExternalForce(self.id, -1, [force, 0, 0], pos, p.WORLD_FRAME)



    def move_backward(self, force=5):

        self.move_forward(-force)



    def move_right(self, force=5):

        pos, _ = p.getBasePositionAndOrientation(self.id)

        p.applyExternalForce(self.id, -1, [0, -force, 0], pos, p.WORLD_FRAME)



    def move_left(self, force=5):

        self.move_right(-force)



    def move_up(self, force=5):

        pos, _ = p.getBasePositionAndOrientation(self.id)

        p.applyExternalForce(self.id, -1, [0, 0, force], pos, p.WORLD_FRAME)



    def move_down(self, force=5):

        self.move_up(-force)



    # ===================================================

    # Rotations

    # ===================================================



    def yaw(self, torque=1):

        p.applyExternalTorque(self.id, -1, [0, 0, torque], p.WORLD_FRAME)



    def pitch(self, torque=1):

        p.applyExternalTorque(self.id, -1, [0, torque, 0], p.WORLD_FRAME)



    def roll(self, torque=1):

        p.applyExternalTorque(self.id, -1, [torque, 0, 0], p.WORLD_FRAME)



    def get_pose(self):

        return p.getBasePositionAndOrientation(self.id)





# ==========================================================

# Helper: build gimbal-adjusted ray vector

# ==========================================================



def make_ray_vector(pan_deg, tilt_deg):

    """

    Base facing direction is +Y (toward the threat zone).

    pan_deg  : yaw offset in degrees  (+ = right, - = left)

    tilt_deg : pitch offset in degrees (+ = up,   - = down)

    Returns a unit vector.

    """

    pan_r  = math.radians(pan_deg)

    tilt_r = math.radians(tilt_deg)



    # Start with +Y, rotate by tilt around X-axis, then pan around Z-axis

    # tilt: rotate (0,1,0) around X by tilt_r  ->  (0, cos(tilt), sin(tilt))

    # pan:  rotate that around Z by pan_r

    vy = math.cos(tilt_r)

    vz = math.sin(tilt_r)

    vx = 0.0



    # now pan around Z

    rx =  vx * math.cos(pan_r) - vy * math.sin(pan_r)

    ry =  vx * math.sin(pan_r) + vy * math.cos(pan_r)

    rz =  vz



    v = np.array([rx, ry, rz])

    v /= np.linalg.norm(v)

    return v.tolist()





# ==========================================================

# Simulation

# ==========================================================



physicsClient = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -9.81)

plane = p.loadURDF("plane.urdf")



# ----------------------------------------------------------

# Swarm configuration

#

# Columns   : Left  → pan = -15°,  Right → pan = +15°

# Altitudes : Low   → tilt = +10° (beam up),  High → tilt = -10° (beam down)

# Radar     : range=500 m, fov=120°, elevation=20° (±10°), angular_res=1°

# ----------------------------------------------------------



RADAR_KWARGS = dict(

    local_pos=[0, 0.15, 0],   # front surface of base_link (faces +Y)

    range=10,

    fov=120,

    elevation=110,

    angular_resolution=10,

)



swarm_config = [

    # --- Echelon 1: Inner Perimeter (y ≈ 2500 m) ---

    # Drone 1  Left  Low   pan=-15  tilt=+10

    {"pos": (-150, 2500, 500),  "pan": -15, "tilt": +10},

    # Drone 2  Left  High  pan=-15  tilt=-10

    {"pos": (-150, 2500, 850),  "pan": -15, "tilt": -10},

    # Drone 3  Right Low   pan=+15  tilt=+10

    {"pos": ( 150, 2500, 500),  "pan": +15, "tilt": +10},

    # Drone 4  Right High  pan=+15  tilt=-10

    {"pos": ( 150, 2500, 850),  "pan": +15, "tilt": -10},



    # --- Echelon 2: Core Mid-Layer (y ≈ 3500 m) ---

    # Drone 5  Left  Low

    {"pos": (-150, 3500, 500),  "pan": -15, "tilt": +10},

    # Drone 6  Left  High

    {"pos": (-150, 3500, 850),  "pan": -15, "tilt": -10},

    # Drone 7  Right Low

    {"pos": ( 150, 3500, 500),  "pan": +15, "tilt": +10},

    # Drone 8  Right High

    {"pos": ( 150, 3500, 850),  "pan": +15, "tilt": -10},



    # --- Echelon 3: Vanguard Outpost (y ≈ 4500 m) ---

    # Drone 9  Left  Low

    {"pos": (-150, 4500, 500),  "pan": -15, "tilt": +10},

    # Drone 10 Left  High

    {"pos": (-150, 4500, 850),  "pan": -15, "tilt": -10},

    # Drone 11 Right Low

    {"pos": ( 150, 4500, 500),  "pan": +15, "tilt": +10},

    # Drone 12 Right High

    {"pos": ( 150, 4500, 850),  "pan": +15, "tilt": -10},

]



drones = []

for cfg in swarm_config:

    foo = [x/100 for x in list(cfg["pos"])]

    d = Drone(position=foo, orientation=(0, 0, 0))

    rv = make_ray_vector(cfg["pan"], cfg["tilt"])

    d.add_radar(ray_vector=rv, **RADAR_KWARGS)

    drones.append(d)



# Reset camera to look at the first echelon (around Y = 25)

p.resetDebugVisualizerCamera(

    cameraDistance=15.0,     # How far back to pull the camera

    cameraYaw=45,            # Angle around the Z axis

    cameraPitch=-30,         # Angle looking down

    cameraTargetPosition=[0, 25.0, 5.0]  # Focus directly on the first row of drones

)



while True:

    for d in drones:

        d.hover()

    p.stepSimulation()

    time.sleep(1 / 240)


import numpy as np
import os, time

from isaacgym import gymtorch
from isaacgym import gymapi

import torch
from typing import Tuple, Dict

from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, torch_rand_float, normalize, quat_apply, quat_rotate_inverse, quat_rotate, get_euler_xyz, quat_from_euler_xyz
from isaacgymenvs.tasks.base.vec_task import VecTask
from utils.constraint_manager import ConstraintManager
from tasks.terrainParkour import Terrain
from texttable import Texttable
import itertools

class SoloParkour(VecTask):
    """Environment to learn locomotion on complex terrains with the Solo-12 quadruped robot."""

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg
        self.init_done = False

        # Client to control the target velocity with a gamepad
        self.useJoystick = self.cfg["env"]["enableJoystick"] and self.cfg["test"]
        if self.useJoystick:
            from Joystick import Joystick

            self.joystick = Joystick()
            self.joystick.update_v_ref(0, 0)

        # Scales of observations
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.height_meas_scale = self.cfg["env"]["learn"]["heightMeasurementScale"]
        self.imu_scale = self.cfg["env"]["learn"]["imuAccelerationScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        # Scales of rewards
        self.rew_scales = {}
        self.rew_scales["survivalBonus"] = self.cfg["env"]["learn"]["survivalBonus"]


        self.numRewards = -1

        # TODO: Remove?
        self.rew_mult = self.cfg["env"]["learn"]["rewMult"]
        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.rew_mult

        # Scales of constraints (CaT)
        self.useConstraints = self.cfg["env"]["learn"]["enableConstraints"]
        if self.useConstraints == "cat":
            # Use constraints as terminations
            self.constraints = {}
            self.constraints["air_time"] = self.cfg["env"]["learn"]["constraints_CaT"]["feetAirTimeConstraint"]
            self.constraints["soft_p"] = self.cfg["env"]["learn"]["constraints_CaT"]["softPConstraint"]
            self.constraints["useSoftPCurriculum"] = self.cfg["env"]["learn"]["constraints_CaT"]["useSoftPCurriculum"]
            self.constraints["curriculum"] = 0.0
            self.cstr_manager = ConstraintManager(tau=self.cfg["env"]["learn"]["constraints_CaT"]["tauConstraint"], 
                                                  min_p=self.cfg["env"]["learn"]["constraints_CaT"]["minPConstraint"])
            self.numConstraints = -1
        else:
            self.numConstraints = 0

        # Values of constraints limits
        self.limits = self.cfg["env"]["learn"]["limits"]

        # Scales for velocity tracking commands
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]
        if self.cfg["env"]["onlyForwards"]:
            # Just going forwards at maximum velocity
            self.command_x_range[0] = self.cfg["env"]["onlyForwardsVelocity"]
            self.command_x_range[1] = self.cfg["env"]["onlyForwardsVelocity"]
            for i in range(2):
                self.command_y_range[i] = 0.
                self.command_yaw_range[i] = 0.

        # Initial state of the robot base
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        self.base_init_state = pos + rot + v_lin + v_ang

        # Other miscellaneous quantities
        self.capturing_video = virtual_screen_capture
        self.height_samples = None
        self.custom_origins = False
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]
        self.decimation = self.cfg["env"]["control"]["decimation"]
        self.dt = self.decimation * self.cfg["sim"]["dt"]
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"] 
        self.max_episode_length = int(self.max_episode_length_s/ self.dt + 0.5)
        self.randomize_friction = self.cfg["env"]["learn"]["randomizeFriction"]
        self.push_enable = self.cfg["env"]["learn"]["pushRobots"]
        self.push_prob = self.dt/self.cfg["env"]["learn"]["pushInterval_s"]
        self.allow_knee_contacts = self.cfg["env"]["learn"]["allowKneeContacts"]
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]
        self.curriculum = self.cfg["env"]["terrain"]["curriculum"]
        self.base_height_target = self.cfg["env"]["learn"]["base_height_target"]
        self.phases_freq = self.cfg["env"]["learn"]["phases_freq"] 
        self.gait_period = self.cfg["env"]["learn"]["gait_period"]
        self.vel_deadzone = self.cfg["env"]["learn"]["vel_deadzone"]
        self.flat_terrain_threshold = self.cfg["env"]["learn"]["flatTerrainThreshold"]
        
        # Compute the dimension of the observations
        self.num_height_points = len(self.cfg["env"]["learn"]["measured_points_x"]) * len(self.cfg["env"]["learn"]["measured_points_y"])
        self.prepare_dim_obs_functions()
        self.sampleObsSize = self.get_dim_observations()
        self.numHistorySamples = self.cfg["env"]["numHistorySamples"]
        self.numHistoryStep = self.cfg["env"]["numHistoryStep"]

        # Save relevant quantities for other parts of the pipeline
        self.cfg["env"]["sampleObsSize"] = self.sampleObsSize
        self.cfg["env"]["numObservations"] = self.sampleObsSize * self.numHistorySamples
        sizeObs = self.sampleObsSize * self.numHistorySamples
        sizeObsHist = self.sampleObsSize * (1 + (self.numHistorySamples - 1) * self.numHistoryStep)

        # Option to scale the rewards by the time step
        # for key in self.rew_scales.keys():
        #    self.rew_scales[key] *= self.dt

        # Initialization of the parent class VecTask
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id,
                         headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # Reapply time step value because it gets overwritten in VecTask
        self.dt = self.decimation * self.cfg["sim"]["dt"]

        # Backward compatibility in gym wrapper
        self.metadata["video.frames_per_second"] = int(np.round(1/(self.dt)))
        self.metadata["video.output_frames_per_second"] = 30

        # self.gym.simulate will be called in pre_physics_step with the decimation mechanism
        # Setting this to 0 avoid an additional call in VecTask.step 
        self.control_freq_inv = 0

        # Prepare the height scan
        self.height_points = self.prepare_height_points()
        self.measured_heights = torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)

        # Prepare observations, i.e gather functions that will compute observations
        # and observation noises
        self.prepare_observation_functions()
        self.prepare_noise_functions()

        # Initialization of the camera position if there is a graphical interface
        if self.graphics_device_id != -1:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # Get gym GPU state tensors (i.e direct memory wrapping of the simulator)
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)  # for imu force sensor

        # Create some wrapper tensors for different slices
        # These tensors will be updated when the associated self.gym.refresh_xxx is called
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        self.force_sensor = gymtorch.wrap_tensor(force_sensor_tensor)

        # Taking only the [0:3] components breaks the automatic update like for previous tensors
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        # TODO: Find a way to have these two keep the view like dof_pos and dof_vel to avoid having to update them manually

        # Initialize some data and tensors used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self.get_noise_scale_vec(self.cfg)
        self.commands = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale], device=self.device, requires_grad=False,)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.phases = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.contacts_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.contacts_last = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.contacts_touchdown = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.feet_gait_time = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_swing_time = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_swing_apex = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_clearance = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_clearance_cstr = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.obs_buf = torch.zeros((self.num_envs, sizeObs), dtype=torch.float, device=self.device, requires_grad=False)
        self.hist_obs_buf = torch.zeros((self.num_envs, sizeObsHist), dtype=torch.float, device=self.device, requires_grad=False)
        self.base_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device, requires_grad=False)
        self.base_quat = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device, requires_grad=False)
        self.base_quat[:, 3] = 1.0
        self.filtered_contact_forces = torch.zeros((self.num_envs, 4, 3, 5), dtype=torch.float, device=self.device, requires_grad=False)
        self.move_up_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.camera_pos = np.zeros(3)
        self.ceilings = torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        # Keeping the 6 last actions and joint states
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, 6, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_pos = torch.zeros(self.num_envs, self.num_dof, 6, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros(self.num_envs, self.num_dof, 6, dtype=torch.float, device=self.device, requires_grad=False)

        # Prepare depth cameras
        self.use_depth = self.cfg["env"]["depth"]["use_depth"]
        if self.use_depth:
            assert self.cfg["env"]["enableCameraSensors"]
            self.depth_clip_min = self.cfg["env"]["depth"]["depth_clip_min"]
            self.depth_clip_max = self.cfg["env"]["depth"]["depth_clip_max"]
            self.depth_update_interval = self.cfg["env"]["depth"]["update_interval"]
            self.depths = torch.zeros(
                    (self.num_envs, self.cfg["env"]["depth"]["image_size"][0], self.cfg["env"]["depth"]["image_size"][1]),
                    dtype=torch.float, device=self.device, requires_grad=False
            )

        # Default joint positions to which the joint position offsets (actions) are added
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_actions):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

        # Logging rewards over the whole episodes (cumulative sum)
        torch_zeros = lambda : torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {"lin_vel": torch_zeros()}

        # Reset all environments once to prepare them for the first action
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Initialisation is done
        self.init_done = True

    def create_sim(self):
        """Create the simulation terrain and the environments (i.e. spawn the robots)."""

        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        # Either load a huge flat ground ("plane") or a complex trimesh
        terrain_type = self.cfg["env"]["terrain"]["terrainType"] 
        if terrain_type=='plane':
            self._create_ground_plane()
        elif terrain_type=='trimesh':
            self._create_trimesh()
            self.custom_origins = True

        # Initialize the environments
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        """Load a huge flat plane in the simulation."""

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg["env"]["terrain"]["staticFriction"]
        plane_params.dynamic_friction = self.cfg["env"]["terrain"]["dynamicFriction"]
        plane_params.restitution = self.cfg["env"]["terrain"]["restitution"]
        self.gym.add_ground(self.sim, plane_params)

    def _create_trimesh(self):
        """Load a complex trimesh in the simulation, created with the Terrain class."""

        # Create the terrain mesh
        self.terrain = Terrain(self.cfg["env"]["terrain"], num_robots=self.num_envs)
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = -self.terrain.border_size 
        tm_params.transform.p.y = -self.terrain.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg["env"]["terrain"]["staticFriction"]
        tm_params.dynamic_friction = self.cfg["env"]["terrain"]["dynamicFriction"]
        tm_params.restitution = self.cfg["env"]["terrain"]["restitution"]

        # Load the mesh into the simulation and save the whole terrain heightmap for easy access
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self, num_envs, spacing, num_per_row):
        """Initalize the environments by spawning one robot (the actor) for each env."""

        # Getting the asset file
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
        asset_file = self.cfg["env"]["urdfAsset"]["file"]
        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        # Default asset options with joint torque control mode
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.collapse_fixed_joints = self.cfg["env"]["urdfAsset"]["collapseFixedJoints"]
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False

        # Loading Solo-12 asset with default properties
        self.solo_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(self.solo_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.solo_asset)

        # Prepare friction randomization
        rigid_shape_prop = self.gym.get_asset_rigid_shape_properties(self.solo_asset)
        friction_range = self.cfg["env"]["learn"]["frictionRange"]
        num_buckets = 100
        friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device=self.device)

        # Prepare default base position
        self.base_init_state = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        # Loading body, DoF, knee, shin and feet names
        body_names = self.gym.get_asset_rigid_body_names(self.solo_asset)
        self.dof_names = self.gym.get_asset_dof_names(self.solo_asset)
        foot_name = self.cfg["env"]["urdfAsset"]["footName"]
        shin_name = self.cfg["env"]["urdfAsset"]["shinName"]
        knee_name = self.cfg["env"]["urdfAsset"]["kneeName"]
        feet_names = [s for s in body_names if foot_name in s]
        shin_names = [s for s in body_names if shin_name in s]
        knee_names = [s for s in body_names if knee_name in s]
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.grf_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.shin_indices = torch.zeros(len(shin_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.base_index = 0

        # Apply the armature value to all joints (i.e. take into account the motor inertia seen at joint level)
        dof_props = self.gym.get_asset_dof_properties(self.solo_asset)
        dof_props["armature"].fill(self.cfg["env"]["urdfAsset"]["armature"])

        # Add a pseudo-Inertia Measurement Unit sensor through the use of a force sensor on the base
        # imu_pose = gymapi.Transform()
        # self.gym.create_asset_force_sensor(self.solo_asset, self.base_index, imu_pose)

        # Gather env origins and spread the robots over the whole terrain
        # The terrain is divided into rows (levels) and columns (types), each cell being a potential spawn location
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        if not self.curriculum: self.cfg["env"]["terrain"]["maxInitMapLevel"] = self.cfg["env"]["terrain"]["numLevels"] - 1
        if self.cfg["env"]["terrain"]["maxInitMapLevel"] < self.cfg["env"]["terrain"]["minInitMapLevel"]:
            self.cfg["env"]["terrain"]["maxInitMapLevel"] = self.cfg["env"]["terrain"]["minInitMapLevel"]
        self.terrain_levels = torch.randint(self.cfg["env"]["terrain"]["minInitMapLevel"], self.cfg["env"]["terrain"]["maxInitMapLevel"]+1, (self.num_envs,), device=self.device)
        #self.terrain_types = torch.randint(0, self.cfg["env"]["terrain"]["numTerrains"], (self.num_envs,), device=self.device)
        self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg["env"]["terrain"]["numTerrains"]), rounding_mode='floor').to(torch.long)
        if self.custom_origins:
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            spacing = 0.
        if self.cfg["test"] and self.cfg["env"]["startAtLevel"] != -1:
            # Force all robots to spawn at the same level
            self.terrain_levels[:] = self.cfg["env"]["startAtLevel"]
        self.terrain_ceilings = torch.from_numpy(self.terrain.ceilings).to(self.device).to(torch.float)

        # Create the env instances and save the handles to interact with them
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.solo_handles = []
        self.envs = []
        self.cam_handles = []
        for i in range(self.num_envs):
            # Create one environment
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            if self.custom_origins:
                self.env_origins[i] = self.terrain_origins[self.terrain_levels[i], self.terrain_types[i]]
                pos = self.env_origins[i].clone()
                pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
                start_pose.p = gymapi.Vec3(*pos)

            # Create one actor (robot) in that environment
            if self.randomize_friction:
                for s in range(len(rigid_shape_prop)):
                    rigid_shape_prop[s].friction = friction_buckets[i % num_buckets]
            self.gym.set_asset_rigid_shape_properties(self.solo_asset, rigid_shape_prop)
            solo_handle = self.gym.create_actor(env_handle, self.solo_asset, start_pose, "solo", i, 0, 0)
            self.gym.set_actor_dof_properties(env_handle, solo_handle, dof_props)
            self.envs.append(env_handle)
            self.solo_handles.append(solo_handle)
            self.attach_camera(i, env_handle, solo_handle)

        # Gather base, knee, shin and feet indices based on their name
        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.solo_handles[0], self.cfg["env"]["urdfAsset"]["baseName"])
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.solo_handles[0], knee_names[i])
        if not self.cfg["env"]["urdfAsset"]["collapseFixedJoints"]:
            # If collapseFixedJoints is True, then the feet are collapsed into the shin and "shin + feet" become feet
            for i in range(len(shin_names)):
                self.shin_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.solo_handles[0], shin_names[i])
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.solo_handles[0], feet_names[i])

        # If feet are collapsed, use shin for ground reaction forces, otherwise use the feet
        if False: # self.cfg["env"]["urdfAsset"]["collapseFixedJoints"]:
            for i in range(len(shin_names)):
                self.grf_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.solo_handles[0], shin_names[i])
        else:
            for i in range(len(feet_names)):
                self.grf_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.solo_handles[0], feet_names[i])
        # TODO: Double-check which one to use

        # Retrieve base mass from body properties
        body_props = self.gym.get_actor_rigid_body_properties(self.envs[0], self.solo_handles[0])
        self.base_mass = body_props[self.base_index].mass

    ####################
    # Observations
    ####################

    def prepare_dim_obs_functions(self):
        """Prepares a list of functions to compute the dimension of the obs vector
        Looks for self.dim_obs_<OBSERVATION_NAME>, where <OBSERVATION_NAME> are names of all True
        observations in observe dict in the cfg.
        """

        # Prepare list of functions
        self.dim_obs_functions = []
        for name in self.cfg["env"]["learn"]["observe"].keys():
            if self.cfg["env"]["learn"]["observe"][name]:
                name = "dim_obs_" + name
                self.dim_obs_functions.append(getattr(self, name))

    def prepare_observation_functions(self):
        """Prepares a list of observation functions, which will be called to compute the total obs vector
        Looks for self.observe_<OBSERVATION_NAME>, where <OBSERVATION_NAME> are names of all True
        observations in observe dict in the cfg.
        """

        # Prepare list of functions
        self.observation_functions = []
        self.observation_names = []
        for name in self.cfg["env"]["learn"]["observe"].keys():
            if self.cfg["env"]["learn"]["observe"][name]:
                self.observation_names.append(name)
                name = "observe_" + name
                self.observation_functions.append(getattr(self, name))

    def prepare_noise_functions(self):
        """Prepares a list of noise functions, which will be called to noise the observations
        Looks for self.noise_<OBSERVATION_NAME>, where <OBSERVATION_NAME> are names of all True
        observations in observe dict in the cfg.
        """

        # Prepare list of functions
        self.noise_functions = []
        for name in self.cfg["env"]["learn"]["observe"].keys():
            if self.cfg["env"]["learn"]["observe"][name]:
                name = "noise_" + name
                self.noise_functions.append(getattr(self, name))

    def get_dim_observations(self):
        """Return the size of the observation vector."""

        num = 0
        for dim_obs_function in self.dim_obs_functions:
            num += dim_obs_function()
        return num

    def get_noise_scale_vec(self, cfg):
        """Return the scale of the noise for each component of the observation."""

        self.add_noise = self.cfg["env"]["learn"]["addNoise"]

        # Initialize empty noise vec then stack noises
        noise_vec = torch.zeros(
            0, dtype=torch.float, device=self.device, requires_grad=False
        )
        for noise_function in self.noise_functions:
            noise_vec = torch.cat(
                (
                    noise_vec,
                    noise_function(),
                ),
                dim=-1,
            )

        noise_vec *= self.cfg["env"]["learn"]["noiseLevel"]

        return noise_vec

    def compute_observations(self):
        """Compute observations, apply observation noises, handle observation history and fill observation buffer accordingly."""

        # Update height scan (environment could have been reset between now and previous update before the rewards)
        self.measured_heights = self.get_heights()

        # Initialize empty obs buffer then stack observations
        obs_meas = torch.zeros(
            0, dtype=torch.float, device=self.device, requires_grad=False
        )
        for observation_function in self.observation_functions:
            obs_meas = torch.cat(
                (
                    obs_meas,
                    observation_function(),                
                ),
                dim=-1,
            )

        # Add noise to observation sample
        if self.add_noise:
            obs_meas += (2 * torch.rand_like(obs_meas) - 1) * self.noise_scale_vec

        # Refresh history of observation for envs that have just been reset
        resetted = (self.progress_buf == 1)
        if torch.any(resetted):
            # No need to refresh last one (will be discarded just after)
            for i in range(0, (self.numHistorySamples - 1) * self.numHistoryStep):
                self.hist_obs_buf[resetted, i * self.sampleObsSize : (i+1) * self.sampleObsSize] = obs_meas[resetted, :]

        # Include new obs sample at the start of the history and discard oldest one
        self.hist_obs_buf = torch.cat((obs_meas, self.hist_obs_buf[:, :-self.sampleObsSize]), dim = -1)

        # Fill observation buffer with numHistorySamples samples, selected every numHistoryStep sample
        # Like if numHistorySamples = 3 and numHistoryStep = 2
        # then history contains [t-1, t-2, t-3, t-4, t-5] and we select [t-1, t-3, t-5]
        for i in range(self.numHistorySamples):
            j = i * self.numHistoryStep * self.sampleObsSize
            self.obs_buf[:, i * self.sampleObsSize : (i+1) * self.sampleObsSize] = self.hist_obs_buf[:, j:(j + self.sampleObsSize)]

    # ------------ dim obs functions ----------------
    def dim_obs_base_lin_vel(self):
        """Dimension of base linear velocity observations."""
        return 3

    def dim_obs_base_ang_vel(self):
        """Dimension of base linear angular observations."""
        return 3

    def dim_obs_commands(self):
        """Dimension of base commands observations."""
        return 3

    def dim_obs_misc(self):
        """Dimension of miscellaneous observations."""
        return 39 # 123

    def dim_obs_heights(self):
        """Dimension of height scan observations."""
        return self.num_height_points

    def dim_obs_phases(self):
        """Dimension of phases observations."""
        return 8

    def dim_obs_imu(self):
        """Dimension of IMU observations."""
        return 3

    def dim_obs_ceilings(self):
        """Dimension of ceilings observations."""
        return 1

    # ------------ obs functions ----------------
    def observe_base_lin_vel(self):
        """Observe base linear velocity."""
        return self.base_lin_vel * self.lin_vel_scale

    def observe_base_ang_vel(self):
        """Observe base angular velocity."""
        return self.base_ang_vel * self.ang_vel_scale

    def observe_commands(self):
        """Observe base commands."""
        #return self.commands[:, :3] * self.commands_scale
        #base_quat = self.root_states[:, 3:7]
        #absolute_commands = torch.zeros((4096,3)).to('cuda')
        #absolute_commands[:, :2] = self.commands[:, :2]
        #relative_commands = quat_rotate_inverse(base_quat, absolute_commands)
        ##relative_commands[:, 2] = self.commands[:, 2]
        #return relative_commands

        base_quat = self.root_states[:, 3:7]
        _, _, yaw = get_euler_xyz(base_quat)
        zero = torch.zeros_like(yaw)
        yaw_quat = quat_from_euler_xyz(zero, zero, yaw)
        command_3d = torch.zeros((self.num_envs,3)).to('cuda')
        command_3d[:, :2] = self.commands[:, :2]
        robot_command = quat_rotate_inverse(yaw_quat, command_3d)
        robot_command[:, 2] = self.commands[:, 2]
        return robot_command * self.commands_scale

    def observe_misc(self):
        """Observe misc quantities.
        Use discrete joint velocities to avoid unreliable simulation reports.
        See https://forums.developer.nvidia.com/t/dof-velocity-offset-at-rest/205799/11
        """
        return torch.cat((self.projected_gravity,  # projected gravity, an image of the orientation (t)
                          self.dof_pos * self.dof_pos_scale, # joint positions (t)
                          ((self.dof_pos - self.last_dof_pos[:, :, 0]) / self.dt) * self.dof_vel_scale, # joint velocities (t)
                          self.actions, # joint position targets (t - 1)
                         ), dim=-1)

    def observe_heights(self):
        """Observe height of surrounding terrain."""
        return torch.clip(self.root_states[:, 2].unsqueeze(1) - self.base_height_target - self.measured_heights, -1, 1.) * self.height_meas_scale

    def observe_phases(self):
        """Observe pairs of gait phases."""
        return torch.cat((torch.cos(self.phases),
                          torch.sin(self.phases)
                         ), dim=-1)

    def observe_imu(self):
        """Observe imu acceleration through force readings on the base.
        force_sensor stores net forces, that are the sum of external forces, contact forces and internal forces.
        A body resting on the ground will have a net force of zero."""
        return self.force_sensor[:, :3] / self.base_mass * self.imu_scale

    def observe_ceilings(self):
        """Observe ceilings."""
        return self.ceilings.unsqueeze(1)

    # ------------ noise obs functions ----------------
    def noise_base_lin_vel(self):
        """Noise for base linear velocity."""
        return self.cfg["env"]["learn"]["linearVelocityNoise"] * self.lin_vel_scale * torch.ones(self.dim_obs_base_lin_vel(), dtype=torch.float, device=self.device, requires_grad=False)

    def noise_base_ang_vel(self):
        """Noise for base angular velocity."""
        return self.cfg["env"]["learn"]["angularVelocityNoise"] * self.ang_vel_scale * torch.ones(self.dim_obs_base_ang_vel(), dtype=torch.float, device=self.device, requires_grad=False)

    def noise_commands(self):
        """Noise for base commands."""
        return torch.zeros(self.dim_obs_commands(), dtype=torch.float, device=self.device, requires_grad=False)

    def noise_misc(self):
        """Noise for misc quantities."""
        return torch.cat((self.cfg["env"]["learn"]["gravityNoise"] * torch.ones(3, dtype=torch.float, device=self.device, requires_grad=False),
                          self.cfg["env"]["learn"]["dofPositionNoise"] * self.dof_pos_scale * torch.ones(12, dtype=torch.float, device=self.device, requires_grad=False),
                          self.cfg["env"]["learn"]["dofVelocityNoise"] * self.dof_vel_scale * torch.ones(12, dtype=torch.float, device=self.device, requires_grad=False),
                          torch.zeros(12, dtype=torch.float, device=self.device, requires_grad=False),
                          ), dim=-1)
        return torch.cat((self.cfg["env"]["learn"]["gravityNoise"] * torch.ones(3, dtype=torch.float, device=self.device, requires_grad=False),
                          self.cfg["env"]["learn"]["dofPositionNoise"] * self.dof_pos_scale * torch.ones(12, dtype=torch.float, device=self.device, requires_grad=False),
                          torch.zeros(36, dtype=torch.float, device=self.device, requires_grad=False),
                          self.cfg["env"]["learn"]["dofVelocityNoise"] * self.dof_vel_scale * torch.ones(12, dtype=torch.float, device=self.device, requires_grad=False),
                          torch.zeros(36 + 24, dtype=torch.float, device=self.device, requires_grad=False),
                          ), dim=-1)

    def noise_heights(self):
        """Noise for height of surrounding terrain."""
        return self.cfg["env"]["learn"]["heightMeasurementNoise"] * self.height_meas_scale * torch.ones(self.dim_obs_heights(), dtype=torch.float, device=self.device, requires_grad=False)

    def noise_phases(self):
        """Noise for pairs of gait phases."""
        return torch.zeros(self.dim_obs_phases(), dtype=torch.float, device=self.device, requires_grad=False)

    def noise_imu(self):
        """Noise for IMU readings."""
        return torch.zeros(self.dim_obs_imu(), dtype=torch.float, device=self.device, requires_grad=False)

    def noise_ceilings(self):
        """Noise for ceilings."""
        return torch.zeros(self.dim_obs_ceilings(), dtype=torch.float, device=self.device, requires_grad=False)

    ####################
    # Depth cameras
    ####################

    def attach_camera(self, i, env_handle, actor_handle):
        if self.cfg["env"]["depth"]["use_depth"]:
            camera_props = gymapi.CameraProperties()
            camera_props.width = self.cfg["env"]["depth"]["image_size"][1]
            camera_props.height = self.cfg["env"]["depth"]["image_size"][0]
            camera_props.enable_tensors = True
            camera_props.horizontal_fov = self.cfg["env"]["depth"]["horizontal_fov"]

            camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
            assert camera_handle != -1, "The camera failed to be created"
            self.cam_handles.append(camera_handle)
            
            local_transform = gymapi.Transform()
            
            camera_position = np.copy(self.cfg["env"]["depth"]["position"])
            camera_angle = np.random.uniform(self.cfg["env"]["depth"]["angle"][0], self.cfg["env"]["depth"]["angle"][1])

            local_transform.p = gymapi.Vec3(*camera_position)
            local_transform.r = gymapi.Quat.from_euler_zyx(0, np.radians(camera_angle), 0)
            root_handle = self.gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)
            
            self.gym.attach_camera_to_body(camera_handle, env_handle, root_handle, local_transform, gymapi.FOLLOW_TRANSFORM)

    def update_depth_buffer(self):
        if not self.use_depth:
            return

        if self.common_step_counter % self.depth_update_interval != 0:
            return

        self.gym.step_graphics(self.sim) # required to render in headless mode
        self.gym.fetch_results(self.sim, True)

        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        for i in range(self.num_envs):
            depth_image_ = self.gym.get_camera_image_gpu_tensor(self.sim, 
                                                                self.envs[i], 
                                                                self.cam_handles[i],
                                                                gymapi.IMAGE_DEPTH)
            
            depth_image = gymtorch.wrap_tensor(depth_image_)
            depth_image = (torch.clip(-depth_image, min=self.depth_clip_min, max=self.depth_clip_max) - self.depth_clip_min) / (self.depth_clip_max - self.depth_clip_min)
            self.depths[i] = depth_image

        self.gym.end_access_image_tensors(self.sim)
        self.extras["depth"] = self.depths

    ####################
    # Rewards
    ####################

    def compute_reward(self):
        """Compute a limited set of rewards for constraints as terminations."""

        command = torch.norm(self.commands[:, :2], dim=1)

        # Velocity reward
        lin_vel = self.root_states[:, 7:9]
        no_move=  (torch.norm(self.commands[:, :2], dim=1) < self.vel_deadzone).float().unsqueeze(1)
        direction = no_move*torch.nn.functional.normalize(-lin_vel) + (1-no_move)*torch.nn.functional.normalize(self.commands[:, :2])
        rew_vel = (direction * lin_vel).sum(1)

        # Survival bonus
        rew_survival = self.rew_scales["survivalBonus"]

        self.rew_buf = rew_vel + rew_survival
        self.rew_buf = torch.clip(self.rew_buf, min=0., max=None)

        # Saving the cumulative sum of rewards over the episodes
        self.episode_sums["lin_vel"] += rew_vel

    ####################
    # CaT Constraints
    ####################

    def compute_constraints_cat(self):
        """Compute various constraints for constraints as terminations. Constraints violations are asssessed then
        handed out to a constraint manager (ConstraintManager class) that will compute termination probabilities."""

        # heading constraint
        base_quat = self.root_states[:, 3:7]
        yaw_quat = 2 * torch.atan2(base_quat[..., 2], base_quat[..., 3])
        command_norm = torch.nn.functional.normalize(self.commands[:, :2], dim=-1)
        yaw_command = torch.atan2(command_norm[..., 1], command_norm[..., 0])
        yaw_diff = torch.atan2(torch.sin(yaw_quat - yaw_command), torch.cos(yaw_quat - yaw_command))
        angle = torch.abs(yaw_diff)

        cstr_heading = (angle-self.limits["heading"]).clamp(max=1.5)* (torch.norm(self.base_lin_vel[:, :2], dim=1) > 0.1).float()

        # heading hard constraint (to avoid rapid spinning)
        cstr_heading_hard = (angle-1.5) * (torch.norm(self.base_lin_vel[:, :2], dim=1) > 0.1).float()

        # ------------ Soft constraints ----------------

        # Torque constraint
        cstr_torque = torch.abs(self.torques) - self.limits["torque"]

        # Joint velocity constraint
        cstr_joint_vel = torch.abs(self.dof_vel) - self.limits["vel"]

        # Joint acceleration constraint
        cstr_joint_acc = torch.abs(self.last_dof_vel[:, :, 0] - self.dof_vel) / self.dt - self.limits["acc"]

        # Action rate constraint (for command smoothness)
        cstr_action_rate = torch.abs(self.actions - self.last_actions[:, :, 0]) / self.dt - self.limits["action_rate"]

        # ------------ Hard constraints ----------------

        # Knee contact constraint
        cstr_knee_contact = torch.norm(self.contact_forces[:, self.knee_indices, :], dim=2)

        # Base contact constraint
        cstr_base_contact = torch.norm(self.contact_forces[:, self.base_index, :], dim=1)

        # Foot contact force constraint
        cstr_foot_contact = torch.norm(self.contact_forces[:, self.grf_indices], dim=2) - self.limits["foot_contact_force"]

        # Stumbling constraint
        cstr_foot_stumble = torch.norm(self.contact_forces[:, self.grf_indices, :2], dim=2) - 4.0 * torch.abs(self.contact_forces[:, self.grf_indices, 2])

        # Constraint on the two front HFE joints
        cstr_HFE = self.dof_pos[:, [1, 4]] - self.limits["HFE"]
        cstr_HFE_min = self.limits["HFE_min"] - self.dof_pos[:, [1, 4]]
        cstr_HFE_style = self.dof_pos[:, [1, 4, 7, 10]] - 1.3

        # Constraint on the two back HFE joints
        cstr_H_HFE = self.dof_pos[:, [7, 10]] - 1.9

        # Constraint on the KFE (knee) joints
        cstr_KFE = self.dof_pos[:, [2, 5, 8, 11]]
        cstr_KFE_min = -2.8 - self.dof_pos[:, [2, 5, 8, 11]]

        # Hip constraint (style constraint on HAA joint)
        cstr_HAA = torch.abs(self.dof_pos[:, [0, 3, 6, 9]] - self.default_dof_pos[:, [0, 3, 6, 9]]) - self.limits["HAA"]

        # Constraint to avoid having the robot upside-down
        cstr_upsidedown = self.projected_gravity[:, 2] > 0

        # Constraint to not fall outside of the tracj
        cstr_lava = self.root_states[:, 2] < -0.05

        cstr_minbaseheight = (self.limits["min_base_height"] - self.root_states[:, 2]) * (self.ceilings >= 0.34).float()

        cstr_foot_stumble = torch.norm(self.contact_forces[:, self.grf_indices, :2], dim=2) - 4.0*torch.abs(self.contact_forces[:, self.grf_indices, 2])

        # ------------ Style constraints ----------------


        # Base orientation constraint (style constraint on roll/pitch angles)
        cstr_base_orientation = torch.norm(self.projected_gravity[:, :2], dim=1) - self.limits["base_orientation"]
        cstr_base_orientation_x = torch.abs(self.projected_gravity[:, 1]) - self.limits["base_orientation"]

        # Air time constraint (style constraint)
        cstr_air_time = (self.constraints["air_time"] - self.feet_swing_time) * self.contacts_touchdown * (torch.norm(self.commands[:, :2], dim=1) > self.vel_deadzone).float().unsqueeze(1)

        # Constraint to stand still when the velocity command is 0 (style constraint)
        cstr_nomove = (torch.abs(self.dof_vel) - 2.0) *(torch.norm(self.commands[:, :2], dim=1) < self.vel_deadzone).float().unsqueeze(1)

        # Constraint to have exactly 2 feet in contact with the ground at any time when walking (style constraint)
        cstr_2footcontact = torch.abs((self.contact_forces[:, self.grf_indices, 2] > 1.0).sum(1) - 2.0)

        # Apply aesthetics constraints only on flat terrains
        self.is_flat_terrain = (((self.measured_heights.var(1) < self.flat_terrain_threshold) & (self.ceilings >= 0.34)) | (self.terrain_levels <= 1)).float()

        cstr_base_orientation *= self.is_flat_terrain

        # We impose a certain walking style on flat surfaces to make the robot walk very nicely
        command_vel = torch.norm(self.commands[:, :2], dim=1)
        cstr_2footcontact *= ((self.measured_heights.var(1) < self.flat_terrain_threshold) & (self.ceilings >= 0.34)).float()
        cstr_2footcontact *= (command_vel > self.vel_deadzone).float()

        cstr_HFE_style *= ((self.ceilings >= 0.34) & (self.terrain_levels <= 4)).unsqueeze(1).float()
        cstr_nomove *= ((self.measured_heights.var(1) < self.flat_terrain_threshold) & (self.ceilings >= 0.34)).float().unsqueeze(1)

        # ------------ Applying constraints ----------------

        # Maximum termination probability for soft and style constraints
        soft_p = self.constraints["soft_p"]
        m_soft_p = 0.1 + soft_p # a10

        sqrt_func = lambda x: x.clamp(min=0.0).sqrt()

        # Curriculum on soft_p value (optionnal)
        # Soft and style constraints are initially less enforced to let the policy explore more.
        if self.constraints["useSoftPCurriculum"]:
            step_cur = 1.0 / (self.cfg["horizon_length"] * self.cfg["max_epochs"])
            self.constraints["curriculum"] = min(self.constraints["curriculum"] + step_cur, 1.0)

            # Linearly interpolate the expected time for episode end: soft_p is the maximum
            # termination probability so it is an image of the expected time of death.
            T_start = 25
            T_end = 1 / soft_p
            soft_p = 1 / (T_start + self.constraints["curriculum"] * (T_end - T_start))

        # Soft constraints
        self.cstr_manager.add("heading", sqrt_func(cstr_heading), max_p=m_soft_p)
        self.cstr_manager.add("heading_hard", sqrt_func(cstr_heading_hard), max_p=1.0)
        self.cstr_manager.add("stumble", sqrt_func(cstr_foot_stumble), max_p=m_soft_p)
        self.cstr_manager.add("torque", sqrt_func(cstr_torque), max_p=soft_p)
        self.cstr_manager.add("joint_acc", sqrt_func(cstr_joint_acc), max_p=soft_p)
        self.cstr_manager.add("joint_vel",  sqrt_func(cstr_joint_vel), max_p=soft_p)
        self.cstr_manager.add("action_rate", sqrt_func(cstr_action_rate), max_p=soft_p)

        # Hard constraints
        self.cstr_manager.add("knee_contact", sqrt_func(cstr_knee_contact), max_p=1.0)
        self.cstr_manager.add("base_contact", sqrt_func(cstr_base_contact), max_p=1.0)
        self.cstr_manager.add("foot_contact", sqrt_func(cstr_foot_contact), max_p=1.0)
        self.cstr_manager.add("upsidedown", cstr_upsidedown, max_p=1.0)
        self.cstr_manager.add("lava", cstr_lava, max_p=1.0)
        self.cstr_manager.add("min_base_height", sqrt_func(cstr_minbaseheight), max_p=1.0)

        # Joint constraints
        soft_p_joint = soft_p 
        self.cstr_manager.add("HFE", sqrt_func(cstr_HFE), max_p=soft_p_joint)
        self.cstr_manager.add("HFE_min", sqrt_func(cstr_HFE_min), max_p=soft_p_joint)
        self.cstr_manager.add("HFE_style", sqrt_func(cstr_HFE_style), max_p=soft_p_joint)
        self.cstr_manager.add("KFE", sqrt_func(cstr_KFE), max_p=soft_p_joint)
        self.cstr_manager.add("KFE_min", sqrt_func(cstr_KFE_min), max_p=soft_p_joint)
        self.cstr_manager.add("H_HFE", sqrt_func(cstr_H_HFE), max_p=soft_p_joint)
        self.cstr_manager.add("HAA", sqrt_func(cstr_HAA), max_p=soft_p_joint)

        # Style constraints
        self.cstr_manager.add("base_ori", sqrt_func(cstr_base_orientation), max_p=soft_p)
        self.cstr_manager.add("base_ori_x", sqrt_func(cstr_base_orientation_x), max_p=m_soft_p)

        self.cstr_manager.add("air_time", cstr_air_time, max_p=soft_p)
        self.cstr_manager.add("no_move", sqrt_func(cstr_nomove), max_p=soft_p)
        self.cstr_manager.add("2footcontact", cstr_2footcontact, max_p=soft_p)

        self.cstr_manager.log_all(self.episode_sums)

        # Timeout if episodes have reached their maximum duration
        timeout = self.progress_buf >= self.max_episode_length - 1

        # Get final termination probability for each env from all constraints
        self.cstr_prob = self.cstr_manager.get_probs()

        # Probability of termination used to affect the discounted sum of rewards
        self.reset_buf = self.cstr_prob

        self.reset_env_buf = timeout | cstr_upsidedown | cstr_lava

        # Store quantities for off-policy CaT
        self.extras["true_dones"] = self.reset_env_buf
        self.extras["truncateds"] = timeout
        self.extras["raw_constraints"] = self.cstr_manager.get_raw_constraints()

    ####################
    # Other
    ####################

    def reset_idx(self, env_ids):
        """Reset environements when episodes have terminated."""

        # Randomize initial joint positions and velocities, as well as (x, y) position and yaw orientation of the base
        positions_offset = torch_rand_float(0.95, 1.05, (len(env_ids), self.num_dof), device=self.device)  # Multiplicative factor
        velocities = torch_rand_float(-0.05, 0.05, (len(env_ids), self.num_dof), device=self.device)
        yaw_offset = torch_rand_float(-1.57, 1.57, (len(env_ids), 1), device=self.device)  # Already divided by 2 for next line
        yaw_offset = torch_rand_float(-0.001, 0.001, (len(env_ids), 1), device=self.device)  # Already divided by 2 for next line
        xy_offset = torch_rand_float(-0.05, 0.05, (len(env_ids), 2), device=self.device)
        quat_offset = torch.cat((torch.zeros((len(env_ids), 2), device=self.device), torch.sin(yaw_offset), torch.cos(yaw_offset)), dim=1)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
        self.last_dof_pos[env_ids] = torch.unsqueeze(self.dof_pos[env_ids], 2)
        self.dof_vel[env_ids] = velocities

        # Set new root states
        if self.custom_origins:
            self.update_terrain_level(env_ids)
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += xy_offset
            self.root_states[env_ids, 3:7] = quat_offset
        else:
            self.root_states[env_ids] = self.base_init_state

        # Apply new root states to simulation
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        # Apply new joint states to simulation
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # Refresh base tensors
        self.base_pos[env_ids] = self.root_states[env_ids, :3]
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]

        # Refresh base commands
        self.resample_commands(env_ids)

        # Log mean value of rewards over the terminated episodes
        if self.common_step_counter > 0 and self.numRewards > 0:
            self.rew_cum_reset[env_ids] = self.rew_mean[env_ids]
            self.rew_mean_reset[env_ids] = self.rew_mean[env_ids] / torch.maximum(self.progress_buf[env_ids].unsqueeze(0).transpose(0, 1), torch.ones((len(env_ids), 1),  dtype=torch.long, device=self.device))
            self.rew_mean[env_ids] = 0.0

        # Log mean value of constraints over the terminated episodes
        if self.common_step_counter > 0 and self.numConstraints > 0 and self.useConstraints in ["cat"]:
            self.cstr_mean_reset[env_ids] = self.cstr_mean[env_ids] / torch.maximum(self.progress_buf[env_ids].unsqueeze(0).transpose(0, 1), torch.ones((len(env_ids), 1),  dtype=torch.long, device=self.device))
            """s = ""
            v = ""
            for i, name in enumerate(self.cstr_names):
                s += "{:6s} | {:6.3f}\n".format(name, torch.mean(self.cstr_mean_reset[:, i]).item())
            print(s)"""
            self.cstr_mean[env_ids] = 0.0

        # Reset a bunch of state tensors
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.contacts_last[env_ids] = False
        self.contacts_filt[env_ids] = False
        self.contacts_touchdown[env_ids] = False
        self.feet_swing_time[env_ids] = 0.
        self.feet_swing_apex[env_ids] = 0.
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # Update ceiling
        self.ceilings[env_ids] = self.terrain_ceilings[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

        # Fill extras and reset the cumulated reward sums of terminated episodes
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            akey = key if key.startswith("cstr_") else 'rew_' + key
            self.extras["episode"][akey] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.

        self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        self.extras["episode"]["terrain_level_max"] = torch.max(self.terrain_levels.float())
        self.extras["episode"]["gait_period"] = torch.max(self.feet_gait_time, dim=1).values
        self.extras["episode"]["foot_clearance"] = torch.max(self.feet_clearance, dim=1).values

    def resample_commands(self, env_ids):
        """Resample the base commands."""
        self.commands[env_ids, 0] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands[env_ids, 1] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze() * self.commands[env_ids, 0]
        self.commands[env_ids, :2] /= torch.norm(self.commands[env_ids, :2], dim=1).clamp(min=1.0).unsqueeze(-1) # Make sure that the command is inside the unit sphere

        self.commands[env_ids, 2] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands[env_ids] *= (torch.any(torch.abs(self.commands[env_ids, :3]) > self.vel_deadzone, dim=1)).unsqueeze(1)  # set small commands to zero


    def update_terrain_level(self, env_ids):
        """Update terrain level of robots as they progress to put them in increasingly difficult terrains."""
        if not self.init_done or not self.curriculum:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # If robots went far enough during their episode, move them up.
        # This is monitored by move_up_flag that is set to True during the episode.
        # If robots did not move enough with respect to their velocity command, move them down.
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5) * ~self.move_up_flag[env_ids] # <==========================replace 0.25 by 0.33
        move_up = distance > self.terrain.env_length * 0.8
        move_down = distance < self.terrain.env_length * 0.5 
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        #self.terrain_levels[env_ids] += 1 * self.move_up_flag[env_ids] - 1 * move_down
        #self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0) % self.terrain.env_rows
        self.terrain_levels[env_ids] = torch.where(
                self.terrain_levels[env_ids]>=self.terrain.env_rows,
                torch.randint_like(self.terrain_levels[env_ids], self.terrain.env_rows),
                torch.clip(self.terrain_levels[env_ids], 0)
        )

        # small probability of teleporting the robots back to the first level if no progress 
        self.terrain_levels[env_ids] *= ~(torch.bernoulli(0.01 * ~move_up).bool())

        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

        # Reset flags
        self.move_up_flag[env_ids] = False

    def process_contacts(self):
        """Perform various processing linked to the contact status of the feet."""

        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        # contacts = self.contact_forces[:, self.grf_indices, 2] > 1.0
        # self.contacts_filt = torch.logical_or(contacts, self.contacts_last)
        # self.contacts_last = contacts
        self.contacts_filt = self.contact_forces[:, self.grf_indices, 2] > 1.0  # Or do not filter
        
        # True when touchdown for a given foot
        self.contacts_touchdown = (self.feet_swing_time > 0) * self.contacts_filt

        # Update swing time and apex height
        self.feet_swing_time += self.dt
        self.feet_swing_apex = torch.max(self.feet_swing_apex, self.get_feet_heights())

        # Keep in memory clearance of last swing phase
        self.feet_clearance[self.contacts_touchdown] = self.feet_swing_apex[self.contacts_touchdown]

        # Contact touchdown that are more filtered to be sure we only detect long swing phases
        filt_contacts_touchdown = torch.logical_and(self.feet_swing_time > 0.04, self.contacts_filt)
        if torch.any(filt_contacts_touchdown):
            self.feet_gait_time[filt_contacts_touchdown] = self.feet_swing_time[filt_contacts_touchdown] * 2

    def push_robots(self):
        """Push the robots by assigning an instantaneous velocity to their base."""
        push_idx = torch.bernoulli(torch.full((self.num_envs, ), self.push_prob, device=self.device)).nonzero(as_tuple=False).flatten()
        self.root_states[push_idx, 7:13] += torch_rand_float(-0.5, 0.5, (len(push_idx), 6), device=self.device) # lin vel x/y

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def pre_physics_step(self, actions):
        """Computing command torques with the PD controller and running the simulation steps between each call of the policy."""
        
        self.actions = actions.clone().to(self.device)
        # If you want constant actions for debug purpose:
        # self.actions[:] = torch.tensor([0.2,  0.1679 , -0.37505, -0.2,  0.1679 , -0.37505, 0.2,  -0.1679 , 0.37505, -0.2,  -0.1679 , 0.37505])

        # There is self.decimation steps of simulation between each call to the policy
        for i in range(self.decimation):

            torques = torch.clip(
                (
                    self.Kp
                    * (
                        self.action_scale * self.actions
                        + self.default_dof_pos
                        - self.dof_pos
                    )
                    - self.Kd * self.dof_vel
                ),
                -10.0,  # Hard higher limit on torques
                10.0,  # Hard lower limit on torques
            )

            # Saturating command torques (on Solo we saturate the max currents)
            # torques = torch.clamp(torques, -3.5, 3.5)

            # Send desired joint torques to the simulation, run one step of simulator then refresh joint states
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
            self.torques = torques.view(self.torques.shape)
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)

            # Gathering feet contact forces over several steps for averaging purpose (to avoid simulation glitches)
            self.filtered_contact_forces[:, :, :, 1:] = self.filtered_contact_forces[:, :, :, :-1]
            self.filtered_contact_forces[:, :, :, 0] = self.contact_forces[:, self.grf_indices, :]         


        # Render the simulation (if there is a graphical interface)
        if self.force_render:
            self.render()

    def post_physics_step(self):
        """Updating tensors, computing rewards, constraints and observations after simulation steps."""

        # Updating all state tensors that are linked with the simulator
        # self.gym.refresh_dof_state_tensor(self.sim)  # done in pre_physics_step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)  # for imu force sensor

        # Increment counters
        self.progress_buf += 1  # Step counter for current episodes, set back to 0 upon env reset
        self.randomize_buf += 1  # Counter for domain randomization in VecTask (not really used)
        self.common_step_counter += 1  # Step counter since the start of the training/testing

        # Randomly push the robots
        if self.push_enable:
            self.push_robots()

        # Stop early if we are capturing a video
        if self.capturing_video and self.common_step_counter > 540:
            print("-- Stopping script early --")
            quit()

        # Update state quantities
        self.base_pos = self.root_states[:, :3]
        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # Update leg phases
        phases_off = torch.tensor([0.0, torch.pi, torch.pi, 0.0], device=self.device)  # Phase offset between the legs
        self.phases = torch.tile(2 * torch.pi * self.phases_freq * self.progress_buf.unsqueeze(1) * self.dt, (1, 4)) + phases_off

        # Manually refresh foot position and velocities because taking the 0:3 splice breaks the automatic update
        self.foot_positions = self.rigid_body_state.view(
            self.num_envs, self.num_bodies, 13
        )[:, self.feet_indices, 0:3]
        self.foot_velocities = self.rigid_body_state.view(
            self.num_envs, self.num_bodies, 13
        )[:, self.feet_indices, 0:3]

        # Update height scan
        self.measured_heights = self.get_heights()

        # Update ceilings
        pos_x = self.base_pos[:, 0]%8.0
        crawling_on = ( ((pos_x > 1.55) & (pos_x < 3.45)) | ((pos_x > 5.55) & (pos_x < 7.45)) ).float()
        self.ceilings = crawling_on * self.terrain_ceilings[self.terrain_levels, self.terrain_types] + (1.0-crawling_on)*0.4

        # Apply style constraints only on flat terrains
        self.is_flat_terrain = (self.measured_heights.var(1) < self.flat_terrain_threshold).float()

        # Update contact-related quantities
        self.process_contacts()

        # check if the robot went far enough to qualify for moving to higher difficulty
        if self.init_done and self.curriculum and self.custom_origins:
            # don't change on initial reset
            self.move_up_flag += torch.norm(self.root_states[:, :2] - self.env_origins[:, :2], dim=1) > (0.8 * self.terrain.env_length)

        # Update damping curriculum
        if "damping_curriculum" in self.cfg["env"]["control"]:
            if self.cfg["env"]["control"]["damping_curriculum"]["enable"]:
                init_Kd = self.cfg["env"]["control"]["damping_curriculum"]["init_damping"]
                last_Kd = self.cfg["env"]["control"]["damping"]
                n_Kd_curriculum_steps = self.cfg["env"]["control"]["damping_curriculum"]["num_steps"]

                self.Kd = init_Kd + (last_Kd - init_Kd) * min(self.common_step_counter / n_Kd_curriculum_steps, 1.0)

        # compute observations, rewards, resets, ...
        if self.useConstraints == "cat":
            self.compute_constraints_cat()
        self.compute_reward()

        # Display information in terminal for monitoring
        self.monitoring()
        
        # Set swing time and apex height to 0 for feet in contact (after reward/constraint computation)
        self.feet_swing_apex *= ~self.contacts_filt
        self.feet_swing_time *= ~self.contacts_filt

        # Retrieve environments that have to be reset
        if self.useConstraints == "cat":
            env_ids = self.reset_env_buf.nonzero(as_tuple=False).flatten()
        else:
            env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()

        # Reset environments
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        # Refresh base velocity commands
        if self.useJoystick:
            # Get new velocity from Joystick
            self.joystick.update_v_ref(self.common_step_counter + self.decimation, 0)
            self.commands[0, 0] = self.joystick.v_ref[0, 0]
            self.commands[0, 1] = self.joystick.v_ref[1, 0]
            self.commands[0, 2] = self.joystick.v_ref[-1, 0]
            self.commands[0] *= torch.any(torch.abs(self.commands[0, :3]) > self.vel_deadzone) # set small commands to zero
        else:
            # Random velocity command resampling
            #no_vel_command = (torch.norm(self.commands[:, :2], dim=1) < self.vel_deadzone).float()
            #p_resample_command = 0.01 * no_vel_command # + (self.dt / self.max_episode_length_s) * (1 - no_vel_command)

            # Resample with prob 2% if a velocity below 0.5 m/s is given
            p_resample_command = 0.01 * (torch.norm(self.commands[:, :2], dim=1) < 0.5) + 0.002
            resample_command_idx = torch.bernoulli(p_resample_command).nonzero(as_tuple=False).flatten()
            if len(resample_command_idx) > 0:
                self.resample_commands(resample_command_idx)



            # Random angular velocity inversion during the episode to avoid having the robot moving in circle
            p_ang_vel = self.dt / self.max_episode_length_s # <- time step / duration of X seconds
            # There will be a probability of 0.63 of having at least one swap after X seconds have elapsed
            # (1 / p) policy steps for X seconds, and the probability of having no swap at all is (1 - p)**(1 / p) = 0.37
            # The mean number of swaps for (1 / p) steps with probability p is 1.
            self.commands[:, 2] *= 1 - 2 * torch.bernoulli(torch.full_like(self.commands[:, 2], p_ang_vel)).float()

            # change the command direction every 100 steps
            self.commands[(torch.arange(self.num_envs) + self.common_step_counter) % 100 == 0, 1] *= -1

            # resample y command to avoid falling into lava
            y_command_pos = (self.root_states[:, 1] - self.env_origins[:, 1]) < -1.0
            self.commands[y_command_pos, 1] = torch.abs(self.commands[y_command_pos, 1])
            y_command_neg = (self.root_states[:, 1] - self.env_origins[:, 1]) > 1.0
            self.commands[y_command_neg, 1] = -torch.abs(self.commands[y_command_neg, 1])

        # Compute observations
        self.update_depth_buffer()
        self.compute_observations()

        # Keep history of previous actions, joint positions and velocities
        self.last_actions = torch.roll(self.last_actions, 1, 2)
        self.last_actions[:, :, 0] = self.actions[:]
        self.last_dof_pos = torch.roll(self.last_dof_pos, 1, 2)
        self.last_dof_pos[:, :, 0] = self.dof_pos[:]
        self.last_dof_vel = torch.roll(self.last_dof_vel, 1, 2)
        self.last_dof_vel[:, :, 0] = self.dof_vel[:]

        # If debug visualization is enabled, draw stuff in the simulation
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)

            # Draw small spheres to represent the height scan
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            sphere_geom = gymutil.WireframeSphereGeometry(0.01, 8, 8, None, color=(1, 0, 0))
            for i in range(self.num_envs):
                base_pos = (self.root_states[i, :3]).cpu().numpy()
                base_pos[2] = 0.1
                heights = self.measured_heights[i].cpu().numpy()
                height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
                for j in range(heights.shape[0]):
                    x = height_points[j, 0] + base_pos[0]
                    y = height_points[j, 1] + base_pos[1]
                    z = heights[j]
                    sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
                if i == 0:
                    self.camera_pos[:] = self.camera_pos[:] * 0.0 + base_pos[:] * 1.0
                    base_pos[:] = self.camera_pos[:]
                    cam_pos = gymapi.Vec3(base_pos[0] + 1.0, base_pos[1] - 0.7, base_pos[2] + 0.5)
                    cam_target = gymapi.Vec3(base_pos[0], base_pos[1], base_pos[2])
                    self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

            # If ground is flat
            if self.height_samples is None:
                return

            # Display a grid around the robot for easier visualization of terrain
            hsples = self.height_samples.cpu().numpy()
            gs = 48 # Grid size
            vs = self.terrain.vertical_scale
            hs = self.terrain.horizontal_scale
            base_pos = (self.root_states[0, :3]).cpu().numpy()
            bi = int(np.round((base_pos[0] + self.terrain.border_size) / hs))
            bj = int(np.round((base_pos[1] + self.terrain.border_size) / hs))

            x = np.linspace(bi - gs, bi + gs, 2 * gs + 1)
            y = np.linspace(bj - gs, bj + gs, 2 * gs + 1)
            grid_x, grid_y = (np.round(np.meshgrid(x, y))).astype(int)
            z = hsples[grid_x, grid_y]

            # Lines along X axis
            xx = (np.repeat(grid_x, 2, axis=1)[:, 1:-1]).reshape((-1, 2))
            yy = (np.repeat(grid_y, 2, axis=1)[:, 1:-1]).reshape((-1, 2))
            zz = (np.repeat(z, 2, axis=1)[:, 1:-1]).reshape((-1, 2))
            xyz = np.hstack((xx[:, 0:1], yy[:, 0:1], zz[:, 0:1], xx[:, 1:2], yy[:, 1:2], zz[:, 1:2]), dtype=np.float32) * np.array([hs, hs, vs] * 2, dtype=np.float32)
            xyz[:, [0, 1, 3, 4]] -= self.terrain.border_size
            self.gym.add_lines(
                self.viewer,
                self.envs[0],
                xyz.shape[0],
                xyz,
                np.zeros((xyz.shape[0], 3), dtype=np.float32),
            )

            # Lines along Y axis
            xx = (np.repeat(grid_x, 2, axis=0)[1:-1, :]).transpose().reshape((-1, 2))
            yy = (np.repeat(grid_y, 2, axis=0)[1:-1, :]).transpose().reshape((-1, 2))
            zz = (np.repeat(z, 2, axis=0)[1:-1, :]).transpose().reshape((-1, 2))
            xyz = np.hstack((xx[:, 0:1], yy[:, 0:1], zz[:, 0:1], xx[:, 1:2], yy[:, 1:2], zz[:, 1:2]), dtype=np.float32) * np.array([hs, hs, vs] * 2, dtype=np.float32)
            xyz[:, [0, 1, 3, 4]] -= self.terrain.border_size
            self.gym.add_lines(
                self.viewer,
                self.envs[0],
                xyz.shape[0],
                xyz,
                np.zeros((xyz.shape[0], 3), dtype=np.float32),
            )

    def monitoring(self):
        """Display information about rewards and constraints in terminal for monitoring."""

        # Initialize tensors for rewards
        if self.numRewards == -1:
            self.numRewards = len([item for item in list(self.episode_sums.keys()) if not item.startswith("cstr_")])
        if self.common_step_counter == 1:
            self.rew_mean = torch.zeros((self.num_envs, self.numRewards), dtype=torch.float, device=self.device, requires_grad=False)
            self.rew_cum_reset = torch.zeros((self.num_envs, self.numRewards), dtype=torch.float, device=self.device, requires_grad=False)
            self.rew_mean_reset = torch.zeros((self.num_envs, self.numRewards), dtype=torch.float, device=self.device, requires_grad=False)

        # If there is any enabled rewards
        if self.numRewards > 0:
            for i, key in enumerate(list(self.episode_sums.keys())[:self.numRewards]):
                # episode_sums are already cumulated sums
                self.rew_mean[:, i] = self.episode_sums[key]

        # Initialize tensors for CaT
        if self.numConstraints == -1:
            self.numConstraints = len(self.cstr_manager.get_names())
        if self.common_step_counter == 1:
            self.cstr_mean = torch.zeros((self.num_envs, self.numConstraints), dtype=torch.float, device=self.device, requires_grad=False)
            self.cstr_mean_reset = torch.zeros((self.num_envs, self.numConstraints), dtype=torch.float, device=self.device, requires_grad=False)

        # If there is any enabled constraint, log violations
        if self.numConstraints > 0:
            if self.useConstraints == "cat":
                for i, key in enumerate(self.cstr_manager.get_names()):
                    self.cstr_mean[:, i] += self.cstr_manager.probs[key].max(1).values.gt(0.0).float()
            else:
                self.cstr_mean += self.cstr_buf

        if self.common_step_counter % self.cfg["horizon_length"] != 0:
            return

        if not self.cfg["test"]:
            print("------- Epoch {0} / {1}".format(int(self.common_step_counter / self.cfg["horizon_length"]) - 1, self.cfg["max_epochs"]))
        else:
            print("------- Testing")
        print("")

        # Display average rewards and constraints violations over the last episodes
        table_rew = Texttable()
        table_rew.set_deco(Texttable.HEADER)
        table_rew.set_cols_align(["l", "r"])
        table_rew.set_cols_dtype(['t', 'f'])
        table_rew.add_rows(
            np.array([
                ["Rewards"] + list(self.episode_sums.keys())[:self.numRewards],
                ["Average/step"] + (torch.mean(self.rew_mean_reset, dim=0)).tolist()
            ]).transpose()
        )

        table_rew_cum = Texttable()
        table_rew_cum.set_deco(Texttable.HEADER)
        table_rew_cum.set_cols_align(["l", "r"])
        table_rew_cum.set_cols_dtype(['t', 'f'])
        #table_rew_cum.add_rows(
        #    np.array([
        #        ["Info"] + ["Cumulated Rewards"] + ["Average level"],
        #        ["Value"] + [(torch.sum(torch.mean(self.rew_cum_reset, dim=0))).item()] + [self.terrain_levels.float().mean().item()]
        #    ]).transpose()
        #)
        table_rew_cum.add_rows(
            np.array([
                ["Info"] + ["Value"],
                ["Cumulated Rewards"] + [(torch.sum(torch.mean(self.rew_cum_reset, dim=0))).item()],
                ["Average level"] + [self.terrain_levels.float().mean().item()],
            ])
        )


        # Split the table strings into lines
        lines_A = table_rew.draw().split("\n") + ["", ""] + table_rew_cum.draw().split("\n") + [""]

        if self.numConstraints > 0:
            cstr_names = self.cstr_manager.get_names()

            table_cstr = Texttable()
            table_cstr.set_deco(Texttable.HEADER)
            table_cstr.set_cols_align(["l", "r"])
            table_cstr.set_cols_dtype(['t', 'f'])
            table_cstr.add_rows(
                np.array([
                    ["Constraints"] + cstr_names,
                    ["Violation (%)"] + (100.0 * torch.mean(self.cstr_mean_reset, dim=0)).tolist()
                ]).transpose()
            )

            # Split the table strings into lines
            lines_B = table_cstr.draw().split("\n") + [""]
        else:
            lines_B = [""]

        # Calculate the maximum and minimum width for each table
        max_width_A = max(len(line) for line in lines_A)
        max_width_B = max(len(line) for line in lines_B)

        # Line padding to bring them to the same length
        for i, line in enumerate(lines_A):
            lines_A[i] += ' ' * (max_width_A - len(line))

        # Add some padding between the tables
        padding = 5

        # Use itertools.zip_longest to handle tables with different numbers of lines
        filling = '' if len(lines_A) > len(lines_B) else ' ' * (max_width_A)
        for line_A, line_B in itertools.zip_longest(lines_A, lines_B, fillvalue=filling):
            print(f"{line_A}{' ' * padding}{line_B}")


    def prepare_height_points(self):
        """Prepare the (x, y) grid of points to sample for the height scan"""

        step = self.cfg["env"]["learn"]["measured_points_step"]
        y = step * torch.tensor(self.cfg["env"]["learn"]["measured_points_y"], device=self.device, requires_grad=False)
        x = step * torch.tensor(self.cfg["env"]["learn"]["measured_points_x"], device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        assert self.num_height_points == grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def get_heights(self, env_ids=None):
        """Compute the height scan by retrieving the corresponding z for each (x, y) point of the grid scan"""

        if self.cfg["env"]["terrain"]["terrainType"] == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg["env"]["terrain"]["terrainType"] == 'none':
            raise NameError("Can't measure height with terrain type 'none'")

        if env_ids:
            #if self.cfg["test"]:
            #    points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
            #else:
            #    root_pos = (2*torch.rand_like(self.root_states[env_ids, :3])-1)*torch.tensor([0.03, 0.03, 0.0]).to(self.device) + self.root_states[env_ids, :3]
            #    points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (root_pos).unsqueeze(1)
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            #if self.cfg["test"]:
            #    points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)
            #else:
            #    root_pos = (2*torch.rand_like(self.root_states[:, :3])-1)*torch.tensor([[0.03, 0.03, 0.0]]).to(self.device) + self.root_states[:, :3]
            #    points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (root_pos).unsqueeze(1)
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)
 
        points += self.terrain.border_size
        points = (points/self.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        # First method
        # heights1 = self.height_samples[px, py]
        # heights2 = self.height_samples[px+1, py+1]
        # heights = torch.min(heights1, heights2)

        # Second method
        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.vertical_scale

    def get_feet_heights(self):
        """Get feet heights with respect to the ground using ground heightmap"""

        if self.cfg["env"]["terrain"]["terrainType"] == 'plane':
            return self.foot_positions[:, :, -1]
        else:
            points = self.foot_positions.clone()
            points += self.terrain.border_size
            points = (points / self.terrain.horizontal_scale).long()
            points[:, :, 0] = torch.clip(
                points[:, :, 0], 0, self.height_samples.shape[0] - 2
            )
            points[:, :, 1] = torch.clip(
                points[:, :, 1], 0, self.height_samples.shape[1] - 2
            )

            heights1 = self.height_samples[points[:, :, 0], points[:, :, 1]]
            heights2 = self.height_samples[points[:, :, 0] + 1, points[:, :, 1]]
            heights3 = self.height_samples[points[:, :, 0], points[:, :, 1] + 1]
            heights = torch.min(heights1, heights2)
            heights = torch.min(heights, heights3)

            return self.foot_positions[:, :, -1] - heights * self.terrain.vertical_scale

####################
# jit functions
####################

@torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

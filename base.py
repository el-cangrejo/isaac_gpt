import copy
import random
import math
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

from pykin.kinematics import transform as tf
from pykin.robots.single_arm import SingleArm

class BaseSim():
    def __init__(self, args):
        self.args = args

        self.gym = gymapi.acquire_gym()
        self.sim_params = gymapi.SimParams()
        self.sim = None
        self.viewer = None
        self.env = None
        self.create_sim()

        self.robot_asset = None
        self.robot_handle = None
        self.load_robot()
        self.set_joint_positions(self.default_dof_pos)

        self.object_handle = None
        self.load_assets()

        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(1, -1, 13)
        self.step()
        self.target_pos = self.cur_pos.copy()

    def create_sim(self):
        # create a simulator
        sim_params = gymapi.SimParams()
        sim_params.substeps = 4
        sim_params.dt = 1.0 / 60.0
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        # sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)

        sim_params.physx.num_position_iterations = 6
        sim_params.physx.num_velocity_iterations = 5
        sim_params.physx.bounce_threshold_velocity = 0.28
        sim_params.physx.contact_offset = 0.001
        sim_params.physx.rest_offset = 0.00001
        sim_params.physx.friction_offset_threshold = 0.01
        sim_params.physx.friction_correlation_distance = 0.05
        sim_params.physx.max_depenetration_velocity = 1000.0

        sim_params.physx.num_threads = self.args.num_threads
        sim_params.physx.use_gpu = self.args.use_gpu

        sim_params.use_gpu_pipeline = False
        if self.args.use_gpu_pipeline:
            print("WARNING: Forcing CPU pipeline.")

        self.sim = self.gym.create_sim(self.args.compute_device_id,
                self.args.graphics_device_id, self.args.physics_engine, sim_params)

        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        # create viewer using the default camera properties
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise ValueError('*** Failed to create viewer')

        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        # set up the env grid
        num_envs = 1
        spacing = 1.5
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, 0.0, spacing)
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, 2)

        # Look at the first env
        cam_pos = gymapi.Vec3(8, 4, 1.5)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.2)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def load_robot(self):
        # add hand urdf asset
        asset_root = "assets"
        self.robot_asset_file = "urdf/franka_description/robots/franka_panda.urdf"

        self.robot_ik = SingleArm('assets/' + self.robot_asset_file, 
                                  tf.Transform(rot=[0.0, 0.0, 0.0], 
                                               pos=[0.0, 0.0, 0.0]))
        self.robot_ik.setup_link_name("panda_link0", "panda_eef")

        # Load asset with default control type of position for all joints
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        asset_options.override_com = True
        asset_options.override_inertia = True
        asset_options.armature = 0.0001
        print("Loading asset '%s' from '%s'" % (self.robot_asset_file, asset_root))
        self.robot_asset = self.gym.load_asset(self.sim, asset_root, self.robot_asset_file, asset_options)

        # initial root pose for hand actors
        self.robot_pose = gymapi.Transform()
        self.robot_pose.p = gymapi.Vec3(0., 0., 0.)
        self.robot_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1)

        self.robot_handle = self.gym.create_actor(self.env, self.robot_asset, self.robot_pose, 'robot', 0, 1)

        self.franka_eef = self.gym.find_actor_rigid_body_handle(self.env,
                                                                self.robot_handle,
                                                                "panda_eef")
        self.set_actor_properties()

    def load_assets(self):
        objects_dict = {"coke" : "coke_can/model.urdf",
                        "pear" : "pear/model.urdf",
                        "meat_can" : "meat_can/model.urdf",
                        "banana" : "banana/banana.urdf",
                        "orange" : "orange/model.urdf"}

        self.object_rb_idxs = {}
        self.object_handles = {}

        asset_root = "assets"
        ## Asset options
        asset_options = gymapi.AssetOptions()
        asset_options.thickness = 0.0001
        asset_options.use_mesh_materials = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.override_inertia = True
        asset_options.override_com = True
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.convex_hull_downsampling = 10
        asset_options.vhacd_params.resolution = 2000000

        ### Load cabinet
        cabinet_asset_file = "urdf/sektion_cabinet_model/urdf/sektion_cabinet.urdf"
        cabinet_pose = gymapi.Transform()
        cabinet_pose.p.x = -1.0
        cabinet_pose.p.y = 0.0
        cabinet_pose.p.z = 0.40

        cabinet_asset = self.gym.load_asset(self.sim, asset_root,
                                            cabinet_asset_file,
                                            asset_options)

        self.cabinet_handle = self.gym.create_actor(self.env, cabinet_asset,
                                                    cabinet_pose, 'cabinet', 0, 0)

        drawer_handle_top_eef = self.gym.find_actor_rigid_body_handle(self.env,
                                                                      self.cabinet_handle,
                                                                      "drawer_handle_top_eef")

        drawer_handle_top_pos = self.gym.find_actor_rigid_body_handle(self.env,
                                                                      self.cabinet_handle,
                                                                      "drawer_handle_top_pos")

        drawer_handle_bottom_eef = self.gym.find_actor_rigid_body_handle(self.env,
                                                                      self.cabinet_handle,
                                                                      "drawer_handle_bottom_eef")

        drawer_handle_bottom_pos = self.gym.find_actor_rigid_body_handle(self.env,
                                                                           self.cabinet_handle,
                                                                           "drawer_handle_bottom_pos")

        self.object_rb_idxs["drawer_handle_top_eef"] = drawer_handle_top_eef
        self.object_rb_idxs["drawer_handle_top_pos"] = drawer_handle_top_pos
        self.object_rb_idxs["drawer_handle_bottom_eef"] = drawer_handle_bottom_eef
        self.object_rb_idxs["drawer_handle_bottom_pos"] = drawer_handle_bottom_pos

        cabinet_dof_props = self.gym.get_actor_dof_properties(self.env,
                                                              self.cabinet_handle)

        cabinet_dof_props["friction"].fill(500.0)
        cabinet_dof_props["armature"].fill(1.0)
        cabinet_dof_props["stiffness"].fill(0.1)
        cabinet_dof_props["damping"].fill(0.1)

        self.gym.set_actor_dof_properties(self.env,
                                          self.cabinet_handle,
                                          cabinet_dof_props)
        ### Load table
        table_pose = gymapi.Transform()
        table_pose.p.x = 0.65
        table_pose.p.y = 0.0
        table_pose.p.z = 0.01

        table_asset = self.gym.load_asset(self.sim, asset_root,
                                          "urdf/objects/table/model.urdf",
                                          asset_options)

        self.table_handle_1 = self.gym.create_actor(self.env, table_asset,
                                                    table_pose, 'table1', 0, 0)

        table_1 = self.gym.find_actor_rigid_body_handle(self.env,
                                                        self.table_handle_1,
                                                        "table")

        self.object_rb_idxs["table_1"] = table_1

        table_pose.p.x = 0.0
        table_pose.p.y = 0.85
        table_pose.p.z = 0.01
        table_pose.r.x = 0.0
        table_pose.r.y = 0.0
        table_pose.r.z = 1.0
        table_pose.r.w = 1.0

        self.table_handle_2 = self.gym.create_actor(self.env, table_asset,
                                                    table_pose, 'table2', 0, 0)

        table_2 = self.gym.find_actor_rigid_body_handle(self.env,
                                                        self.table_handle_2,
                                                        "table")

        self.object_rb_idxs["table_2"] = table_2

        ### Load objects
        object_pose = gymapi.Transform()
        object_pose.p.x = 0.6
        object_pose.p.y = -0.65
        object_pose.p.z = 0.4

        items = list(objects_dict.items()) # List of tuples
        random.shuffle(items)
        print ('')
        for i, (k, v) in enumerate(items):
            object_pose.p.y += 0.2
            object_pose.p.x = 0.5 + random.random() / 5.

            object_asset = self.gym.load_asset(self.sim, asset_root,
                                               "urdf/objects/" +
                                               objects_dict[k], asset_options)

            object_name = self.gym.get_asset_rigid_body_names(object_asset)[0]

            print (f'Object rigid body name: {object_name}')

            object_handle = self.gym.create_actor(self.env, object_asset,
                                                  object_pose, object_name, 0, 0)

            object_rb_idx = self.gym.find_actor_rigid_body_handle(self.env,
                                                                  object_handle,
                                                                  object_name)
            self.object_rb_idxs[object_name] = object_rb_idx
            self.object_handles[object_name] = object_handle

        object_pose.p.x = 0.2
        object_pose.p.y = 0.65
        object_pose.p.z = 0.5
        object_asset = self.gym.load_asset(self.sim, asset_root,
                                           "urdf/objects/bowl/model.urdf",
                                           asset_options)

        object_name = self.gym.get_asset_rigid_body_names(object_asset)[0]
        print (f'Object rigid body name: {object_name}')

        object_handle = self.gym.create_actor(self.env, object_asset,
                                              object_pose, object_name, 0, 0)

        object_rb_idx = self.gym.find_actor_rigid_body_handle(self.env,
                                                              object_handle,
                                                              object_name)
        self.object_rb_idxs[object_name] = object_rb_idx

    def set_actor_properties(self):
        # Configure DOF properties
        robot_dof_props = self.gym.get_actor_dof_properties(self.env, self.robot_handle)

        robot_lower_limits = robot_dof_props["lower"]
        robot_upper_limits = robot_dof_props["upper"]
        robot_ranges = robot_upper_limits - robot_lower_limits
        robot_mids = 0.3 * (robot_upper_limits + robot_lower_limits)

        # use position drive for all dofs
        controller = "ik"
        if controller == "ik":
            robot_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
            robot_dof_props["stiffness"][:7].fill(1000.0)
            robot_dof_props["damping"][:7].fill(100.0)
        else:       # osc
            robot_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
            robot_dof_props["stiffness"][:7].fill(0.0)
            robot_dof_props["damping"][:7].fill(0.0)
        # grippers
        robot_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
        robot_dof_props["effort"][7:].fill(800.0)
        robot_dof_props["stiffness"][7:].fill(8000.0)
        robot_dof_props["damping"][7:].fill(40.0)

        self.gym.set_actor_dof_properties(self.env,
                                          self.robot_handle,
                                          robot_dof_props)
        self.num_dofs = len(robot_dof_props)

        # default dof states and position targets
        self.default_dof_pos = np.zeros(self.num_dofs, dtype=np.float32)
        self.default_dof_pos[:7] = robot_mids[:7]
        self.default_dof_pos[3] += 0.2
        self.default_dof_pos[5] += 0.2
        # grippers open
        self.default_dof_pos[7:] = robot_upper_limits[7:]

    def get_joint_positions(self):
        robot_dof_states = self.gym.get_actor_dof_states(self.env, self.robot_handle, gymapi.STATE_POS)
        return robot_dof_states['pos']

    def set_joint_positions(self, joint_positions):
        robot_dof_states = self.gym.get_actor_dof_states(self.env, self.robot_handle, gymapi.STATE_POS)

        for i in range(self.num_dofs):
            robot_dof_states['pos'][i] = joint_positions[i]

        self.gym.set_actor_dof_states(self.env, self.robot_handle, robot_dof_states, gymapi.STATE_POS)
        self.gym.set_actor_dof_position_targets(self.env, self.robot_handle, joint_positions)

    def get_object_pos(self, obj):
        if obj == "drawer":
            pose = self.get_object_pose_idx(self.object_rb_idxs["drawer_handle_top_pos"])
        else:
            pose = self.get_object_pose_idx(self.object_rb_idxs[obj])

        return np.array([pose.p.x, pose.p.y, pose.p.z])

    def set_joint_target_positions(self, joint_positions):
        self.gym.set_actor_dof_position_targets(self.env, self.robot_handle, joint_positions)

    def get_object_pose_idx(self, object_idx):
        pose = self.rigid_body_states[:, object_idx][:, 0:7]
        pose = array_to_transform(pose)
        return pose

    def set_object_pose(self, pose, object_handle):
        rb_state = self.gym.get_actor_rigid_body_states(self.env, object_handle, gymapi.STATE_ALL)
        rb_state['pose']['p'][0][0] = pose.p.x
        rb_state['pose']['p'][0][1] = pose.p.y
        rb_state['pose']['p'][0][2] = pose.p.z
        rb_state['pose']['r'][0][0] = pose.r.x
        rb_state['pose']['r'][0][1] = pose.r.y
        rb_state['pose']['r'][0][2] = pose.r.z
        rb_state['pose']['r'][0][3] = pose.r.w

        rb_state['vel']['linear'][0][0] = 0
        rb_state['vel']['linear'][0][1] = 0
        rb_state['vel']['linear'][0][2] = 0
        rb_state['vel']['angular'][0][0] = 0
        rb_state['vel']['angular'][0][1] = 0
        rb_state['vel']['angular'][0][2] = 0
        self.gym.set_actor_rigid_body_states(self.env, object_handle, rb_state, gymapi.STATE_POS)

    def draw_pose(self, pose, axis_size=0.1):
        axis_geom = gymutil.AxesGeometry(axis_size)

        axis_pose = gymapi.Transform()

        axis_pose.p.x = pose[0]
        axis_pose.p.y = pose[1]
        axis_pose.p.z = pose[2]

        axis_pose.r.x = pose[3]
        axis_pose.r.y = pose[4]
        axis_pose.r.z = pose[5]
        axis_pose.r.w = pose[6]

        gymutil.draw_lines(axis_geom, self.gym, self.viewer, self.env, axis_pose)

    def step(self, n=1):
        for i in range(n):
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)

            # Wait for dt to elapse in real time.
            # This synchronizes the physics simulation with the rendering rate.
            self.gym.sync_frame_time(self.sim)

            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.cur_pos = self.get_joint_positions()

    def get_ik_sol(self, goal_pose):
        return list(self.robot_ik.inverse_kin(self.cur_pos, goal_pose,
                                              method="LM", max_iter=1000))

    def pick_up(self, object_name):
        pose = self.get_object_pose_idx(self.object_rb_idxs[object_name])
        # (angle_z, angle_y, angle_x) = pose.r.to_euler_zyx()

        # angle_z += np.pi
        # pose.r = gymapi.Quat().from_euler_zyx(angle_z, angle_y, angle_x)
        # pose.p.z += 0.05

        pose.r.x = 1.0
        pose.r.y = 0.0
        pose.r.z = 0.0
        pose.r.w = 0.0
        # self.draw_pose(transform_to_np(pose)[0])
        
        goal_pose = [pose.p.x,
                     pose.p.y,
                     pose.p.z + 0.1,
                     pose.r.w,
                     pose.r.x,
                     pose.r.y,
                     pose.r.z]

        self.target_pos = self.get_ik_sol(goal_pose)
        self.set_joint_target_positions(self.target_pos)
        self.step(150)

        goal_pose = [pose.p.x,
                     pose.p.y,
                     pose.p.z + 0.03,
                     pose.r.w,
                     pose.r.x,
                     pose.r.y,
                     pose.r.z]

        self.target_pos = self.get_ik_sol(goal_pose)
        self.set_joint_target_positions(self.target_pos)
        self.step(100)

        self.target_pos = self.cur_pos.copy()
        self.close_gripper()
        self.step(50)
        self.lift()

    def transport_to(self, object_name):
        self.step()
        if object_name == "bowl":
            pose = self.get_object_pose_idx(self.object_rb_idxs["bowl"])
        elif object_name == "drawer":
            pose = self.get_object_pose_idx(self.object_rb_idxs["drawer_handle_top_pos"])

        (angle_z, angle_y, angle_x) = pose.r.to_euler_zyx()

        angle_z += np.pi
        pose.r = gymapi.Quat().from_euler_zyx(angle_z, angle_y, angle_x)
        pose.p.z += 0.05

        # pose.r.x = 1.0
        # pose.r.y = 0.0
        # pose.r.z = 0.0
        # pose.r.w = 0.0
        # self.draw_pose(transform_to_np(pose)[0])

        goal_pose = [pose.p.x,
                     pose.p.y,
                     pose.p.z + 0.3,
                     pose.r.w,
                     pose.r.x,
                     pose.r.y,
                     pose.r.z]

        self.target_pos[:-2] = self.get_ik_sol(goal_pose)[:-2]
        self.set_joint_target_positions(self.target_pos)
        self.step(250)

        goal_pose = [pose.p.x,
                     pose.p.y,
                     pose.p.z + 0.05,
                     pose.r.w,
                     pose.r.x,
                     pose.r.y,
                     pose.r.z]

        self.target_pos[:-2] = self.get_ik_sol(goal_pose)[:-2]
        self.set_joint_target_positions(self.target_pos)
        self.step(100)

        self.open_gripper()
        self.lift()
        self.go_to_home()

    def put_first_on_second(self, first_item, pos):
        self.pick_up(first_item)
        self.move_to_pos(pos)

    def move_to_pos(self, pos):
        self.step()

        pose = gymapi.Transform()
        pose.p.x = pos[0]
        pose.p.y = pos[1]
        pose.p.z = pos[2]
        pose.r.x = 1.0
        pose.r.y = 0.0
        pose.r.z = 0.0
        pose.r.w = 0.0
        # self.draw_pose(transform_to_np(pose)[0])

        goal_pose = [pose.p.x,
                     pose.p.y,
                     pose.p.z + 0.3,
                     pose.r.w,
                     pose.r.x,
                     pose.r.y,
                     pose.r.z]

        self.target_pos[:-2] = self.get_ik_sol(goal_pose)[:-2]
        self.set_joint_target_positions(self.target_pos)
        self.step(250)

        goal_pose = [pose.p.x,
                     pose.p.y,
                     pose.p.z + 0.05,
                     pose.r.w,
                     pose.r.x,
                     pose.r.y,
                     pose.r.z]

        self.target_pos[:-2] = self.get_ik_sol(goal_pose)[:-2]
        self.set_joint_target_positions(self.target_pos)
        self.step(100)

        self.open_gripper()
        self.lift()
        self.go_to_home()

    def open_gripper(self):
        self.step()
        self.target_pos[-1] = 0.04
        self.target_pos[-2] = 0.04
        self.set_joint_target_positions(self.target_pos)

    def close_gripper(self):
        self.step()
        self.target_pos[-1] = 0.0
        self.target_pos[-2] = 0.0
        self.set_joint_target_positions(self.target_pos)

    def lift(self):
        self.step()
        pose = self.rigid_body_states[:, self.franka_eef][:, 0:7]
        pose = array_to_transform(pose)
        goal_pose = [pose.p.x,
                     pose.p.y,
                     pose.p.z + 0.4,
                     pose.r.w,
                     pose.r.x,
                     pose.r.y,
                     pose.r.z]

        self.target_pos[:-2] = self.get_ik_sol(goal_pose)[:-2]
        self.set_joint_target_positions(self.target_pos)
        self.step(100)

    def reset(self):
        object_pose = gymapi.Transform()
        object_pose.p.x = 0.6
        object_pose.p.y = -0.65
        object_pose.p.z = 0.5

        items = list(self.object_handles.items()) # List of tuples
        random.shuffle(items)
        for i, (k, v) in enumerate(items):
            object_pose.p.y += 0.2
            object_pose.p.x = 0.45 + random.random() / 5.
            self.set_object_pose(object_pose, self.object_handles[k])

    def open_drawer(self, drawer='top'):
        if drawer == "top":
            drawer_pose_t = self.get_object_pose_idx(self.object_rb_idxs["drawer_handle_top_eef"])
            # drawer_pose = self.rigid_body_states[:, self.drawer_handle_top_eef][:, 0:7]
        elif drawer == "bottom":
            drawer_pose_t = self.get_object_pose_idx(self.object_rb_idxs["drawer_handle_bottom_eef"])
            # drawer_pose = self.rigid_body_states[:, self.drawer_handle_bottom_eef][:, 0:7]

        # drawer_pose_t = array_to_transform(drawer_pose)

        (angle_z, angle_y, angle_x) = drawer_pose_t.r.to_euler_zyx()
        angle_x += 3 * np.pi / 2.
        angle_z += np.pi / 2.
        # print (angle_z, angle_y, angle_x)
        drawer_pose_t.r = gymapi.Quat().from_euler_zyx(angle_z, angle_y, angle_x)
        
        # self.draw_pose(transform_to_np(drawer_pose_t)[0])

        goal_pose = [drawer_pose_t.p.x + 0.09,
                     drawer_pose_t.p.y,
                     drawer_pose_t.p.z,
                     drawer_pose_t.r.w,
                     drawer_pose_t.r.x,
                     drawer_pose_t.r.y,
                     drawer_pose_t.r.z]

        self.target_pos[:-2] = self.get_ik_sol(goal_pose)[:-2]
        self.set_joint_target_positions(self.target_pos)
        self.step(150)

        goal_pose = [drawer_pose_t.p.x + 0.005,
                     drawer_pose_t.p.y,
                     drawer_pose_t.p.z,
                     drawer_pose_t.r.w,
                     drawer_pose_t.r.x,
                     drawer_pose_t.r.y,
                     drawer_pose_t.r.z]

        self.target_pos[:-2] = self.get_ik_sol(goal_pose)[:-2]
        self.set_joint_target_positions(self.target_pos)
        self.step(50)

        self.target_pos = self.cur_pos.copy()
        self.close_gripper()
        self.step(50)

        goal_pose = [drawer_pose_t.p.x + 0.3,
                     drawer_pose_t.p.y,
                     drawer_pose_t.p.z,
                     drawer_pose_t.r.w,
                     drawer_pose_t.r.x,
                     drawer_pose_t.r.y,
                     drawer_pose_t.r.z]

        self.target_pos[:-2] = self.get_ik_sol(goal_pose)[:-2]
        self.set_joint_target_positions(self.target_pos)
        self.step(50)

        self.open_gripper()
        self.step(50)
        self.go_to_home()

    def close_drawer(self, drawer='top'):
        if drawer == "top":
            drawer_pose_t = self.get_object_pose_idx(self.object_rb_idxs["drawer_handle_top_eef"])
        elif drawer == "bottom":
            drawer_pose_t = self.get_object_pose_idx(self.object_rb_idxs["drawer_handle_bottom_eef"])

        (angle_z, angle_y, angle_x) = drawer_pose_t.r.to_euler_zyx()
        angle_x += 3 * np.pi / 2.
        angle_z += np.pi / 2.
        drawer_pose_t.r = gymapi.Quat().from_euler_zyx(angle_z, angle_y, angle_x)
        
        # self.draw_pose(transform_to_np(drawer_pose_t)[0])

        goal_pose = [drawer_pose_t.p.x + 0.09,
                     drawer_pose_t.p.y,
                     drawer_pose_t.p.z,
                     drawer_pose_t.r.w,
                     drawer_pose_t.r.x,
                     drawer_pose_t.r.y,
                     drawer_pose_t.r.z]

        self.target_pos[:-2] = self.get_ik_sol(goal_pose)[:-2]
        self.set_joint_target_positions(self.target_pos)
        self.step(150)

        self.target_pos = self.cur_pos.copy()
        self.close_gripper()
        self.step(50)

        goal_pose = [drawer_pose_t.p.x - 0.2,
                     drawer_pose_t.p.y,
                     drawer_pose_t.p.z,
                     drawer_pose_t.r.w,
                     drawer_pose_t.r.x,
                     drawer_pose_t.r.y,
                     drawer_pose_t.r.z]

        self.target_pos[:-2] = self.get_ik_sol(goal_pose)[:-2]
        self.set_joint_target_positions(self.target_pos)
        self.step(150)

        goal_pose = [drawer_pose_t.p.x,
                     drawer_pose_t.p.y,
                     drawer_pose_t.p.z,
                     drawer_pose_t.r.w,
                     drawer_pose_t.r.x,
                     drawer_pose_t.r.y,
                     drawer_pose_t.r.z]

        self.target_pos[:-2] = self.get_ik_sol(goal_pose)[:-2]
        self.set_joint_target_positions(self.target_pos)
        self.step(50)

        self.open_gripper()
        self.go_to_home()

    def go_to_home(self):
        self.set_joint_target_positions(self.default_dof_pos)
        self.step(100)

def array_to_transform(p):
    pose = gymapi.Transform()
    pose.p.x = p[0][0]
    pose.p.y = p[0][1]
    pose.p.z = p[0][2]
    pose.r.x = p[0][3]
    pose.r.y = p[0][4]
    pose.r.z = p[0][5]
    pose.r.w = p[0][6]
    return pose

def transform_to_torch(t):
    return torch.tensor([[t.p.x, t.p.y, t.p.z, t.r.x, t.r.y, t.r.z, t.r.w]])

def transform_to_np(t):
    return np.array([[t.p.x, t.p.y, t.p.z, t.r.x, t.r.y, t.r.z, t.r.w]])


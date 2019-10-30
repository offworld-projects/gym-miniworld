import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box
from ..params import DEFAULT_PARAMS
from gym import spaces

# Simulation parameters
# These assume a robot about 15cm tall with a pi camera module v2
sim_params = DEFAULT_PARAMS.copy()
# sim_params.set('forward_step', 0.035, 0.028, 0.042)
# sim_params.set('turn_step', 17, 13, 21)
sim_params.set('forward_step', 0.05, 0., 0.1)
sim_params.set('turn_step', 10, 0, 10)
sim_params.set('bot_radius', 0.4, 0.38, 0.42) # FIXME: not used
sim_params.set('cam_pitch', -15, -15, -15)
sim_params.set('cam_fov_y', 49, 45, 55)
sim_params.set('cam_height', 0.1, 0.1, 0.1)
sim_params.set('cam_fwd_disp', 0, -0.02, 0.02)

# TODO: modify lighting parameters

class SimToReal2Env(MiniWorldEnv):
    """
    Environment designed for sim-to-real transfer.
    In this environment, the robot has to go to the red box.
    """

    def __init__(self, **kwargs):
        super().__init__(
            # max_episode_steps=100,
            params=sim_params,
            domain_rand=False,
            **kwargs
        )

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward+1)
        self.last_action=[0,0]
        sim_params.set('forward_step', 0.0, 0., 0.1)

    def _gen_world(self):
        # 1-2 meter wide rink
        # size = self.rand.float(1, 2)
        size = 2.

        # wall_height = self.rand.float(0.20, 0.50)
        wall_height = 0.50

        # box_size = self.rand.float(0.07, 0.12)
        box_size = 0.08
        self.agent.radius = 0.05

        floor_tex = self.rand.choice([
            'cardboard',#
            # 'wood',
            # 'wood_planks',
        ])

        wall_tex = self.rand.choice([
            # 'drywall',
            'stucco',#
            # 'cardboard',
            # Chosen because they have visible lines/seams
            # 'concrete_tiles',
            # 'ceiling_tiles',
        ])

        self.one_room = True
        if self.one_room:
            # Create a long rectangular room
            room = self.add_rect_room(
                min_x=0,
                max_x=size,
                min_z=0,
                max_z=size,
                no_ceiling=True,
                wall_height=wall_height,
                wall_tex=wall_tex,
                floor_tex=floor_tex
            )
            self.place_agent()
            self.box1 = self.place_entity(Box(color='red', size=0.5*box_size))
            self.box2 = self.place_entity(Box(color='green', size=box_size))
            # make sure it is a bit far away from the agent
            while np.linalg.norm(self.box1.pos[[0,2]]- self.agent.pos[[0,2]]) < 1.:
                self.box1.pos = self.rand.float(
                    low = [0+self.box1.radius, 0, 0+self.box1.radius],
                    high = [size-self.box1.radius, 0, size-self.box1.radius]
                )
            while np.linalg.norm(self.box2.pos[[0,2]]- self.agent.pos[[0,2]]) < 1.:
                self.box2.pos = self.rand.float(
                    low = [0+self.box2.radius, 0, 0+self.box2.radius],
                    high = [size-self.box2.radius, 0, size-self.box2.radius]
                )
        else:
            room1 = self.add_rect_room(
                min_x=0,
                max_x=size/2.-0.05,
                min_z=0,
                max_z=size,
                no_ceiling=True,
                wall_height=wall_height,
                wall_tex=wall_tex,
                floor_tex=floor_tex
            )
            room2 = self.add_rect_room(
                min_x=size/2.+0.05,
                max_x=size,
                min_z=0,
                max_z=size,
                no_ceiling=True,
                wall_height=wall_height,
                wall_tex='wood',
                # wall_tex=wall_tex,
                floor_tex=floor_tex
            )
            # self.connect_rooms(room1, room2, min_z=1.3, max_z=1.7, max_y=wall_height)
            self.connect_rooms(room1, room2, min_z=size/2.-.30, max_z=size/2.+.30, max_y=wall_height)

            self.box = self.place_entity(Box(color='red', size=box_size), room=room2)

            # place the agent in the other room
            self.place_agent(room=room1)

    def reset(self):
        obs = super().reset()
        return [obs, [self.agent.pos[0], self.agent.pos[2], self.agent.dir, int(self.agent.carrying!=None)]]

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if not self.agent.carrying:
            if self.near(self.box1):
                box_vec = self.box1.pos - self.agent.pos
                box_vec = [box_vec[0], box_vec[2]]
                dir_vec = [self.agent.dir_vec[0], self.agent.dir_vec[2]]
                perp_dist = math.sqrt(np.linalg.norm(box_vec)**2-(np.dot(box_vec, dir_vec)/np.linalg.norm(dir_vec))**2)
                if perp_dist < 0.01:
                    self.agent.carrying = self.box1
        else:
            if self.near(self.box1, self.box2):
                box_vec = self.box2.pos - self.agent.pos
                box_vec = [box_vec[0], box_vec[2]]
                dir_vec = [self.agent.dir_vec[0], self.agent.dir_vec[2]]
                perp_dist = math.sqrt(np.linalg.norm(box_vec)**2-(np.dot(box_vec, dir_vec)/np.linalg.norm(dir_vec))**2)
                if perp_dist < 0.01:
                    reward += 1
                    done = True


        obs = [obs, [self.agent.pos[0], self.agent.pos[2], self.agent.dir, int(self.agent.carrying!=None)]]
        return obs, reward, done, info

class SimToReal2ContEnv(SimToReal2Env):
    def __init__(self, **kwargs):
        super().__init__(
            **kwargs
        )
        cont_params = sim_params.copy()
        # if self.one_room:
        cont_params.set('forward_step', 0.01, -0.1, 0.1)
        # else:
            # cont_params.set('forward_step', 0.01, 0., 0.1)
            # cont_params.set('forward_step', 0.01, 0., 0.05)
            # cont_params.set('forward_step', 0.01, -0.05, 0.05)
        cont_params.set('turn_step', 0, -10, 10)
        self.params = cont_params
        self.action_space = spaces.Box(-1,1, shape=(2,))
        self.last_action = [0.,0.]

    def step(self, action):
        """
        Perform one continuous action and update the simulation
        Actions are only moving and turning
        """

        self.last_action = action
        self.step_count += 1
        rand = self.rand
        # fwd_step = (action[0]+1)*self.params.get_max('forward_step')/2 #fwd mvt only
        fwd_step = action[0]*self.params.get_max('forward_step') #fwd mvt only
        turn_step = action[1]*self.params.get_max('turn_step')
        self.turn_agent(turn_step)
        self.move_agent(fwd_step)

        # Generate the current camera image
        obs = self.render_obs()

        # If the maximum time step count is reached
        if self.step_count >= self.max_episode_steps:
            done = True
            reward = 0
            return obs, reward, done, {}

        reward = 0
        done = False

        # energy penalty
        energy_penalty = np.linalg.norm(action)
        energy_penalty *=-0.0025
        reward += energy_penalty
        if not self.agent.carrying:
            if self.near(self.box1):
                box_vec = self.box1.pos - self.agent.pos
                box_vec = [box_vec[0], box_vec[2]]
                dir_vec = [self.agent.dir_vec[0], self.agent.dir_vec[2]]
                perp_dist = math.sqrt(np.linalg.norm(box_vec)**2-(np.dot(box_vec, dir_vec)/np.linalg.norm(dir_vec))**2)
                if perp_dist < 0.02:
                    self.agent.carrying = self.box1
        else:
            if self.near(self.box1, self.box2):
                box_vec = self.box2.pos - self.agent.pos
                box_vec = [box_vec[0], box_vec[2]]
                dir_vec = [self.agent.dir_vec[0], self.agent.dir_vec[2]]
                perp_dist = math.sqrt(np.linalg.norm(box_vec)**2-(np.dot(box_vec, dir_vec)/np.linalg.norm(dir_vec))**2)
                if perp_dist < 0.02:
                    reward += 1
                    done = True

        obs = [obs, [self.agent.pos[0], self.agent.pos[2], self.agent.dir, int(self.agent.carrying!=None)]]

        return obs, reward, done, {}

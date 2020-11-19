import os

import numpy as np
from flow.controllers import ContinuousRouter, RLController
from flow.core.params import InitialConfig, SimParams
from flow.core.params import NetParams
from flow.core.params import SumoParams, EnvParams, \
    SumoCarFollowingParams, VehicleParams
from flow.core.params import TrafficLightParams
from flow.envs import Env
from flow.networks import Network
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

ADDITIONAL_ENV_PARAMS = {
    "max_actors_in_frame": 10,
    "actors_edge_id": 'hwy_in',
    "host_edge_id": 'on_ramp',
    "target_edge_id": 'hwy_out',
    "cut_in_allowed_headway": 1.0,
    "penalty_score": 0,
    "success_score": 1,
    "maximal_velocity": 25.0,
    "road_length": 300,
    "target_velocity": 10
}


class SimpleNetwork(Network):
    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):
        initial_config.spacing = "custom"
        super().__init__(name, vehicles, net_params, initial_config, traffic_lights)

    @classmethod
    def new(cls):
        vehicles = VehicleParams()
        vehicles.add(
            veh_id="rl",
            acceleration_controller=(RLController, {}),
            car_following_params=SumoCarFollowingParams(speed_mode="aggressive", min_gap=0),
            routing_controller=(ContinuousRouter, {}),
            num_vehicles=1)

        net_params = NetParams(template=os.path.join(os.path.dirname(__file__), 'simple.net.xml'))
        net_params.no_internal_links = False

        return cls(name='SimpleNetwork', vehicles=vehicles, net_params=net_params)

    @staticmethod
    def gen_custom_start_pos(cls, net_params, initial_config, num_vehicles):
        return [('road', 0)], [0]

    def specify_routes(self, net_params):
        rts = {
            "road": ["road"]
        }
        return rts

    @staticmethod
    def is_failure(env):
        return env.k.vehicle.get_position(env.host_id) > env.k.network.edge_length('road') * 0.9 \
               and env.k.vehicle.get_speed(env.host_id) > env.env_params.additional_params["target_velocity"]

    @staticmethod
    def is_success(env):
        return env.k.vehicle.get_position(env.host_id) > env.k.network.edge_length('road') * 0.9 and \
               env.k.vehicle.get_speed(env.host_id) <= env.env_params.additional_params["target_velocity"]


class SimpleEnv(Env):
    network_class = SimpleNetwork

    def __init__(self, env_params: EnvParams, sim_params: SimParams, network: Network):
        super().__init__(env_params, sim_params, network)

        self.action_horizon = 3  # sec

    @classmethod
    def new(cls, network, render=False):
        env_params = EnvParams(
            additional_params=ADDITIONAL_ENV_PARAMS,
            sims_per_step=10,
            warmup_steps=0)

        sim_params = SumoParams(render=render)

        return cls(env_params, sim_params, network)

    @classmethod
    def get_instance(cls, render=False):
        network = cls.network_class.new()
        env = cls.new(network=network, render=render)
        return env

    @property
    def host_id(self):
        return self.k.vehicle.get_rl_ids()[0]

    ##############################################
    ### Overrides - environment implementation ###
    ##############################################

    @property
    def action_space(self):
        """ 6 commands available: reaching [0%, 20%, ..., 100%] of max velocity
            acceleration will be calculated by attempting to reach the target velocity in a <self.action_horizon> secs
        """
        return Discrete(n=5)

    @property
    def observation_space(self):
        """
        state space includes:
        First Box   - host vehicle's (distance to junction, velocity, acceleration)
        Second Box  - counter for actors (used to crop Third Box)
        Third Box   - all actors on the target lane, ordered by proximity to the junction, each one has:
                    (distance to junction,
                    velocity - longitudinal,
                    acceleration - longitudinal)
        Fourth Box - 1d array with:
                    - maximal distance to junction (actors lane)
                    - maximal velocity
        """
        return Box(low=0, high=self.env_params.additional_params["maximal_velocity"], shape=(2,), dtype=np.float32)

    def get_state(self):
        edge_len = self.k.network.edge_length('road')
        host_s = self.k.vehicle.get_position(self.host_id)

        host_v = self.k.vehicle.get_speed(self.host_id)
        max_v = self.env_params.additional_params["maximal_velocity"]

        return np.array([host_s/edge_len, host_v/max_v])

    def compute_reward(self, rl_actions, **kwargs) -> (float, bool):
        return self.network.is_success(self) * 10000 / self.time_counter

    def _apply_rl_actions(self, rl_actions):
        current_velocity = self.k.vehicle.get_speed(self.host_id)

        target_velocity = self.env_params.additional_params["maximal_velocity"] * float(
            rl_actions) / (self.action_space.n - 1)
        acceleration = (target_velocity - current_velocity) / self.action_horizon

        self.k.vehicle.apply_acceleration([self.host_id], [acceleration])

    def step(self, rl_actions):
        next_observation, reward, done, infos = super().step(rl_actions)
        infos["host_velocity"] = next_observation[0]
        return next_observation, reward, self.network.is_success(self) or self.network.is_failure(self), infos

import opensim as osim
import numpy as np
from build_hopper import build_hopper

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation

from stable_baselines3 import PPO
import wandb
from wandb.integration.sb3 import WandbCallback

from time import time

# Based on https://gymnasium.farama.org/environments/mujoco/hopper/

class HopperEnv(gym.Env):
    def __init__(self):
        super(HopperEnv, self).__init__()
        
        # Load the hopper model and create an initial state.
        self._model = build_hopper(controller='constant')
        self._state = self._model.initSystem()
        self._nq = self._state.getNQ()
        self._nu = self._state.getNU()
        self._dt = 0.001
        self._time = 0.0
        self._num_steps = 0
        self._noise_scale = 5e-3

        # Define the hopper's (i.e., agent's) kinematic state. 
        self._q = np.zeros(self._nq)
        self._u = np.zeros(self._nu)
        self._udot = np.zeros(self._nu)
        self._prev_q = np.zeros(self._nq)
        self._prev_u = np.zeros(self._nu)
        self._target_height = 1.0 

        # Define the observation space (e.g., the coordinate values and speeds).
        self._q_bounds = np.array([
            [-90.0*(np.pi / 180.0), 10.0*(np.pi / 180.0)],   # Lower bounds
            [110.0*(np.pi / 180.0), 140.0*(np.pi / 180.0)],  # Upper bounds
        ])
        self._u_bounds = np.array([
            [-50.0, -50.0, -50.0],  # Lower bounds
            [ 50.0,  50.0,  50.0],  # Upper bounds
        ])

        self.observation_space = gym.spaces.Dict(
            {
                "q": gym.spaces.Box(low=self._q_bounds[0], high=self._q_bounds[1], 
                                    dtype=np.float32),
                "u": gym.spaces.Box(low=self._u_bounds[0], high=self._u_bounds[1],
                                    dtype=np.float32),
            }
        )
        
        # Define the action space (e.g., the 'vastus' muscle excitation).
        self.action_space = spaces.Box(low=0.0, high=1.0, 
                                       shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomly set a target height between [0.8 1.2] meters.
        self._target_height = np.random.uniform(0.8, 1.2)

        # Randomly set 'q' and 'u' within their bounds.
        self._q = np.array([1.05, 0.35, 0.75], dtype=np.float32)
        self._u = np.array([0.0, 0.0, 0.0], dtype=np.float32) 
        # Add noise to the initial state.
        self._q += np.random.uniform(-self._noise_scale, self._noise_scale, size=self._nq)
        self._u += np.random.uniform(-self._noise_scale, self._noise_scale, size=self._nu)
        # Reset the time and step count.
        self._time = 0.0
        self._num_steps = 0
        # Reset everything else.
        self._vertical_force = 0.0
        self._com_y_velocity = 0.0
        self._com_y_acceleration = 0.0
        self._prev_q = self._q.copy()
        self._prev_u = self._u.copy()

        observation = self._get_obs()
        info = {'target_height': self._target_height, 'q': self._q, 'u': self._u}
        return observation, info

    def step(self, action):

        # Update the state.
        self._state.setTime(self._time)
        self._state.updQ()[0] = float(self._q[0])
        self._state.updQ()[1] = float(self._q[1])
        self._state.updQ()[2] = float(self._q[2])
        self._state.updU()[0] = float(self._u[0])
        self._state.updU()[1] = float(self._u[1])
        self._state.updU()[2] = float(self._u[2])

        # Set the previous pelvis y position.
        self._prev_q = self._q.copy()
        self._prev_u = self._u.copy()

        # Update the control.
        controller = self._model.updComponent('/controllerset/prescribedcontroller')
        constant = osim.Constant.safeDownCast(controller.upd_ControlFunctions().get(0))
        constant.setValue(float(action[0]))

        # Integrate the model for a small time step.
        manager = osim.Manager(self._model)
        manager.setIntegratorMinimumStepSize(self._dt)
        manager.setIntegratorMaximumStepSize(self._dt)
        manager.initialize(self._state)
        manager.integrate(self._time + self._dt)

        # Get the updated  state after integration.
        state = manager.getState()
        self._model.realizeAcceleration(state)

        # Update the state variables.
        self._q = np.array([state.getQ()[i] for i in range(self._nq)])
        self._u = np.array([state.getU()[i] for i in range(self._nu)])
        self._udot = np.array([state.getUDot()[i] for i in range(self._nu)])

        # Update the time and step count.
        self._time += self._dt
        self._num_steps += 1

        # Get observations, rewards, termination status, and info.
        obs = self._get_obs()
        terminated = np.abs(self._q[0] - self._target_height) < 0.001
        reward = 1 if terminated else 0
        truncated = self._num_steps >= 5000 or self._q[0] < 0.5
        info = {'q': self._q, 'u': self._u}
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        return { "q": self._q[1:], "u": self._u }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or evaluate the Hopper environment.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate"],
        default="train",
        help="Mode to run the script in: 'train' or 'evaluate'.",
    )
    args = parser.parse_args()

    if args.mode == "train":
        config = {
            "total_timesteps": 1_000_000,
            "policy_type": "MlpPolicy",
            "env_name": "HopperEnv",
        }

        run = wandb.init(
            project="opensim-hopper",  # change this to your project name
            config=config,
            sync_tensorboard=True,  # optional: if you use tensorboard
        )

        env = HopperEnv()
        wrapped_env = FlattenObservation(env)
        model = PPO(config['policy_type'], 
                    wrapped_env, 
                    device="cpu", 
                    verbose=1, 
                    tensorboard_log=f"runs/{run.id}")      

        start_time = time()  
        model.learn(
            total_timesteps=config['total_timesteps'], 
            callback=WandbCallback(
                gradient_save_freq=100,
                model_save_path="models/",
                verbose=2,
            )
        )
        end_time = time()
        print(f"Training completed in {end_time - start_time:.2f} seconds.")
        model.save("ppo_hopper")
        run.finish()

    elif args.mode == "evaluate":
        model = PPO.load("ppo_hopper")
        env = HopperEnv()
        wrapped_env = FlattenObservation(env)
        
        obs, info = wrapped_env.reset()
        print('Target height:', info['target_height'])
        for _ in range(5000):
            action, _states = model.predict(obs)
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            print(f'y-pos: {info['q'][0]:.2f} | y-vel: {info['u'][0]:.2f} | Action: {action[0]:.2f} | Reward: {reward:.2f}')
            
            if terminated or truncated:
                break


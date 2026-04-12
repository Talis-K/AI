import sys
sys.path.append('..')
import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import simple_driving
import time
import os
import math

# ========================================================
# Reward Function Configuration Parameters
# ========================================================
OBSTACLE_PENALTY = -100.0
GOAL_REWARD = 100.0
STEP_PENALTY = -0.01
PROGRESS_REWARD_SCALE = 1.0
MINIMUM_SAFE_DISTANCE = 1.0

def custom_observation(client, car_pos, car_orn, goal_pos, goal_orn, obstacle_pos, has_obstacle):
    """
    Computes the observation array for the neural network.
    
    Args:
        client (bullet_client): The PyBullet physics client.
        car_pos (list of float): The global [x, y, z] position of the car.
        car_orn (list of float): The global [x, y, z, w] quaternion orientation of the car.
        goal_pos (list of float): The global [x, y, z] position of the goal.
        goal_orn (list of float): The global [x, y, z, w] quaternion orientation of the goal.
        obstacle_pos (tuple of float or None): The global (x, y) position of the obstacle, if it exists.
        has_obstacle (bool): True if an obstacle spawned this episode, False otherwise.
        
    Returns:
        list of float: The computed observation state array.
    """
    # ========================================================
    # TODO: Calculate the Observation Space for the Neural Network
    # By default, PyBullet returns global coordinates (X, Y).
    # You must convert the goal position and obstacle position into 
    # RELATIVE coordinates (where is the object relative to the car?)
    # HINT: Look up client.invertTransform and client.multiplyTransforms
    # ========================================================

    inv_pos, inv_orn = client.invertTransform(car_pos, car_orn)

    relative_goal_pos, _ = client.multiplyTransforms(inv_pos, inv_orn, goal_pos, goal_orn)
    goal_x, goal_y = relative_goal_pos[0], relative_goal_pos[1]
    
    if has_obstacle is True and obstacle_pos is not None:
        # Env passes (x, y); PyBullet needs [x, y, z]. Cylinder center is z=0.5 (see resources/obstacle.py).
        ox, oy = float(obstacle_pos[0]), float(obstacle_pos[1])
        obstacle_world_pos = [ox, oy, 0.5]
        relative_obstacle_pos, _ = client.multiplyTransforms(
            inv_pos, inv_orn, obstacle_world_pos, [0, 0, 0, 1]
        )
        obs_x, obs_y = relative_obstacle_pos[0], relative_obstacle_pos[1]
        observation = [goal_x, goal_y, obs_x, obs_y, 1.0]
    else:
        observation = [goal_x, goal_y, 0.0, 0.0, 0.0]

    return observation


def custom_reward(car_pos, goal_pos, obstacle_pos, has_obstacle, prev_dist_to_goal, dist_to_goal, reached_goal):
    """
    Computes the scalar reward for the current timestep.
    
    Args:
        car_pos (list of float): The global [x, y, z] position of the car.
        goal_pos (list of float): The global [x, y, z] position of the goal.
        obstacle_pos (tuple of float or None): The global (x, y) position of the obstacle, if it exists.
        has_obstacle (bool): True if an obstacle spawned this episode.
        prev_dist_to_goal (float): The distance to the goal in the previous physics frame.
        dist_to_goal (float): The distance to the goal in the current physics frame.
        reached_goal (bool): True if the car reached the goal this frame.
        
    Returns:
        float: The exact mathematical reward for this timestep.
    """
    # ========================================================
    # TODO: Write your reward function
    # 1. Give the agent a basic STEP_PENALTY every frame
    # 2. Reward it for getting closer to the goal
    # 3. Give it a large GOAL_REWARD if it reached_goal
    # 4. Give it a large OBSTACLE_PENALTY if it gets too close to the obstacle
    # 
    # HINT: If your agent has trouble avoiding the obstacle and drives right into it,
    # you can try adding a "proximity penalty" (repulsive field). If the car gets 
    # within a certain distance of the obstacle, start gradually subtracting reward!
    # ========================================================

    # --- Progress reward ---
    if prev_dist_to_goal is not None:
        progress_reward = PROGRESS_REWARD_SCALE * (prev_dist_to_goal - dist_to_goal)
    else:
        progress_reward = 0.0

    # --- Goal reached ---
    if reached_goal:
        return GOAL_REWARD

    # --- Heading reward ---
    # Reward the car for moving TOWARD the goal, not just getting closer over time.
    # We compare the direction of movement to the direction of the goal.
    if prev_dist_to_goal is not None and prev_dist_to_goal > 0:
        # Vector from car to goal
        dx_goal = goal_pos[0] - car_pos[0]
        dy_goal = goal_pos[1] - car_pos[1]
        goal_dist = math.sqrt(dx_goal**2 + dy_goal**2)

        if goal_dist > 0.01:  # avoid division by zero near goal
            # Normalised direction to goal
            goal_dir_x = dx_goal / goal_dist
            goal_dir_y = dy_goal / goal_dist

            # How much the agent closed distance this step (signed)
            # dot product of movement direction with goal direction
            movement = prev_dist_to_goal - dist_to_goal
            heading_reward = 0.3 * movement * (goal_dir_x + goal_dir_y) / 2.0
        else:
            heading_reward = 0.0
    else:
        heading_reward = 0.0

    # --- Obstacle logic ---
    if has_obstacle and obstacle_pos is not None:
        dist_to_obs = math.sqrt(
            (car_pos[0] - obstacle_pos[0]) ** 2 +
            (car_pos[1] - obstacle_pos[1]) ** 2
        )

        # Hard collision penalty
        if dist_to_obs < MINIMUM_SAFE_DISTANCE:
            return OBSTACLE_PENALTY

        # Proximity repulsive field
        PROXIMITY_RADIUS = 3.0
        if dist_to_obs < PROXIMITY_RADIUS:
            proximity_penalty = -2.0 * (PROXIMITY_RADIUS - dist_to_obs) / PROXIMITY_RADIUS
        else:
            proximity_penalty = 0.0
    else:
        proximity_penalty = 0.0

    # --- Assemble final reward ---
    reward = STEP_PENALTY + progress_reward + heading_reward + proximity_penalty
    return reward

# You can change these variables for more training steps or if you have a powerful CPU:
TOTAL_TIMESTEPS = 75000      # define the number of steps used during the training
N_ENVS = 1                   # number of processor core used for multithreading

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on SimpleDriving-v0")
    parser.add_argument(
        "--resume-model",
        type=str,
        default=None,
        help="Optional path to an existing PPO checkpoint (without or with .zip) to continue training.",
    )
    parser.add_argument(
        "--save-name",
        type=str,
        default="ppo_model_heading",
        help="Output model base name (without .zip).",
    )
    args = parser.parse_args()

    env_kwargs = {
        "renders": False, 
        "isDiscrete": False,
        "reward_callback": custom_reward,
        "observation_callback": custom_observation,
    }
    env = make_vec_env(
        "SimpleDriving-v0", 
        n_envs=N_ENVS, 
        vec_env_cls=SubprocVecEnv, 
        env_kwargs=env_kwargs,
        vec_env_kwargs={"start_method": "spawn"},
    )

    # ========================================================
    # TODO: Implement PPO using stable_baselines3!
    # 1. Instantiate the PPO agent ("MlpPolicy")
    #    HINT: SB3's default PPO parameters are optimized for long tasks. 
    #    For our short driving environment, training will be painfully slow
    #    unless you override these hyperparameters during instantiation:
    #      - learning_rate=0.0003
    #      - n_steps=512
    #      - batch_size=256
    #      - ent_coef=0.01
    #    You can play around with different parameters, change the number of
    #    TOTAL_TIMESTEPS, learning_rate, etc.
    # 2. Tell the agent to log metrics to a local tensorboard directory.
    # 3. Call agent.learn(total_timesteps=TOTAL_TIMESTEPS)
    # 4. Save the agent when done
    # 
    # Optional: to speed up the training and avoiding to start from scratch every time, 
    # you can reload previously trained models 
    # (look up Curriculum Learning/Transfer Learning to learn more about this)
    # 
    # If you do, keep track of the previous reward function you used for the VIVA 
    # (or retrain from scratch to make sure your function works properly)
    # ========================================================
    

    tensorboard_dir = "./ppo_tensorboard/"
    os.makedirs(tensorboard_dir, exist_ok=True)

    # Curriculum learning option: load previous policy and continue training.
    if args.resume_model:
        resume_path = args.resume_model
        if not resume_path.endswith(".zip"):
            resume_path = resume_path + ".zip"
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Resume model not found: {resume_path}")

        print(f"Resuming training from: {os.path.abspath(resume_path)}")
        agent = PPO.load(
            args.resume_model,
            env=env,
            tensorboard_log="./ppo_tensorboard/",
            verbose=1,
        )
    else:
        print("Starting training from scratch.")
        agent = PPO(
            "MlpPolicy",
            env,
            learning_rate=0.0003,
            n_steps=1024,
            batch_size=256,
            ent_coef=0.001,
            tensorboard_log="./ppo_tensorboard/",
            verbose=1,
        )

    agent.learn(total_timesteps=TOTAL_TIMESTEPS)
    agent.save(args.save_name)

    print(f"TensorBoard: tensorboard --logdir {os.path.abspath(tensorboard_dir)}")

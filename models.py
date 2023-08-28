# common library
import pandas as pd
import numpy as np
import torch as th
import time
import wandb
import gym
import ipdb


import config
from preprocessor.preprocessors import FeatureEngineer, data_split, series_decomposition

from env.env_portfolio import StockPortfolioEnv

from A2C.a2c import A2C
from tools.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
# from stable_baselines3.common.vec_env import DummyVecEnv


MODELS = {"a2c": A2C} #can be extended with other RL algos


MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}

NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}


class DRLAgent:
    """Provides implementations for DRL algorithms

    Attributes
    ----------
        env: gym environment class
            user-defined class

    Methods
    -------
        train_PPO()
            the implementation for PPO algorithm
        train_A2C()
            the implementation for A2C algorithm
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    @staticmethod
    def DRL_prediction(model, environment):
        test_env, test_state = environment.get_sb_env()
        def softmax_normalization(actions):
            numerator = np.exp(actions)
            denominator = np.sum(np.exp(actions))
            softmax_output = numerator / denominator
            return softmax_output
        """make a prediction"""
        account_memory = []
        actions_memory = []
        episode_start = 1
        test_env.reset()
        fuse_action_return, action1_return, action2_return, action3_return, action4_return, action5_return = [1], [1], [1], [1], [1], [1]
        for i in range(len(environment.df.index.unique())):
            test_multi_obs = series_decomposition(test_state[0], config.MAX_LEVEL) 
            if episode_start == 1:
                last_action = th.zeros([config.AGENT_NUM, test_state.shape[2]])
            action, agent_actions = model.predict(test_multi_obs, last_action, th.eye(config.AGENT_NUM))
            # print(fuse_action_return[-1])
            # wandb.log({
            #     "fuse":fuse_action_return[-1],
            #     "action1":action1_return[-1],
            #     "action2":action2_return[-1],
            #     "action3":action3_return[-1],
            # })

            fuse_action_return.append(fuse_action_return[-1] + np.sum(((test_state[0][-1,:] / test_state[0][-2,:]) - 1) * softmax_normalization(action)))
            action1_return.append(action1_return[-1] + np.sum(((test_state[0][-1,:] / test_state[0][-2,:]) - 1) * softmax_normalization(agent_actions[0])))
            action2_return.append(action2_return[-1] + np.sum(((test_state[0][-1,:] / test_state[0][-2,:]) - 1) * softmax_normalization(agent_actions[1]))) 
            action3_return.append(action3_return[-1] + np.sum(((test_state[0][-1,:] / test_state[0][-2,:]) - 1) * softmax_normalization(agent_actions[2]))) 
            # action4_return.append(action4_return[-1] + np.sum(((test_state[0][-1,:] / test_state[0][-2,:]) - 1) * softmax_normalization(agent_actions[3]))) 
            # action5_return.append(action5_return[-1] + np.sum(((test_state[0][-1,:] / test_state[0][-2,:]) - 1) * softmax_normalization(agent_actions[4]))) 
            

            test_state, rewards, dones, info = test_env.step(action)
            episode_start = dones
            last_action = th.from_numpy(agent_actions)
            if i == (len(environment.df.index.unique()) - 2):
              account_memory = test_env.env_method(method_name="save_asset_memory")
              actions_memory = test_env.env_method(method_name="save_action_memory")
            if dones[0]:
                print("hit end!")
                break
        # wandb.log({"return lines" : wandb.plot.line_series(
        #     xs=range(len(fuse_action_return)),
        #     ys=[fuse_action_return,action1_return,action2_return,action3_return],
        #     keys=["Fuse action","Agent 1","Agent 2","Agent 3"],
        #     title="Return")})
        return account_memory[0], actions_memory[0]

    def __init__(self, env):
        self.env = env

    def get_model(
        self,
        model_name,
        # policy="MlpPolicy",
        policy_kwargs=None,
        model_kwargs=None,
        verbose=1,
    ):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS[model_name]

        if "action_noise" in model_kwargs:
            n_actions = self.env.action_space.shape[-1]
            model_kwargs["action_noise"] = NOISE[model_kwargs["action_noise"]](
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )
        print(model_kwargs)
        model = MODELS[model_name](
            # policy=policy,
            env=self.env,
            tensorboard_log=f"{config.TENSORBOARD_LOG_DIR}/{model_name}",
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            **model_kwargs,
        )
        return model

    def train_model(self, model, tb_log_name, total_timesteps=5000, eval_env=None):
        model = model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name, eval_env=eval_env)
        return model


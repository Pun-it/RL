from stable_baselines3 import PPO 
import os
from SnakeGame import snakeEnvironment
from torch.utils import tensorboard

modeldir = "models/1693285980"

env = snakeEnvironment()

obs = env.reset()

while True :
    obs = env.reset()
    done = False
    while not done :

        # un comment any one of these 

        model = PPO.load(f"{modeldir}/310000.zip",env = env)  #
        #model = PPO.load(f"{modeldir}/320000.zip",env = env)
        action,_states = model.predict(obs)
        obs,reward,done,info = env.step(action)
from WAB_Environment2 import WAB_Environment2
import pandas as pd
import numpy as np
import torch
import random

num_ostriches = 10

num_wolves = 3

num_bushes = 20

num_entities = num_ostriches + num_wolves + num_bushes


env = WAB_Environment2(20, 20)

env.create_ostriches(num_ostriches)

env.create_wolves(num_wolves)

env.create_bushes(num_bushes)

epochs = 100


def generate_ostrich_action(obs):
    return random.randint(0, 5)


def generate_wolf_action(obs):
    return random.randint(0, 4)


def generate_bush_action(obs):
    return 0

if __name__ == "__main__":

    for i in range(epochs):
        env.reset_environment()
        done = False

        acting_entities = num_ostriches + num_wolves
        print("starting epoch " + str(i))
        while not done:
            # determine the action using the observation and then take it
            i = 0
            print("Ostrich turns")
            while i < num_ostriches:
                dead = False
                obs = env.get_obs(i)
                # this will not learn its just an example
                action = generate_ostrich_action(obs)

                reward, dead = env.take_action(i, action)

                if dead:
                    acting_entities -= 1

                # do backpropagation here using reward

                i += 1

            print("Wolf turns")
            while i < num_ostriches + num_wolves:
                dead = False
                obs = env.get_obs(i)
                # this will not learn its just an example
                action = generate_wolf_action(obs)

                reward, dead = env.take_action(i, action)

                if dead:
                    acting_entities -= 1

                # do backpropagation here using reward

                i += 1
            print("Bush turns")
            while i < num_entities:
                dead = False
                obs = env.get_obs(i)

                action = generate_bush_action(obs)
                reward, dead = env.take_action(i, action)

                i += 1

            if acting_entities == 0:
                done = True


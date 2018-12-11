#!/usr/bin/env python3.6

## IMPORTS ##

import gym
import time
import numpy as np
import matplotlib.pyplot as plt


## CLASSES ##

class PacManBeast:
    def __init__(self, states_dim, disc_factor=0.9, learn_rate=0.1, allowed_actions=np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8]),
                    backward_memory=1, greediness=0.95):

    # From class instantiation
        self.states_dim = states_dim
        self.disc_factor = disc_factor
        self.learn_rate = learn_rate
        self.allowed_actions = allowed_actions
        self.backward_memory = backward_memory
        self.greediness = greediness

    # Temporal abstraction
        self.QEvol = np.empty([(1+self.backward_memory), np.shape(allowed_actions)[0]])
        self.QEvol[:] = -100
        self.ObsEvol = np.empty([(1+self.backward_memory), self.states_dim])
        self.ObsEvol[:] = np.nan
        self.pleasure = np.nan
        self.action_lastcall = np.nan

    # Feature space
        self.dim_of_feat_space = 5
        self.params = np.random.randn(self.dim_of_feat_space)
        self.features_lastcall = np.random.randn(self.dim_of_feat_space)
        self.features_lastcall[:] = np.nan

# Features
    def feat_01(self, state, action):
        return 1.0

    def feat_02(self, state, action):
        return 1.0

    def feat_03(self, state, action):
        return 1.0

    def feat_04(self, state, action):
        return 1.0

    def feat_05(self, state, action):
        return 1.0

    def features(self, state, action):
        self.features_lastcall = np.asarray([self.feat_01(state, action), self.feat_02(state, action), self.feat_03(state, action), self.feat_04(state, action), self.feat_05(state, action)])
        return self.features_lastcall

# Functions
    def QFunc(self, state, action, params):
        return np.dot(self.features(state, action), params)

    def observe(self, state_vector):
        for i in range(self.backward_memory, 0, -1):
            self.ObsEvol[i] = self.ObsEvol[i-1]
            self.ObsEvol[0] = state_vector

    def enjoy(self, env_reward):
        for i in range(self.backward_memory, 0, -1):
            self.pleasure = env_reward

    def predict(self):
        for i in range(np.shape(self.allowed_actions)[0]):
            self.QEvol[0][i] = self.QFunc(self.ObsEvol[0], self.allowed_actions[i], self.params)

    def policy_choice(self):
        if np.random.random() > self.greediness:
            self.action_lastcall = np.random.random_integers(0, np.shape(self.allowed_actions)[0]-1)
        else:
            self.action_lastcall = np.argmax(self.QEvol[0])
        return self.action_lastcall

    def update_params(self):
        difference = self.pleasure - self.QEvol[1][self.action_lastcall] + self.disc_factor*np.max(self.QEvol[0])
        self.params += self.learn_rate*difference*self.features_lastcall


## GAMEPLAY ##

env = gym.make('MsPacman-ram-v0')
state = env.reset()
done = False
time.sleep(0.03)
env.render()

myPacman = PacManBeast(np.shape(state)[0])

myPacman.observe(state)
myPacman.predict()

while not(done):
    time.sleep(0.03)
    env.render()

    state, reward, done, info = env.step(myPacman.policy_choice())

    myPacman.observe(state)
    myPacman.enjoy(reward)
    myPacman.predict()
    myPacman.update_params()

env.close()

import random
from threading import Thread

import tensorflow as tf

from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import timeit
import numpy as np
import gym.spaces
import multiprocessing
import matplotlib.pyplot as plt
import gym
from collections import deque

from keras import backend as K
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=18, inter_op_parallelism_threads=18)))

class BipedalAgent(object):

    def __init__(self, action_space, state_size, joints_number):
        self.action_space = action_space
        self.input_size = state_size

        self.joints_number = joints_number
        self.action_size = 2
        self.output_size = 2

        self.memory = deque(maxlen=2000)
        self.first_hl = 24
        self.second_hl = 24

        self.gamma = 0.75  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(self.first_hl, input_dim=self.input_size, activation='relu'))
        model.add(Dense(self.second_hl, activation='relu'))
        model.add(Dense(self.output_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=10, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    env.seed(0)
    # env.render()

    state_size = env.observation_space.shape[0]
    agent = BipedalAgent(env.action_space, state_size, joints_number=2)

    TRAINING_EPISODES = 200
    ACTIONS_PER_EPISODE = 1000
    REINFORCEMENT_EPISODES = 1000
    reward = 0
    done = False
    batch_size = 64

    for e in range(TRAINING_EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        for action_t in range(ACTIONS_PER_EPISODE):
            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                # print the score and break out of the loop
                #print("episode: {}/{}, score: {}"
                      #.format(e, TRAINING_EPISODES, time_t))

                break
        print("episode: {}/{}, actions: {}"
              .format(e, TRAINING_EPISODES, action_t))
        if(len(agent.memory) > batch_size):
            agent.replay(batch_size)

    print("\n--------- Finished training ---------\n")

    # agent.epsilon = 1.0  # exploration rate

    reward_arr = []
    reward_arr_avg = []

    for e in range(REINFORCEMENT_EPISODES):
        reward_sum = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for action_t in range(ACTIONS_PER_EPISODE):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            reward_sum += reward
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        interval = 20
        reward_arr.append(reward_sum)
        if len(reward_arr) > interval:
            reward_arr_avg.append(np.average(reward_arr[-interval:]))
            plt.plot(reward_arr_avg)
            plt.show()
            plt.pause(0.0001)
        print("episode: {}/{}\t actions: {}\t reward: {}\t e: {:.2}"
              .format(e, REINFORCEMENT_EPISODES, action_t, reward_sum, agent.epsilon))


    env.env.close()

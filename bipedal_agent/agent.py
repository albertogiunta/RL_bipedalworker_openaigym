from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import gym.spaces
import gym

class BipedalAgent(object):

    def __init__(self, action_space, state):
        self.action_space = action_space
        self.state_size = len(state)
        self.action_size = len(action_space)
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, observation, reward, done):
        return self.action_space.sample()


if __name__ == '__main__':
    env = gym.make("BipedalWalker-v2")
    env.seed(0)
    env.render()

    print env.action_space
    state = np.reshape(env.reset(), [1, 24])
    agent = BipedalAgent(env.action_space, state)


    episode_count = 100
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            env.render()
            if done:
                break

    env.env.close()

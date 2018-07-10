from keras import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
import numpy as np
import gym.spaces
import gym

class BipedalAgent(object):

    def __init__(self, action_space, state):
        self.action_space = action_space
        self.state_size = len(state)
        self.action_size = 4
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=24, activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(80, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def learn(self, state, next_state, action, reward):
        target = (reward + self.gamma *
                      np.amax(self.model.predict(next_state)[0]))
        target_f = self.model.predict(state)
        target_f[0][action] = target


        self.model.fit(observation, reward, 1, 0)


    def act(self, observation, reward, done):
        return self.model.predict(observation)


if __name__ == '__main__':
    env = gym.make("BipedalWalker-v2")
    env.seed(0)
    env.render()

    state = np.reshape(env.reset(), [1, 24])
    agent = BipedalAgent(env.action_space, state)

    episode_count = 100
    reward = 0
    done = False

    #learning phase
    state = env.reset()
    for i in range(200):
        action = env.action_space.sample()
        next_state, reward, _, _  = env.step(action)
        next_state = np.reshape(next_state, [1, 24])
        agent.learn(state, next_state, action, reward)
        state = next_state


    for i in range(episode_count):
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            env.render()
            if done:
              break

    #env.env.close()

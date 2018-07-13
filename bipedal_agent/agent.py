import random
from threading import Thread

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

class BipedalAgent(object):

    def __init__(self, action_space, state_size, step_size, joints_number):
        self.action_space = action_space
        self.input_size = state_size

        self.step_size = step_size
        self.joints_number = joints_number
        self.action_size = 2 / self.step_size * joints_number
        self.actions_per_joint = int(2 / self.step_size)
        self.output_size = self.actions_per_joint * self.joints_number

        self.memory = deque(maxlen=5000)
        self.first_hl = 96
        self.second_hl = 96

        self.gamma = 0.75  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.5
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

    def take_four_maxes(self, array):
        # Matrice 4*3, dove ogni riga contiene [ reward, azione ottima, indice dell'azione ] per ogni giunto
        maxes = np.zeros((self.joints_number, 3))
        half = len(array) / 2
        packeted_output = np.reshape(array, (self.joints_number, self.actions_per_joint))
        for joint_index, a in enumerate(packeted_output):

            # Indice sull'array di 20 dell'elemento con reward massima
            max_index = np.argmax(a)

            # Salvo la reward massima trovata per il giunto
            maxes[joint_index][0] = a[max_index]

            # Normalizza l'indice considerando solo la meta' positiva dell'intervallo (0-1) per poter ragionare
            # una sola volta a prescindere dal segno
            if max_index < half:
                # Es: l'indice di partenza 1 diventa |1-2|-1 = 0
                partial_index = np.abs(max_index - half) - 1
            else:
                # Nel caso positivo non e' necessario sottrarre l'1  Es: |2-2| = 0
                partial_index = max_index - half

            # Calcolo l'azione media del bucket ottenuto come (lower_bound + upper_bound)/2
            middle_bucket = ((partial_index * self.step_size) + (partial_index * self.step_size + self.step_size)) / 2

            # Se l'indice era dell'intervallo negativo (-1, 0) allora l'azione sara' negativa
            if max_index < half:
                middle_bucket = -middle_bucket

            # Salvo l'azione media come azione ottimale
            maxes[joint_index][1] = middle_bucket
            # Salvo l'indice globale (su 80) dell'azione ottimale
            maxes[joint_index][2] = max_index + (joint_index * max_index)

        return maxes

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # print "random"
            return self.action_space.sample()

        # print "NOT random"
        act_values = self.model.predict(state)
        # Restituiamo solo l'array che definisce l'azione ottima globale
        return np.transpose(self.take_four_maxes(act_values))[1]

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        # pool = multiprocessing.Pool(4)

        for item in minibatch:
            self.train(item)
            # pool.apply_async(self.train, args=(state, action, reward, next_state, done))

        # pool.close()
        # pool.join()

        if self.epsilon > self.epsilon_min:
            # print(self.epsilon)
            self.epsilon *= self.epsilon_decay

    def train(self, item):
        (state, action, reward, next_state, done) = item
        # Inizializza q_reward come un array di 4 elementi uguali
        q_reward = [reward] * self.joints_number

        # Prendiamo le reward dell'azione ottima allo stato successivo
        self.model.predict(next_state)

        maxes = self.take_four_maxes(self.model.predict(next_state)[0])

        if not done:
            # Aggiorniamo la nostra q_reward con la Q function
            q_reward = (reward + self.gamma * (np.transpose(maxes)[0]))

        # Prendo l'output che la rete mi da attualmente
        nn_output = self.model.predict(state)

        # Modifico l'output ottenuto con la q_reward calcolata
        for index, max in enumerate(maxes):
            nn_output[0][int(max[2])] = q_reward[index]

        # Addestro la rete con l'output ricalcolato
        self.model.fit(state, nn_output, epochs=10, verbose=0)


if __name__ == '__main__':
    env = gym.make("BipedalWalker-v2")
    env.seed(0)
    # env.render()

    state_size = env.observation_space.shape[0]
    agent = BipedalAgent(env.action_space, state_size, step_size=0.05, joints_number=4)

    TRAINING_EPISODES = 200
    ACTIONS_PER_EPISODE = 250
    REINFORCEMENT_EPISODES = 1000
    reward = 0
    done = False
    batch_size = 32

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
            # env.render()
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

        interval = 10
        reward_arr.append(reward_sum)
        if len(reward_arr) > interval:
            reward_arr_avg.append(np.average(reward_arr[:-interval]))
            plt.plot(reward_arr_avg)
            plt.show()
            plt.pause(0.0001)
        print("episode: {}/{}\t actions: {}\t reward: {}\t e: {:.2}"
              .format(e, REINFORCEMENT_EPISODES, action_t, reward_sum, agent.epsilon))


    env.env.close()

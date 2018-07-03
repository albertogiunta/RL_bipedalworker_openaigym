import gym
from keras.models import Sequential

class BipedalAgent(object):

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        print "TODO"


if __name__ == '__main__':
    env = gym.make("BipedalWalker-v2")
    env.seed(0)
    env.render()

    agent = BipedalAgent(env.action_space)

    episode_count = 100
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break

    env.env.close()

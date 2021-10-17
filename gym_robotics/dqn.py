import gym_robot
import gym
import numpy as np
import random
from gym import error, spaces
from gym import utils
from gym import spaces, logger
from gym.utils import seeding
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
#%matplotlib inline
import matplotlib.pyplot as plt
from collections import deque
import sys
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#Number of episodes
episodes=200

class DQN():
    def __init__(self):
        self.load_model=True
        self.action_space=[0,1,2,3,4,5,6]
        self.action_size=7
        self.state_size= 2
        self.discount_factor=0.99
        self.learning_rate=0.001
        #Exploration Exploitation trade off
        self.epsilon=1.0
        self.epsilon_decay=0.999
        self.epsilon_min=0.01

        self.train_start = 1000

        self.batch_size = 64
        #Build models
        self.model=self.build_model()
        self.target_model=self.build_model()
        #set target models parameters to parameters of model
        self.target_model.set_weights(self.model.get_weights())
        #episode memeory
        self.memory = deque(maxlen=2000)
        if self.load_model:
            self.epsilon=0.0001
            self.model.load_weights('dqn.h5')

    def build_model(self):
        model = Sequential()
        model.add(Dense(30, input_dim=self.state_size, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
    # Policy
    def get_action(self,state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = np.float32(state)
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        target = self.model.predict(state)[0]
        batch_size=self.batch_size
        mini_batch = random.sample(self.memory, batch_size)
        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []
        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])
        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target)
        for i in range(self.batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    np.amax(target_val[i]))
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)

if __name__ == '__main__' :

    env = gym.make('gym_robot-v0')
    agent=DQN()
    plt.ion()
    scores, episode = [], []
    for e in range(episodes):
        done=False
        score = 0
        state=env.reset()
        state = np.reshape(state, [1, 2])
        while not done:
            #sample next action
            action=agent.get_action(state)
            action1=agent.action_space[action]
            next_state,reward,done,info=env.step(action1)
            next_state = np.reshape(next_state, [1, 2])

            agent.append_sample(state, action, reward, next_state, done)

            agent.train_model()
            state=next_state
            score+=reward
            if done:
                #set target models parameters to parameters of model
                agent.target_model.set_weights(agent.model.get_weights())
                scores.append(score)
                episode.append(e)
                #plot graph for debugging
                if not agent.load_model:
                    plt.plot(episode, scores)
                    plt.draw()
                    plt.savefig('dqn.png')
                    plt.pause(0.1)
                    plt.show()
                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon)
        #save weights
        agent.model.save_weights('dqn.h5')

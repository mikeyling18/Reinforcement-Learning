import gym
from gym.wrappers import Monitor
import numpy as np
import random
import tensorflow as tf
from collections import deque
import os


# Initialize Hyper-Parameters

gamma = 0.99                    #discount rate

hidden_layer_size1 = 512
hidden_layer_size2 = 512
hidden_layer_size3 = 512

alpha = 0.0005                   #learning rate
alpha_decay = 1                 #learning rate decay

number_of_episodes = 3000
memory_capacity = 100000
minibatch_size = 1000

eps_init = 0.99                  #starting epsilon value
eps_final = 0.05                #ending epsilon value
steps_needed_to_start_exp_decay = 3000      #number of steps it will take to decrease epsilion from init to final values
eps_decay_rate = 0.95           #exponential rate epsilon will decay after reaching final value
dimensions_in_state = 8
dimensions_in_action = 4

log_dir = os.path.abspath('./logs/test_1')
save_dir = ('logs/model')
monitor_directory = 'logs/monitor'

class Deep_Q_Network():
    def __init__(self):
        self.replay_memory = deque()

        self.epsilon = eps_init

        self.create_network()
        self.training_method()

        self.saver = tf.train.Saver()

        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())


    def save_network(self, network):
        self.saver.save(self.session, network)


    def create_network(self):
        self.state_input = tf.placeholder(tf.float32, [None, dimensions_in_state], name='state')

        # l1 = tf.contrib.layers.fully_connected(self.state_input, hidden_layer_size1, activation_fn = tf.nn.relu)
        # l2 = tf.contrib.layers.fully_connected(l1, hidden_layer_size2, activation_fn = tf.nn.relu)
        # l3 = tf.contrib.layers.fully_connected(l2, hidden_layer_size3, activation_fn = tf.nn.relu)
        #
        # self.Q_values = tf.contrib.layers.fully_connected(l3, dimensions_in_action, activation_fn = None)

        self.w1 = tf.get_variable('w1', [dimensions_in_state, hidden_layer_size1])
        self.b1 = tf.Variable(tf.constant(0.01,  shape=[hidden_layer_size1, ]))

        self.w2 = tf.get_variable('w2', [hidden_layer_size1, hidden_layer_size2])
        self.b2 = tf.Variable(tf.constant(0.01,  shape=[hidden_layer_size2, ]))

        self.w3 = tf.get_variable('w3', [hidden_layer_size2, hidden_layer_size3])
        self.b3 = tf.Variable(tf.constant(0.01,  shape=[hidden_layer_size3, ]))

        self.output_layer_weights = tf.get_variable('output_layer', [hidden_layer_size3, dimensions_in_action], initializer = tf.random_normal_initializer(0,1))
        self.output_layer_biases = tf.Variable(tf.constant(0.01, shape=[dimensions_in_action, ]))

        l1 = tf.nn.relu(tf.matmul(self.state_input, self.w1) + self.b1)
        l2 = tf.nn.relu(tf.matmul(l1, self.w2) + self.b2)
        l3 = tf.nn.relu(tf.matmul(l2, self.w3) + self.b3)

        self.Q_values = tf.matmul(l3, self.output_layer_weights) + self.output_layer_biases

    def get_action(self, state_input, greedy):
        Q_value = self.Q_values.eval(feed_dict={self.state_input: [state_input]})[0]

        if (self.epsilon > eps_final):
            self.epsilon = self.epsilon - ((eps_init - eps_final) / steps_needed_to_start_exp_decay)
        else:
            self.epsilon = self.epsilon * eps_decay_rate

        if greedy == 1:
            if np.random.random() <= self.epsilon:
                return random.randint(0, dimensions_in_action-1)
            else:
                return np.argmax(Q_value)
        else:
            return np.argmax(Q_value)

    def training_method(self):
        self.action_input = tf.placeholder(tf.float32, [None, dimensions_in_action])
        self.Y = tf.placeholder(tf.float32, [None])
        Q_values_final = tf.reduce_sum(tf.multiply(self.Q_values, self.action_input),reduction_indices = 1)

        self.loss = tf.reduce_mean(tf.square(self.Y - Q_values_final))
        self.optimizer = tf.train.AdamOptimizer(0.00005).minimize(self.loss)


    def train_neural_network(self, S_batch, a_batch, r_batch, S_prime_batch, random_minibatch):
        Q_batch = self.Q_values.eval(feed_dict = {self.state_input: S_prime_batch})
        Y_j = []
        for sample in range(0,minibatch_size):
            terminal = random_minibatch[sample][4]
            if terminal:
                # set Y_j = r_j for terminal theta_j+1
                Y_j.append(r_batch[sample])
            else:
                # set Y_j = r_j + gamma * max(Q(theta_j+1, a'; theta)) for non-terminal theta_j+1
                Y_j.append(r_batch[sample] + gamma * np.max(Q_batch[sample]))

        feed_dict = {self.Y: Y_j,
                     self.action_input: a_batch,
                     self.state_input: S_batch}
        self.session.run(self.optimizer, feed_dict)


    def initialize_replay_memory(self, env):
        for episode in range(1000):
            S = env.reset()
            finished = False
            while not finished:
                action = random.randint(0, dimensions_in_action-1)
                action_one_hot = np.zeros(dimensions_in_action)
                action_one_hot[action] = 1
                S_prime, r, finished, _ = env.step(action)
                self.replay_memory.append((S, action_one_hot, r, S_prime, finished))
                S = S_prime
                if len(self.replay_memory) > memory_capacity:
                    self.replay_memory.popleft()
            if episode%100 == 0:
                print('Episode ', episode, ' Replay Memory Size: ', len(self.replay_memory) )

    # def store_transition_in_replay_memory(self, S, a, r, S_prime, finished):
    #     action = random.randint(0, dimensions_in_action-1)
    #     action_one_hot = np.zeros(dimensions_in_action)
    #     action_one_hot[action] = 1
    #     self.replay_memory.append(S, a, r, S_prime, finished)
    #     instances_in_replay_memory = len(self.replay_memory)
    #     if (instances_in_replay_memory > memory_capacity):
    #         self.replay_memory.popleft()
    #
    #     if (instances_in_replay_memory > minibatch_size):
    #         self.train_neural_network()


# initialize game parameter and stuff
env = gym.envs.make("LunarLander-v2")
spaceship = Deep_Q_Network()


# set seeds to 1
env.seed(1)
np.random.seed(1)
steps_in_experience = 1000
num_test_episodes = 20
max_steps_in_training_episode = 1000
# The following section of the code is an implementation of Algorithm 1 in DeepMind's Atari paper
total_reward = 0
spaceship.initialize_replay_memory(env)     # Initialize replay memory

env = Monitor(env, monitor_directory, force=True)
rewards_for_all_episodes = 0
for episode in range(number_of_episodes):
    S = env.reset()
    finished = False
    reward_per_episode = 0
    steps = 1

    for step in range(steps_in_experience):
    # while not finished:
        # env.render()
        a = spaceship.get_action(S, greedy=1)
        S_prime, r, finished, _ = env.step(a)
        action_one_hot = np.zeros(dimensions_in_action)
        spaceship.replay_memory.append((S, action_one_hot, r, S_prime, finished)) # Store transition (S, a, r, S_prime) into replay memory
        instances_in_replay_memory = len(spaceship.replay_memory)
        if (instances_in_replay_memory > memory_capacity):
            spaceship.replay_memory.popleft()
        if (instances_in_replay_memory > minibatch_size):                 # Sample random minibatch of transitions from replay memory
            random_minibatch = random.sample(spaceship.replay_memory, minibatch_size)
            S_batch = [sample[0] for sample in random_minibatch]
            a_batch = [sample[1] for sample in random_minibatch]
            r_batch = [sample[2] for sample in random_minibatch]
            S_prime_batch = [sample[3] for sample in random_minibatch]
            spaceship.train_neural_network(S_batch, a_batch, r_batch, S_prime_batch, random_minibatch)       # Process transition
        reward_per_episode += r
        S = S_prime
        if finished:
            break

    print('Reward/Episode: ', reward_per_episode)
    rewards_for_all_episodes += reward_per_episode
    if (episode%100 == 0):
        spaceship.save_network(save_dir)
        print('Avg Reward of last 100 Episodes: ', rewards_for_all_episodes/100)
        if rewards_for_all_episodes/100 >= 200:
            reward = 0
            print('NOW WE ARE TESTING BABY!!!!')
            for episodes in range(num_test_episodes):
                S = env.reset()
                finished = False
                steps = 1
                # while steps < max_steps_in_training_episode:
                while not finished:
                    # env.render()
                    a = spaceship.get_action(S)
                    S, r, finished, _ = env.step(a)
                    reward = reward + r
                    # steps += 1
                    # if finished:
                    #     break
            avg_reward = reward/num_test_episodes
            print('episode: ', episodes, "Avg Reward: ", avg_reward)
            if (avg_reward >= 200):
                env.close()
                break
        rewards_for_all_episodes = 0



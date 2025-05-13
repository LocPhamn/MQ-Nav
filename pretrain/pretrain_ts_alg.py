"""
Smart Multi-Agent Deep Q-Network (SMADQN) Implementation

This module implements a Deep Q-Network algorithm for multi-agent systems.
It uses a neural network to approximate Q-values and includes features like:
- Experience replay memory
- Target network for stable learning
- Soft parameter updates
- RMSProp optimization

The network architecture consists of:
- Input layer: State features
- Hidden layer 1: 300 units with ReLU
- Hidden layer 2: 200 units with ReLU
- Output layer: Q-values for each action
"""

import numpy as np
import tensorflow as tf
if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

tf.set_random_seed(1)


class SMADQN:
    """
    Smart Multi-Agent Deep Q-Network implementation.
    
    This class implements the DQN algorithm with several improvements
    for multi-agent reinforcement learning.
    """
    
    def __init__(
            self,
            n_actions,
            n_features,
            mode,
            model_path,
            learning_rate=0.01,
            reward_decay=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            tau=0.001, 
    ):
        """
        Initialize the SMADQN agent.
        
        Args:
            n_actions (int): Number of possible actions
            n_features (int): Number of state features
            mode (str): 'train' or 'test' mode
            model_path (str): Path to save/load model parameters
            learning_rate (float): Learning rate for optimization
            reward_decay (float): Discount factor for future rewards
            replace_target_iter (int): Steps between target network updates
            memory_size (int): Size of replay memory
            batch_size (int): Size of training batches
            tau (float): Soft update parameter
        """
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.model_path = model_path+"Net_parameter.ckpt"
        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))  # s, a, r, s'

        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        self.soft_replace_dqn = [[tf.assign(t, (1 - tau) * t + tau * e)] for t, e in zip(t_params, e_params)]
        self.sess = tf.Session()
        if mode == 'train':
            self.sess.run(tf.global_variables_initializer())
        else:
            saver = tf.train.Saver()
            saver.restore(self.sess, self.model_path)

    def _build_net(self):
        """
        Build the neural network architecture.
        Creates both evaluation and target networks with identical structure:
        - Input layer: state features
        - Hidden layer 1: 300 units with ReLU
        - Hidden layer 2: 200 units with ReLU
        - Output layer: Q-values for each action
        """
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')
        with tf.variable_scope('eval_net'):
            c_names, n_l1, n_l2,  w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 300, 200,\
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers
            with tf.variable_scope('l1'):
                self.w1_eval = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer,
                                               collections=c_names)
                self.b1_eval = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, self.w1_eval) + self.b1_eval)
            with tf.variable_scope('l2'):
                self.w2_eval = tf.get_variable('w2', [n_l1, n_l2], initializer=w_initializer, collections=c_names)
                self.b2_eval = tf.get_variable('b2', [1, n_l2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, self.w2_eval) + self.b2_eval)
            with tf.variable_scope('l3'):
                self.w3_eval = tf.get_variable('w3', [n_l2, self.n_actions], initializer=w_initializer,
                                               collections=c_names)
                self.b3_eval = tf.get_variable('b3', [1, self.n_actions], initializer=b_initializer,
                                               collections=c_names)
                self.q_eval = tf.matmul(l2, self.w3_eval) + self.b3_eval
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            with tf.variable_scope('l1'):
                self.w1_tar = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer,
                                              collections=c_names)
                self.b1_tar = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, self.w1_tar) + self.b1_tar)
            with tf.variable_scope('l2'):
                self.w2_tar = tf.get_variable('w2', [n_l1, n_l2], initializer=w_initializer, collections=c_names)
                self.b2_tar = tf.get_variable('b2', [1, n_l2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, self.w2_tar) + self.b2_tar)
            with tf.variable_scope('l3'):
                self.w3_tar = tf.get_variable('w3', [n_l2, self.n_actions], initializer=w_initializer, collections=c_names)
                self.b3_tar = tf.get_variable('b3', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l2, self.w3_tar) + self.b3_tar

    def store_transition(self, s, a, r, s_, done):
        """
        Store transition in replay memory.
        
        Args:
            s: Current state
            a: Action taken
            r: Reward received
            s_: Next state
            done: Whether episode ended
        """
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        """
        Choose action based on current observation using epsilon-greedy policy.
        
        Args:
            observation: Current state observation
            
        Returns:
            int: Selected action index
        """
        observation = observation[np.newaxis, :]
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action = np.argmax(actions_value)
        return action

    def learn(self):
        """
        Perform one step of training on a batch from replay memory.
        Updates both evaluation and target networks.
        """
        self.sess.run(self.soft_replace_dqn)
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # next state
                self.s: batch_memory[:, :self.n_features],    # current state
            })

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)  # action index
        reward = batch_memory[:, self.n_features + 1]  # reward

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                              self.q_target: q_target})
        self.learn_step_counter += 1

    def save_parameters(self):
        """Save the trained model parameters to file."""
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, self.model_path)
        print('Save parameters.')

"""
Hybrid Deep Reinforcement Learning Algorithm Implementation

This module implements a hybrid reinforcement learning algorithm that combines:
1. Deep Deterministic Policy Gradient (DDPG) - For continuous action space
2. Deep Q-Network (DQN) - For discrete action space

The algorithm is designed to handle both continuous and discrete action spaces in a single framework.
It uses experience replay and target networks to stabilize training.

Key Components:
- DDPG: Actor-Critic architecture for continuous control
- DQN: Q-learning with neural networks for discrete actions
- Experience Replay: Store and sample transitions for stable learning
- Target Networks: Slowly updated networks to provide stable training targets
"""

import numpy as np
import tensorflow as tf
if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

tf.set_random_seed(1)


class HIST_Alg(object):
    """
    Hybrid Intelligent System Training Algorithm (HIST)
    
    This class implements a hybrid reinforcement learning agent that combines DDPG and DQN
    to handle both continuous and discrete action spaces. The algorithm can switch between
    DDPG and DQN based on the current state and action requirements.
    
    Args:
        a_dim (int): Dimension of continuous action space
        n_actions (int): Number of discrete actions for DQN
        s_dim (int): Total state space dimension
        s_dim_ddpg (int): State dimension for DDPG
        s_dim_dqn (int): State dimension for DQN
        a_bound (float): Action bound for DDPG's continuous actions
        envSize (int): Environment size parameter
        model_path_dqn (str): Path to load DQN model
        save_path_ca (str): Path to save the combined agent
        mode (str): 'train' for training mode, else evaluation mode
        LR_DQN (float, optional): Learning rate for DQN. Defaults to 0.01
        LR_A (float, optional): Learning rate for DDPG actor. Defaults to 0.001
        LR_C (float, optional): Learning rate for DDPG critic. Defaults to 0.002
        GAMMA (float, optional): Discount factor. Defaults to 1
        TAU (float, optional): Soft update parameter. Defaults to 0.01
        BATCH_SIZE (int, optional): Batch size for training. Defaults to 32
        MEMORY_SIZE_ddpg (int, optional): Size of DDPG replay buffer. Defaults to 3000
    
    Attributes:
        memory_ddpg (np.ndarray): Experience replay buffer for DDPG
        sess (tf.Session): TensorFlow session for DDPG
        sess1 (tf.Session): TensorFlow session for DQN
    """
    def __init__(
            self, a_dim, n_actions, s_dim, s_dim_ddpg, s_dim_dqn, a_bound, envSize, model_path_dqn, save_path_ca, mode,
            LR_DQN=0.01,
            LR_A=0.001,  # learning rate for ddpg's actor 
            LR_C=0.002,  # learning rate for ddpg's critic 
            GAMMA=1,   # reward discount factor
            TAU=0.01,  # soft replacement
            BATCH_SIZE=32,
            MEMORY_SIZE_ddpg=3000
    ):
        self.envSize = envSize
        self.a_dim, self.n_actions, self.s_dim, self.s_dim_ddpg, self.s_dim_dqn, self.a_bound = \
            a_dim, n_actions, s_dim, s_dim_ddpg, s_dim_dqn, a_bound
        self.LR_DQN, self.GAMMA, self.MEMORY_SIZE_ddpg, self.BATCH_SIZE = \
            LR_DQN, GAMMA, MEMORY_SIZE_ddpg, BATCH_SIZE
        self.model_path_dqn = model_path_dqn+"Net_parameter.ckpt"
        self.save_path_ca = save_path_ca+"Net_parameter.ckpt"
        self.pointer_ddpg = 0
        self.pointer_dqn = 0
        self.memory_ddpg = np.zeros(
            (self.MEMORY_SIZE_ddpg, s_dim_ddpg + s_dim + a_dim + 1 + 1 + 1 + 1),
            dtype=np.float32)  # s, a, s_, r, action_, agentdone, nextDDPG, fireDone, n_fireDone
        self.build_dqn_net()
        self.S = tf.placeholder(tf.float32, [None, self.s_dim_ddpg], 'sddpg')
        self.S_ = tf.placeholder(tf.float32, [None, self.s_dim_ddpg], 'sddpg_')
        self.q_target_ddpg = tf.placeholder(tf.float32, [None, self.a_dim], name='Q_target_ddpg')
        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            self.a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            self.q_ddpg = self._build_c(self.S, self.a, scope='eval', trainable=True)
            self.q_ddpg_ = self._build_c(self.S_, self.a_, scope='target', trainable=False)
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')
        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]
        self.hard_replace = [[tf.assign(ta, ea), tf.assign(tc, ec)] for ta, ea, tc, ec in
                             zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]
        td_error = tf.losses.mean_squared_error(labels=self.q_target_ddpg, predictions=self.q_ddpg)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)
        a_loss = - tf.reduce_mean(self.q_ddpg)
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)
        self.sess = tf.Session()
        if mode == 'train':
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.hard_replace)
        else:
            saver = tf.train.Saver()
            saver.restore(self.sess, self.save_path_ca)

    def build_dqn_net(self):
        """
        Builds the DQN neural network architecture.
        
        Creates two networks:
        1. Evaluation Network: Used for action selection
        2. Target Network: Used for stable Q-value targets
        
        Network Architecture:
        - Input Layer: State dimension (s_dim_dqn)
        - Hidden Layer 1: 300 units with ReLU activation
        - Hidden Layer 2: 200 units with ReLU activation
        - Output Layer: n_actions units (Q-values for each action)
        """
        g1 = tf.Graph()
        with g1.as_default():
            self.s = tf.placeholder(tf.float32, [None, self.s_dim_dqn], name='s')
            self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')
            with tf.variable_scope('eval_net'):
                c_names, n_l1, n_l2, w_initializer, b_initializer = \
                    ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 300, 200, \
                    tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
                with tf.variable_scope('l1'):
                    self.w1_eval = tf.get_variable('w1', [self.s_dim_dqn, n_l1], initializer=w_initializer, collections=c_names)
                    self.b1_eval = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                    l1 = tf.nn.relu(tf.matmul(self.s, self.w1_eval) + self.b1_eval)
                with tf.variable_scope('l2'):
                    self.w2_eval = tf.get_variable('w2', [n_l1, n_l2], initializer=w_initializer, collections=c_names)
                    self.b2_eval = tf.get_variable('b2', [1, n_l2], initializer=b_initializer, collections=c_names)
                    l2 = tf.nn.relu(tf.matmul(l1, self.w2_eval) + self.b2_eval)
                with tf.variable_scope('l3'):
                    self.w3_eval = tf.get_variable('w3', [n_l2, self.n_actions], initializer=w_initializer, collections=c_names)
                    self.b3_eval = tf.get_variable('b3', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    self.q_eval = tf.matmul(l2, self.w3_eval) + self.b3_eval
            with tf.variable_scope('loss'):
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
            with tf.variable_scope('train'):
                self._train_op = tf.train.RMSPropOptimizer(self.LR_DQN).minimize(self.loss)
            # ------------------ target_net ------------------
            self.s_ = tf.placeholder(tf.float32, [None, self.s_dim_dqn], name='s_')
            with tf.variable_scope('target_net'):
                c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
                with tf.variable_scope('l1'):
                    self.w1_tar = tf.get_variable('w1', [self.s_dim_dqn, n_l1], initializer=w_initializer, collections=c_names)
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
            self.sess1 = tf.Session(graph=g1)
            saver = tf.train.Saver()
            saver.restore(self.sess1, self.model_path_dqn)

    def choose_action_dqn(self, observation):
        """
        Selects an action using the DQN network.
        
        Args:
            observation (np.ndarray): Current state observation
            
        Returns:
            int: Selected action index based on maximum Q-value
        """
        observation = observation[np.newaxis, :]
        actions_value = self.sess1.run(self.q_eval, feed_dict={self.s: observation})
        action = np.argmax(actions_value)
        return action

    def _build_a(self, s, scope, trainable):
        """
        Builds the DDPG actor network.
        
        Args:
            s (tf.Tensor): Input state tensor
            scope (str): TensorFlow variable scope
            trainable (bool): Whether the network is trainable
            
        Returns:
            tf.Tensor: Continuous action output scaled to action bounds
        """
        with tf.variable_scope(scope):
            net_1 = tf.layers.dense(s, 100, activation=tf.nn.relu, name='l1_actor', trainable=trainable)
            net_2 = tf.layers.dense(net_1, 100, activation=tf.nn.relu, name='l2_actor', trainable=trainable)
            a = tf.layers.dense(net_2, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        """
        Builds the DDPG critic network.
        
        Args:
            s (tf.Tensor): Input state tensor
            a (tf.Tensor): Input action tensor
            scope (str): TensorFlow variable scope
            trainable (bool): Whether the network is trainable
            
        Returns:
            tf.Tensor: Q-value prediction
        """
        with tf.variable_scope(scope):
            n_l1 = 100
            n_l2 = 100
            w1_s = tf.get_variable('w1_critic_s', [self.s_dim_ddpg, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_critic_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1_critic', [1, n_l1], trainable=trainable)
            net_1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            w2 = tf.get_variable('w2_critic', [n_l1, n_l2], trainable=trainable)
            b2 = tf.get_variable('b2_critic', [1, n_l2], trainable=trainable)
            net_2 = tf.nn.relu(tf.matmul(net_1, w2) + b2)
            return tf.layers.dense(net_2, 1, trainable=trainable)

    def choose_action_ddpg(self, s):
        """
        Selects a continuous action using the DDPG actor network.
        
        Args:
            s (np.ndarray): Current state
            
        Returns:
            np.ndarray: Continuous action values
        """
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn_ddpg(self):
        """
        Performs one step of learning for DDPG.
        
        Process:
        1. Sample batch from replay buffer
        2. Compute target Q-values using target networks
        3. Update critic by minimizing TD error
        4. Update actor using policy gradient
        5. Soft update target networks
        """
        self.sess.run(self.soft_replace)
        sample_index = np.random.choice(self.MEMORY_SIZE_ddpg, size=self.BATCH_SIZE)
        bt = self.memory_ddpg[sample_index, :]
        bs = bt[:, :self.s_dim_ddpg]
        bsDDPG = bs
        ba = bt[:, self.s_dim_ddpg: self.s_dim_ddpg + self.a_dim]
        br = bt[:, self.s_dim_ddpg + self.a_dim: self.s_dim_ddpg + self.a_dim + 1]
        bs_ = bt[:, self.s_dim_ddpg + self.a_dim + 1: self.s_dim_ddpg + self.a_dim + 1 + self.s_dim]
        bsDQN_ = bs_[:, 0:self.s_dim_dqn]
        bsDDPG_ = bs_
        bn = bt[:, -1:]  # using DDPG or not at next step
        bd = bt[:, -2:-1]  # done or not
        ba_ = bt[:, -3:-2]  # target selection at next step
        q_target = br
        eval_actNext_index = ba_.astype(int)
        for i in range(self.BATCH_SIZE):
            if bd[i] == 0:
                if bn[i] == 1:
                    q_ddpg_ = self.sess.run(self.q_ddpg_, feed_dict={self.S_: np.hstack((bsDDPG_[i: i + 1, 2*eval_actNext_index[i][0]: 2*eval_actNext_index[i][0]+2], bsDDPG_[i: i + 1, self.s_dim_dqn:]))})
                    q_target[i] = br[i] + self.GAMMA * q_ddpg_
                else:
                    q_dqn = self.sess1.run(self.q_next, feed_dict={self.s_: bsDQN_[i: i + 1, :]})
                    q_target[i] = br[i] + self.GAMMA * np.max(q_dqn)
            else:
                q_dqn = self.sess1.run(self.q_next, feed_dict={self.s_: bsDQN_[i: i + 1, :]})
                q_target[i] = br[i] + self.GAMMA * np.max(q_dqn)
        self.sess.run(self.atrain, {self.S: bsDDPG})
        self.sess.run(self.ctrain, {self.S: bsDDPG, self.a: ba, self.q_target_ddpg: q_target})

    def store_transition_ddpg(self, s, a, r, s_, action_, agentdone, nextDDPG):
        """
        Stores a transition in the DDPG replay buffer.
        
        Args:
            s (np.ndarray): Current state
            a (np.ndarray): Action taken
            r (float): Reward received
            s_ (np.ndarray): Next state
            action_ (int): Next action index
            agentdone (bool): Whether episode is done
            nextDDPG (bool): Whether to use DDPG for next step
        """
        transition = np.hstack((s, a, [r], s_, action_, agentdone, nextDDPG))
        index_ddpg = self.pointer_ddpg % self.MEMORY_SIZE_ddpg
        self.memory_ddpg[index_ddpg, :] = transition
        self.pointer_ddpg += 1

    def save_Parameters(self):
        """
        Saves the model parameters to disk.
        
        Saves both DDPG and DQN network parameters to the specified paths.
        """
        saver = tf.train.Saver()
        save_path_ca = saver.save(self.sess, self.save_path_ca)
        print('Save parameters.')





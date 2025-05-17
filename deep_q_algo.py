import random
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import gym
from collections import deque
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class Deep_Q_Algo:
    def __init__(self,state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Khởi tạo replay buffer
        self.replay_buffer = deque(maxlen=50000)

        # Khởi tạo tham số của Agent
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.98
        self.learning_rate = 0.001
        self.update_targetnn_rate = 10

        # Khởi tạo các list để lưu thông số
        self.episode_rewards = []
        self.episode_losses = []
        self.episode_exploration_rates = []
        self.episode_steps = []
        self.episode_success_rates = []
        self.episode_found_targets = []

        # Khởi tạo session và graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.compat.v1.Session()
            tf.compat.v1.keras.backend.set_session(self.sess)
            
            # Khởi tạo mạng
            self.main_network = self.get_nn()
            self.target_network = self.get_nn()
            
            # Update weight của mạng target = mạng main
            self.target_network.set_weights(self.main_network.get_weights())
            
            # Khởi tạo tất cả các biến
            self.sess.run(tf.compat.v1.global_variables_initializer())
            
            # Tạo saver để lưu mô hình
            self.saver = tf.compat.v1.train.Saver()
            
        print(self.state_size, type(self.state_size))

    def get_nn(self):
        model = Sequential()
        model.add(Dense(32, activation='relu', input_dim=self.state_size, name='dense_1'))
        model.add(Dense(32, activation='relu', name='dense_2'))
        model.add(Dense(self.action_size, name='dense_3'))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def save_model(self, path):
        with self.graph.as_default():
            self.saver.save(self.sess, path)

    def load_model(self, path):
        with self.graph.as_default():
            self.saver.restore(self.sess, path)

    def save_experience(self, state, action, reward, next_state, terminal):
        self.replay_buffer.append((state, action, reward, next_state, terminal))

    def get_batch_from_buffer(self, batch_size):

        exp_batch = random.sample(self.replay_buffer, batch_size)
        state_batch = np.array([batch[0] for batch in exp_batch]).reshape(batch_size, self.state_size)
        action_batch = np.array([batch[1] for batch in exp_batch])
        reward_batch = [batch[2] for batch in exp_batch]
        next_state_batch = np.array([batch[3] for batch in exp_batch]).reshape(batch_size, self.state_size)
        terminal_batch = [batch[4] for batch in exp_batch]
        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch

    def plot_training_metrics(self, save_path=None):
        """
        Vẽ đồ thị các thông số quan trọng trong quá trình huấn luyện
        """
        plt.figure(figsize=(20, 20))  # Tăng kích thước figure
        
        # Plot 1: Episode Rewards
        plt.subplot(3, 2, 1)
        plt.plot(self.episode_rewards)
        plt.title('Episode Rewards', fontsize=14, pad=20)  # Tăng font size và padding
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Total Reward', fontsize=12)
        plt.grid(True)
        
        # Plot 2: Loss
        plt.subplot(3, 2, 2)
        plt.plot(self.episode_losses)
        plt.title('Training Loss', fontsize=14, pad=20)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True)
        
        # Plot 3: Exploration Rate
        plt.subplot(3, 2, 3)
        plt.plot(self.episode_exploration_rates)
        plt.title('Exploration Rate (Epsilon)', fontsize=14, pad=20)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Epsilon', fontsize=12)
        plt.grid(True)
        
        # Plot 4: Steps per Episode
        plt.subplot(3, 2, 4)
        plt.plot(self.episode_steps)
        plt.title('Steps per Episode', fontsize=14, pad=20)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Steps', fontsize=12)
        plt.grid(True)

        # Plot 5: Success Rate
        plt.subplot(3, 2, 5)
        # Tính success rate trung bình cho mỗi 10 episode
        window_size = 10
        success_rates = np.array(self.episode_success_rates)
        rolling_success = np.convolve(success_rates, np.ones(window_size)/window_size, mode='valid')
        plt.plot(rolling_success)
        plt.title('Success Rate (Rolling Average)', fontsize=14, pad=20)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Success Rate', fontsize=12)
        plt.grid(True)
        plt.ylim(0, 1)

        # Plot 6: Found Targets
        plt.subplot(3, 2, 6)
        # Tính số target tìm thấy trung bình cho mỗi 10 episode
        found_targets = np.array(self.episode_found_targets)
        rolling_targets = np.convolve(found_targets, np.ones(window_size)/window_size, mode='valid')
        plt.plot(rolling_targets)
        plt.title('Found Targets (Rolling Average)', fontsize=14, pad=20)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Number of Targets', fontsize=12)
        plt.grid(True)
        
        # Tăng khoảng cách giữa các subplot
        plt.subplots_adjust(
            top=0.95,      # Khoảng cách từ top
            bottom=0.05,   # Khoảng cách từ bottom
            left=0.1,      # Khoảng cách từ left
            right=0.9,     # Khoảng cách từ right
            hspace=0.4,    # Khoảng cách theo chiều dọc giữa các subplot
            wspace=0.3     # Khoảng cách theo chiều ngang giữa các subplot
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Tăng DPI và thêm bbox_inches
        plt.show()

    def update_metrics(self, episode_reward, episode_loss, episode_steps, success_rate, found_targets):
        """
        Cập nhật các thông số sau mỗi episode
        """
        self.episode_rewards.append(episode_reward)
        self.episode_losses.append(episode_loss)
        self.episode_exploration_rates.append(self.epsilon)
        self.episode_steps.append(episode_steps)
        self.episode_success_rates.append(success_rate)
        self.episode_found_targets.append(found_targets)

    def train_main_network(self, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.get_batch_from_buffer(
            batch_size)

        # Lấy Q value của state hiện tại
        q_values = self.main_network.predict(state_batch, verbose=0)

        # Lấy Max Q values của state S' (State chuyển từ S với action A)
        next_q_values = self.target_network.predict(next_state_batch, verbose=0)
        max_next_q = np.amax(next_q_values, axis=1)

        for i in range(batch_size):
            new_q_values = reward_batch[i] if terminal_batch[i] else reward_batch[i] + self.gamma * max_next_q[i]
            q_values[i][action_batch[i]] = new_q_values

        # Lưu loss để theo dõi
        history = self.main_network.fit(state_batch, q_values, verbose=0)
        return history.history['loss'][0]  # Trả về loss của batch hiện tại

    def make_decision(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.action_size)

        state = state.reshape((1, self.state_size))
        q_values = self.main_network.predict(state, verbose=0)
        return np.argmax(q_values[0])



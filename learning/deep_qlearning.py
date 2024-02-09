import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from environment.env import Environnement
from matplotlib import pyplot as plt

class DQN(nn.Module):
    """Class for Deep Q-Learning Network"""

    def __init__(self, input_dims=1, n_actions=4, learning_rate=1e-3, params_list=[32, 32], loss_fct='mse'):
        super(DQN, self).__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.network = nn.Sequential(*self.layers).to(self.device)

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        
    def forward(self, state):
        
    
class ReplayBuffer():
    """Class for Experience Replay Buffer"""

    def __init__(self, memory_size, batch_size, state_input_shape, device='cpu'):
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory_counter = 0
        # Initialize the replay buffer tensors 
        self.current_state_memory = np.zeros((self.memory_size, *state_input_shape))
        self.next_state_memory = np.zeros((self.memory_size, *state_input_shape))
        self.action_memory = np.zeros(self.memory_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.memory_size)

    def store_transition(self, current_state, action, reward, next_state):
        # Update the index, wrapping around when it exceeds the memory size
        idx = self.memory_counter % self.memory_size

        # Store the transitions
        self.current_state_memory[idx] = current_state
        self.next_state_memory[idx] = next_state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward

        # Increment the counter
        self.memory_counter += 1

    def sample_buffer(self):
        max_value = min(self.memory_counter, self.memory_size)
        batch_indices = np.random.choice(max_value, self.batch_size, replace=False)

        current_state_batch = self.current_state_memory[batch_indices]
        next_state_batch = self.next_state_memory[batch_indices]
        action_batch = self.action_memory[batch_indices]
        reward_batch = self.reward_memory[batch_indices]

        return current_state_batch, action_batch, reward_batch, next_state_batch


class DeepQLearningAgent():
    """Class for Deep Q-Learning Algorithm"""

    def __init__(self, 
                    environment: Environnement, 
                    gamma:float = 0.99, 
                    learning_rate: float = 1e-2, 
                    params_list=[32, 32], 
                    replay_buffer_memory_size=32*10, # to get 10 batches
                    batch_size=32, 
                    input_dims=1, # as we are just inputting the state index
                    loss_fct='mse',
                    epsilon_min=0.01,
                    n_epochs=100, 
                    n_time_steps=1000, 
                    freq_update_target=5,
                    epsilon_decay=0.999,
                ):

        self.environment = environment
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.params_list = params_list
        self.replay_buffer_memory_size = replay_buffer_memory_size
        self.batch_size = batch_size

        self.input_dims = input_dims 
        self.n_states = self.environment.state_space.n_states
        self.n_actions = self.environment.action_space.n_actions

        self.evaluation_q_network = DQN(input_dims=self.input_dims, n_actions=self.n_actions, learning_rate=self.learning_rate, params_list=self.params_list)
        self.target_q_network = DQN(input_dims=self.input_dims, n_actions=self.n_actions, learning_rate=self.learning_rate, params_list=self.params_list)

        # the target network is initialized with the same weights as the evaluation network
        self.target_q_network.load_state_dict(self.evaluation_q_network.state_dict())

        self.replay_buffer = ReplayBuffer(memory_size=self.replay_buffer_memory_size, batch_size=self.batch_size, state_input_shape=(self.input_dims,), device='cpu')

        self.policy = np.zeros((self.n_states, self.n_actions))

        self.n_epochs = n_epochs
        self.n_time_steps = n_time_steps
        self.freq_update_target = freq_update_target
        self.epsilon_decay = epsilon_decay

    def choose_action(self, state_index, epsilon):
        """Choose an action following Epsilon Greedy Policy."""
        if np.random.random() < epsilon:
            action = np.random.choice(self.n_actions)
        else:
            action = np.argmax(self.policy[state_index])
        return action
        

    def train(self):

        losses = []

        print(f"Initialize Training for DQN Network on Device : {self.target_q_network.device}")

        epsilon = 1
        epsilons = []

        for epoch in tqdm(range(self.n_epochs)):
            

            # Generate a Random Initial State

            for _ in range(self.n_time_steps):

                # <--- Collect Experience and store them in the Replay Buffer --->
                
                # get the current state index
                current_state_index = self.environment.reset()
                
                # Choose an action following Epsilon Greedy Policy
                action = self.choose_action(current_state_index, epsilon)

                # Update State
                next_state_index, reward = self.environment.step(action)

                # Store the transition in the Replay Buffer
                self.replay_buffer.store_transition(current_state_index, action, reward, next_state_index)

                if self.replay_buffer.memory_counter > self.batch_size:
                    # Sample a batch from the Replay Buffer
                    current_state_batch, action_batch, reward_batch, next_state_batch = self.replay_buffer.sample_buffer()

                    # Start the training process
                    loss_value = self.learn(current_state_batch, action_batch, reward_batch, next_state_batch)
                    losses.append(loss_value)
            
            avg_loss = np.mean(losses[-10:])
            print(f'[INFO] Last loss: {loss_value}')
            print(f'[INFO] Average loss: {avg_loss}')
            print(f'[INFO] epsilon: {epsilon}')

            # update the epsilon value
            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
        
        # print(f'[INFO] Best Score: {max(score_history)}')
        print(f'[INFO] Average loss: {avg_loss}')
        print("[INFO] Deep Q-Learning Training : Process Completed !")
        
        # Extract policy
        self.update_policy()


        # Plot the convergence of the algorithm

        plt.plot(losses, 'b', label='loss')
        # plt.plot(epsilons, 'r', label='Exploration Rate')
        plt.title("Convergence of DQN Algorithm")
        plt.xlabel("Epoch")
        plt.ylabel("Average loss")
        plt.savefig("./figures/convergence_deep_q_learning.png")

        return losses
    
    
    def learn(self, current_state_batch, action_batch, reward_batch, next_state_batch):
        """Train the Deep Q-Learning Network."""
        current_state_batch = torch.tensor(current_state_batch, dtype=torch.float).to(self.evaluation_q_network.device)
        action_batch = torch.tensor(action_batch, dtype=torch.int64).to(self.evaluation_q_network.device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float).to(self.evaluation_q_network.device)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float).to(self.evaluation_q_network.device)

        self.evaluation_q_network.optimizer.zero_grad()

        # Get Q-values for the current state-action pairs
        # Get Q-values for the next state
        q_eval = self.evaluation_q_network.forward(current_state_batch)[np.arange(self.batch_size), action_batch]

        # Get the maximum Q-value for each next state
        q_next = self.target_q_network.forward(next_state_batch).max(dim=1)[0]
        
        # Calculate the target Q-values using the Bellman equation
        q_target = reward_batch + self.gamma * q_next

        # Calculate the loss between the predicted and target Q-values
        loss = nn.MSELoss()(q_target, q_eval)
        
        # Backpropagation
        loss.backward()

        self.evaluation_q_network.optimizer.step()

        return loss.item()

    def update_policy(self):
        """Update the policy based on the evaluation Q-network."""
        for state_index in range(self.n_states):
            state_index_tensor = torch.tensor(state_index, dtype=torch.float).unsqueeze(0).to(self.evaluation_q_network.device)
            self.policy[state_index] = self.evaluation_q_network.forward(state_index_tensor).detach().cpu().numpy()[0]
         
      

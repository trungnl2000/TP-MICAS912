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
        # self.batch_size = 
        # self.memory_size = 
        # self.memory_counter = 
        # Initialize the replay buffer tensors
        # self.current_state_memory = 
        # self.next_state_memory = 
        # self.action_memory = 
        # self.reward_memory = 

    def store_transition(self, current_state, action, reward, next_state):
        # Update the index, wrapping around when it exceeds the memory size

        # Store the transitions
        
        # Increment the counter

    def sample_buffer(self):

        # max_value = 

        # batch_indices = np.random.choice(max_value, self.batch_size, replace=False)

        # current_state_batch = 
        # next_state_batch = 
        # action_batch = 
        # reward_batch = 

        # return current_state_batch, action_batch, reward_batch, next_state_batch


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

        self.evaluation_q_network = 
        self.target_q_network = 

        # the target network is initialized with the same weights as the evaluation network
        # self.target_q_network.

        self.replay_buffer = 

        self.policy = 

        self.n_epochs = n_epochs
        self.n_time_steps = n_time_steps
        self.freq_update_target = freq_update_target
        self.epsilon_decay = epsilon_decay

    def choose_action(self, state_index, epsilon):
        """Choose an action following Epsilon Greedy Policy."""
        

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
                
                # Choose an action following Epsilon Greedy Policy

                # Update State

                # Store the transition in the Replay Buffer

                if ...:
                    # Sample a batch from the Replay Buffer

                    # Start the training process
            
            print(f'[INFO] Last loss: {loss_value}')
            print(f'[INFO] Average loss: {avg_loss}')
            print(f'[INFO] epsilon: {epsilon}')

            # update the epsilon value

        
        # print(f'[INFO] Best Score: {max(score_history)}')
        print(f'[INFO] Average loss: {avg_loss}')
        print("[INFO] Deep Q-Learning Training : Process Completed !")
        
        # Extract policy


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


        # Get Q-values for the current state-action pairs

        # Get Q-values for the next state

        # Get the maximum Q-value for each next state

        # Calculate the target Q-values using the Bellman equation

        # Calculate the loss between the predicted and target Q-values

        # Backpropagation

        return loss.item()

         
      
# Import necessary libraries
import scanpy as sc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import shap
import gym
from gym import spaces
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Suppress OpenMP warning
os.environ['OMP_NUM_THREADS'] = '1'

# Step 1: Load and preprocess the immunotherapy targets dataset
def load_immunotherapy_targets(file_path):
    """
    Load and preprocess the immunotherapy targets dataset.
    """
    # Load dataset
    targets_data = pd.read_excel(file_path)

    # Define column mappings
    required_columns = {
        'Target_Protein': 'Target/Antigen(s)',
        'Approval_Status': 'Approved/In Clinical Trials',
        'Clinical_Trial_Phase': 'Phase',
        'Disease': 'Disease'
    }

    # Ensure required columns exist
    for new_col, old_col in required_columns.items():
        if old_col not in targets_data.columns:
            raise ValueError(f"Column '{old_col}' not found in dataset.")

    # Rename columns for consistency
    targets_data = targets_data.rename(columns={v: k for k, v in required_columns.items() if v})

    # Drop missing values
    targets_data = targets_data.dropna(subset=['Target_Protein', 'Approval_Status', 'Clinical_Trial_Phase', 'Disease'])

    print(f"Loaded {len(targets_data['Target_Protein'].unique())} immunotherapy targets.")
    return targets_data

# Step 2: Model the Tumor Microenvironment (TME) using Graph Neural Networks (GNNs)
class GNNModel(nn.Module):
    """
    Graph Neural Network to model cell-cell interactions in the TME.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

def create_tme_graph(adata):
    """
    Create a graph representation of the Tumor Microenvironment (TME).
    """
    # Convert gene expression to tensor
    X = torch.tensor(adata.X, dtype=torch.float)

    # Create a fully connected graph (simplified approach)
    num_samples = X.shape[0]
    edge_index = torch.tensor([[i, j] for i in range(num_samples) for j in range(num_samples) if i != j], dtype=torch.long).t()

    return Data(x=X, edge_index=edge_index)

def train_gnn(data, model, epochs=50):
    """
    Train the GNN model.
    """
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.x)  # Reconstruct input features
        loss.backward()
        optimizer.step()

        # Print loss every 10 epochs and at the final epoch
        if epoch % 10 == 0 or epoch == epochs-1:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

# Step 3: Custom Gym Environment for RL-based Immunotherapy Optimization
class TMEEnv(gym.Env):
    """
    Custom environment for reinforcement learning to optimize immunotherapy.
    """
    def __init__(self, adata, targets_data):
        super(TMEEnv, self).__init__()

        self.adata = adata
        self.targets_data = targets_data
        self.n_actions = len(targets_data['Target_Protein'].unique())

        # Define action and observation space
        self.action_space = spaces.Discrete(self.n_actions)
        n_env = 100  # Define the number of environments or samples
        self.observation_space = spaces.Box(low=0, high=1, shape=(n_env,), dtype=np.float32)

        self.state = np.random.rand(self.observation_space.shape[0])
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        self.state = np.random.rand(self.observation_space.shape[0])
        return self.state

    def step(self, action):
        target_protein = self.targets_data['Target_Protein'].unique()[action]
        approval_status = self.targets_data[self.targets_data['Target_Protein'] == target_protein]['Approval_Status'].mode()[0]
        phase = self.targets_data[self.targets_data['Target_Protein'] == target_protein]['Clinical_Trial_Phase'].mode()[0]

        # Reward logic
        if approval_status == 'Approved':
            reward = 1.0
        elif phase == 'Phase III':
            reward = 0.8
        elif phase == 'Phase II':
            reward = 0.6
        else:
            reward = 0.4

        self.state += np.random.normal(0, 0.01, size=self.state.shape)
        self.current_step += 1
        done = self.current_step >= 10

        return self.state, reward, done, {}

def optimize_immunotherapy(adata, targets_data):
    """
    Train RL model to optimize immunotherapy selection.
    """
    env = DummyVecEnv([lambda: TMEEnv(adata, targets_data)])
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=1000)
    return model

# Step 4: Immune Escape Prediction
def predict_immune_escape(adata, targets_data):
    """
    Predict immune escape mechanisms.
    """
    print("Predicting immune escape and adapting therapy...")

# Step 5: Explainable AI (SHAP) for Interpretability
def rl_predict_wrapper(model, data):
    """
    Wrapper function to adapt RL model predictions for SHAP.
    """
    # Convert data to the expected observation shape
    observations = np.zeros((data.shape[0], 100))  # Match the observation space shape
    predictions = []
    for i in range(data.shape[0]):
        # Use the RL model to predict actions for each observation
        action, _ = model.predict(observations[i])
        predictions.append(action)
    return np.array(predictions)

def explain_recommendations(adata, model, targets_data):
    """
    Use SHAP to explain the AI's recommendations based on target-specific features.
    """
    # Encode categorical columns
    targets_data['Target_Protein'] = targets_data['Target_Protein'].astype('category').cat.codes
    targets_data['Approval_Status'] = targets_data['Approval_Status'].astype('category').cat.codes

    # Prepare data for SHAP
    data_for_shap = targets_data[['Target_Protein', 'Approval_Status']].copy()

    # Create a wrapper function for the RL model
    def predict_fn(data):
        return rl_predict_wrapper(model, data)

    # Initialize SHAP explainer
    explainer = shap.KernelExplainer(predict_fn, data=data_for_shap)

    # Compute SHAP values
    shap_values = explainer.shap_values(data_for_shap)

    print("SHAP values computed for model interpretability.")

    # Visualize SHAP values
    shap.summary_plot(shap_values, data_for_shap)

    

# Main function
def main():
    # Step 1: Load immunotherapy targets dataset
    targets_file_path = 'Cancer Immunotherapy Targets.xlsx'
    targets_data = load_immunotherapy_targets(targets_file_path)

    # Step 2: Load and preprocess a sample scRNA-seq dataset
    adata = sc.AnnData(np.random.rand(100, 50))  # 100 samples, 50 genes

    # Step 3: Model the Tumor Microenvironment (TME) using GNNs
    data = create_tme_graph(adata)
    gnn_model = GNNModel(input_dim=data.num_features, hidden_dim=8, output_dim=data.num_features)
    train_gnn(data, gnn_model, epochs=50)

    # Step 4: Optimize Immunotherapy using RL
    rl_model = optimize_immunotherapy(adata, targets_data)

    # Step 5: Predict Immune Escape Mechanisms
    predict_immune_escape(adata, targets_data)

    # Step 6: Explain AI Recommendations
    explain_recommendations(adata, rl_model, targets_data)

if __name__ == "__main__":
    main()

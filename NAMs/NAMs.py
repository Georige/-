import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ExU(nn.Module):
    """
    Exp-Centered Unit (ExU) Activation Layer.
    
    As described in Section 2, Eq (2) of the paper:
    h(x) = f(e^w * (x - b))
    
    The weights are initialized in the range [3, 4] to create steep slopes 
    capable of modeling sharp jumps in data.
    """
    def __init__(self, in_features, out_features):
        super(ExU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weights (w) and Bias (b)
        self.weights = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(in_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Initialization is crucial for ExU units to work.
        # Paper suggests Normal(x, 0.5) where x in [3, 4].
        nn.init.normal_(self.weights, mean=4.0, std=0.5)
        nn.init.uniform_(self.bias, -1, 1) # Biases usually centered around input domain

    def forward(self, x):
        # x shape: [batch_size, in_features]
        # Eq (2): exp(w) * (x - b)
        # We need to broadcast to handle the matrix multiplication logic for the custom operation
        
        # Calculate e^w
        ex_weights = torch.exp(self.weights) 
        
        # For a standard MLP logic where in_features=1 (per feature net), 
        # we can use simple broadcasting.
        # (x - b) -> [batch, in] - [in] -> [batch, in]
        # Then multiply by weights.
        
        # Note: In standard NAMs, the input to the first layer is usually 1 single feature.
        # This implementation assumes in_features=1 for the first layer of a FeatureNN.
        output = (x - self.bias)
        output = output.matmul(ex_weights)
        
        # Activation: The paper uses ReLU-n (capped ReLU).
        # Here we use ReLU capped at 1 (ReLU-1) as a default.
        output = torch.clamp(output, min=0, max=1)
        
        return output

class FeatureNN(nn.Module):
    """
    Neural Network for a single input feature.
    
    NAMs learn a linear combination of these networks:
    g(E[y]) = beta + f_1(x_1) + ... + f_k(x_k).
    """
    def __init__(self, num_units=1024, dropout=0.1, feature_dropout=0.0):
        super(FeatureNN, self).__init__()
        
        # First layer is ExU to capture sharp transitions
        # Input dimension is always 1 because each net attends to 1 feature.
        self.layer1 = ExU(1, num_units)
        
        # Second layer helps smooth or adjust the function
        self.layer2 = nn.Linear(num_units, num_units // 2)
        
        # Output layer projects back to a scalar contribution
        self.layer3 = nn.Linear(num_units // 2, 1)
        
        self.dropout = nn.Dropout(p=dropout)
        self.feature_dropout = feature_dropout # Probability of dropping the whole feature
        self.activation = nn.ReLU() # Standard ReLU for subsequent layers

    def forward(self, x):
        # x shape: [batch_size, 1]
        
        # Feature Dropout (Simulated during training)
        if self.training and self.feature_dropout > 0:
            if torch.rand(1) < self.feature_dropout:
                return torch.zeros_like(x)

        x = self.layer1(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        out = self.layer3(x)
        return out

class NAM(nn.Module):
    """
    The Neural Additive Model architecture.
    """
    def __init__(self, num_inputs, num_units=1024, dropout=0.1, feature_dropout=0.0):
        super(NAM, self).__init__()
        self.num_inputs = num_inputs
        
        # Create a list of Feature Networks, one per input feature
        self.feature_nns = nn.ModuleList([
            FeatureNN(num_units, dropout, feature_dropout) for _ in range(num_inputs)
        ])
        
        # Global bias term (beta in Eq 1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x shape: [batch_size, num_inputs]
        
        feature_outputs = []
        
        # Iterate over each feature index and its corresponding network
        for i in range(self.num_inputs):
            # Extract the i-th feature column: [batch, 1]
            feature_slice = x[:, i].unsqueeze(1)
            
            # Pass through the sub-network
            out = self.feature_nns[i](feature_slice)
            feature_outputs.append(out)
        
        # Stack outputs: [batch, num_inputs]
        feature_contributions = torch.cat(feature_outputs, dim=1)
        
        # Sum contributions across features + bias
        prediction = torch.sum(feature_contributions, dim=1) + self.bias
        
        return prediction, feature_contributions

# ==========================================
# Example Usage & Training Loop Simulation
# ==========================================

def train_nam_example():
    # 1. Setup Data (Synthetic)
    # Batch size 32, 5 Features
    x_train = torch.rand(32, 5) * 2 - 1 # Random values in [-1, 1]
    y_train = torch.rand(32) # Random targets
    
    # 2. Initialize Model
    # Paper uses hidden units ~1024
    model = NAM(num_inputs=5, num_units=64, dropout=0.1) 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 3. Training Loop
    print("Starting Training Step...")
    model.train()
    
    # Forward pass
    # Returns prediction and individual feature contributions
    prediction, contributions = model(x_train)
    
    # Loss Calculation
    # Task Loss (MSE for regression)
    mse_loss = F.mse_loss(prediction, y_train)
    
    # Regularization: Output Penalty
    # The paper penalizes the L2 norm of the feature net predictions
    # reg_loss = lambda * (1/K) * sum(f_k(x)^2)
    output_penalty_lambda = 0.01
    output_penalty = output_penalty_lambda * torch.mean(torch.sum(contributions ** 2, dim=1))
    
    total_loss = mse_loss + output_penalty
    
    # Backward
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print(f"Loss: {total_loss.item():.4f}")
    print("NAM Training Step Complete.")
    
    # 4. Interpretability
    # To interpret, we simply look at 'contributions' vs 'x_train' for a specific feature.
    print("\nInterpretability Check:")
    print(f"Feature 0 Input: {x_train[0,0]:.2f}")
    print(f"Feature 0 Contribution: {contributions[0,0]:.2f}")

if __name__ == "__main__":
    train_nam_example()
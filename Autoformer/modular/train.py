# train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models import Autoformer
from utils import get_default_config, update_config

def train_step(model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, X: torch.Tensor, y: torch.Tensor, I: int, O: int) -> float:
    """
    Performs a single training step.

    Args:
        model (nn.Module): The Autoformer model
        optimizer (torch.optim.Optimizer): The optimizer
        loss_fn (nn.Module): The loss function
        X (torch.Tensor): Input tensor
        y (torch.Tensor): Target tensor
        I (int): Input length
        O (int): Prediction length

    Returns:
        float: The loss value for this step
    """
    optimizer.zero_grad()
    output = model(X, I, O)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()
    return loss.item()

def train(model: nn.Module, train_loader: DataLoader, num_epochs: int, learning_rate: float, I: int, O: int) -> None:
    """
    Trains the Autoformer model.

    Args:
        model (nn.Module): The Autoformer model
        train_loader (DataLoader): DataLoader for training data
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate for the optimizer
        I (int): Input length
        O (int): Prediction length
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for X, y in train_loader:
            loss = train_step(model, optimizer, loss_fn, X, y, I, O)
            epoch_loss += loss
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    # Example usage
    config = get_default_config()
    config = update_config(config, input_dim=5, model_dim=256)

    model = Autoformer(config)
    
    # Generate some dummy data
    I, O = 100, 10  # Input length, Output length
    batch_size = 32
    num_samples = 1000
    
    X = torch.randn(num_samples, I, config['input_dim'])
    y = torch.randn(num_samples, O, config['input_dim'])
    
    # Create DataLoader
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Train the model
    num_epochs = 10
    learning_rate = 0.001
    
    train(model, train_loader, num_epochs, learning_rate, I, O)

    # Save the trained model
    torch.save(model.state_dict(), 'autoformer_model.pth')

    print("Training completed. Model saved as 'autoformer_model.pth'")

    # Example of how to load and use the trained model
    loaded_model = Autoformer(config)
    loaded_model.load_state_dict(torch.load('autoformer_model.pth'))
    loaded_model.eval()

    # Generate a test sample
    test_input = torch.randn(1, I, config['input_dim'])
    
    with torch.no_grad():
        prediction = loaded_model(test_input, I, O)
    
    print(f"Test input shape: {test_input.shape}")
    print(f"Prediction shape: {prediction.shape}")
    print(f"Prediction: {prediction}")
"""
File to test if pytorch is using the GPU
"""

import torch
import time
import pandas as pd


def test(device_str: str = "cuda", epochs: int = 100) -> float:
    # Check if device is available
    try:
        device = torch.device(device_str)
        print(f"Using device: {device}")
    except:
        raise ValueError("Device not available")

    # Create a simple neural network
    class SimpleNN(torch.nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = torch.nn.Linear(128, 256)
            self.fc2 = torch.nn.Linear(256, 512)
            self.fc3 = torch.nn.Linear(512, 256)
            self.fc4 = torch.nn.Linear(256, 128)
            self.fc5 = torch.nn.Linear(128, 64)
            self.fc6 = torch.nn.Linear(64, 32)
            self.fc7 = torch.nn.Linear(32, 16)
            self.fc8 = torch.nn.Linear(16, 8)
            self.fc9 = torch.nn.Linear(8, 4)
            self.fc10 = torch.nn.Linear(4, 2)
            self.fc11 = torch.nn.Linear(2, 1)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.relu(self.fc3(x))
            x = self.relu(self.fc4(x))
            x = self.relu(self.fc5(x))
            x = self.relu(self.fc6(x))
            x = self.relu(self.fc7(x))
            x = self.relu(self.fc8(x))
            x = self.relu(self.fc9(x))
            x = self.relu(self.fc10(x))
            x = self.relu(self.fc11(x))
            return x

    start = time.time()

    # Initialize model, loss function, and optimizer
    model = SimpleNN().to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create dummy data
    x = torch.randn(512, 128).to(device)
    y = torch.randn(512, 1).to(device)

    # Train for a few epochs
    print(f"Starting training... on {device_str} with {epochs} epochs")
    for _ in range(epochs):
        # Forward pass
        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(f"Epoch {epoch+1}/5, Loss: {loss.item():.4f}")

    # print("Training complete!")
    end = time.time()

    return end - start


def plot(data: pd.DataFrame):
    import matplotlib.pyplot as plt

    # Plot the data in one plot (device have different colors)
    fig, ax = plt.subplots()
    for device in data["device"].unique():
        df = data[data["device"] == device]
        ax.plot(df["epochs"], df["time_taken"], label=device)

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Time taken (s)")
    ax.set_title("Time taken for training")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    DEVICES = ["cuda", "cpu"]
    EPOCHS = [100, 1000, 10000]

    data = []
    for device in DEVICES:
        for epoch in EPOCHS:
            time_taken = test(device, epoch)
            data.append({"device": device, "epochs": epoch, "time_taken": time_taken})

    df = pd.DataFrame(data)
    plot(df)


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from circleDataset import FixCircleDataset
from torch.utils.tensorboard import SummaryWriter

def starting_train(train_datasize, val_datasize, model, batch_size, epochs):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize dataloaders
    train_loader = DataLoader(FixCircleDataset(data_size=train_datasize), batch_size=64, shuffle=True)
    val_loader = DataLoader(FixCircleDataset(data_size=val_datasize), batch_size=64, shuffle=True)

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    
    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for batch in tqdm(train_loader):  # show the times for each batch

            samples, labels = batch[0], batch[1]

            samples.to(device)
            labels.to(device)
            
            outputs = model(samples)
            
            # Backpropagation and gradient descent

            loss = iou(outputs.T, labels.T)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()  # reset gradients before next iteration

            step += 1

        print(loss)


def iou(a,b) -> float:
    """Calculate the intersection over union of two circles"""
    r1 = a[2]
    r2 = b[2]
    d = torch.norm(a[0:2] - b[0:2],dim=0)
    
    r1_sq, r2_sq = r1**2, r2**2

    d1 = (r1_sq - r2_sq + d**2) / (2 * d)
    d2 = d - d1
    h1 = r1_sq * torch.acos(torch.clamp(d1 / r1, -0.999,0.999))
    h2 = d1 * torch.sqrt(torch.clamp(r1_sq - d1**2,0))
    h3 = r2_sq * torch.acos(torch.clamp(d2 / r2, -0.999,0.999))
    h4 = d2 * torch.sqrt(torch.clamp(r2_sq - d2**2,0))
    
    intersection = h1 + h2 + h3 + h4
    union = (r1_sq + r2_sq) * np.pi - intersection
    
    print(a)
    print(d)
    
    return torch.mean(intersection / (union+0.000001))

if __name__ == "__main__":
    from network import Network
    model = Network()
    starting_train(1999, 100, model, 64, 100)
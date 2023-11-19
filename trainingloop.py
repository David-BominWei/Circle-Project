import torch
import torch.nn as nn
import torchvision
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
    lossfunc = nn.MSELoss()
    
    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for batch in tqdm(train_loader):  # show the times for each batch

            samples, labels = batch[0], batch[1]

            samples = samples.to(device)
            labels = labels.to(device)
            model = model.to(device)
            
            outputs = model(samples)
            
            # Backpropagation and gradient descent

            loss = lossfunc(outputs, labels.to(torch.float32))
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()  # reset gradients before next iteration

            step += 1

        print(loss)
            
        
# def lossfunc(box1,box2):
#     box1x1 = (box1[:,0] - box1[:,2]).reshape(-1,1)
#     box1x2 = (box1[:,0] + box1[:,2]).reshape(-1,1)
#     box1y1 = (box1[:,1] - box1[:,2]).reshape(-1,1)
#     box1y2 = (box1[:,1] + box1[:,2]).reshape(-1,1)

#     box2x1 = (box2[:,0] - box2[:,2]).reshape(-1,1)
#     box2x2 = (box2[:,0] + box2[:,2]).reshape(-1,1)
#     box2y1 = (box2[:,1] - box2[:,2]).reshape(-1,1)
#     box2y2 = (box2[:,1] + box2[:,2]).reshape(-1,1)
    
#     print(torch.cat((box1x1,box1y1,box1x2,box1y2),dim=1))

#     return torch.mean(torchvision.ops.complete_box_iou_loss(torch.cat((box1x1,box1y1,box1x2,box1y2),dim=1), torch.cat((box2x1,box2y1,box2x2,box2y2),dim=1)))

if __name__ == "__main__":
    from network import ResNet, ResidualBlock
    model = ResNet(ResidualBlock,[3,4,6,3])
    starting_train(1999, 100, model, 64, 100)

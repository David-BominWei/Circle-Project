import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from circleDataset import FixCircleDataset
from circle.circle_detection_data import iou, CircleParams
from torch.utils.tensorboard import SummaryWriter

def starting_train(train_datasize, val_datasize, model, batch_size, epochs):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize dataloaders
    train_loader = DataLoader(FixCircleDataset(data_size=train_datasize), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(FixCircleDataset(data_size=val_datasize), batch_size=val_datasize, shuffle=True)

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    lossfunc = nn.MSELoss()
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # training loop
        for batch in tqdm(train_loader):

            samples, labels = batch[0], batch[1]

            samples = samples.to(device)
            labels = labels.to(device)
            model = model.to(device)
            
            outputs = model(samples)

            loss = lossfunc(outputs, labels.to(torch.float32))
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        print(loss)

        # test loop
        with torch.no_grad():
            batch = next(iter(val_loader))
            samples, labels = batch[0], batch[1]

            samples = samples.to(device)
            
            output = model(samples)
            
            output = output.to("cpu")
           
            loss = []
            
            for i in range(len(samples)):
                loss.append(iou(CircleParams(output[i][0],output[i][1],output[i][2]), CircleParams(labels[i][0],labels[i][1],labels[i][2])))
                
            
            print(np.mean(np.array(loss)))        
            

if __name__ == "__main__":
    from network import ResNet, ResidualBlock
    model = ResNet(ResidualBlock,[3,4,6,3])
    starting_train(1999, 100, model, 64, 100)

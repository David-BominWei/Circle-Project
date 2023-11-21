import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from circleDataset import FixCircleDataset
from circle.circle_detection_data import iou, CircleParams, draw_circle
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def starting_train(model, train_datasize=10000, val_datasize=100, batch_size=64, epochs=100):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize dataloaders
    train_loader = DataLoader(FixCircleDataset(data_size=train_datasize), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(FixCircleDataset(data_size=val_datasize), batch_size=val_datasize, shuffle=True)

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    lossfunc = nn.MSELoss()
    
    model.apply(initialize_weights)
    
    # Initalize Tensorboard
    writer = SummaryWriter()
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        loss_list = []
        # training loop
        for batch in tqdm(train_loader):

            samples, labels = batch[0], batch[1]

            samples = samples.to(device)
            labels = labels.to(device)
            model = model.to(device)
            
            outputs = model(samples)

            loss = lossfunc(outputs, labels.to(torch.float32))
            loss_list.append(loss.item())
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        scheduler.step()
            
        print("train loss: ", np.mean(np.array(loss_list)))

        # test loop
        with torch.no_grad():
            batch = next(iter(val_loader))
            samples, labels = batch[0], batch[1]
            picksample = np.array(samples[0])

            samples = samples.to(device)
            
            output = model(samples)
            
            output = output.to("cpu")
            
            # plot the figure to show the prediction
            fig = plt.figure(figsize=(10,10))
            plt.imshow(draw_circle(np.zeros([100,100]),min(max(int(output[0][0]),0),100),
                                            min(max(int(output[0][1]),0),100),int(output[0][2])), cmap='Blues')
            plt.imshow(picksample, cmap='gray',alpha=0.5)
            writer.add_figure("example figure", figure=fig, global_step=epoch)
           
            loss = []
            
            for i in range(len(samples)):
                loss.append(iou(CircleParams(output[i][0],output[i][1],output[i][2]), CircleParams(labels[i][0],labels[i][1],labels[i][2])))
                
            
        print("iou: ", np.mean(np.array(loss)))
        
        # write information
        writer.add_scalar("training loss", np.mean(np.array(loss_list)), epoch)
        writer.add_scalar("evaluate iou", np.mean(np.array(loss)), epoch)
        
        # autosave
        torch.save(model.state_dict(), "autosave_model.pkl")
            
    writer.close()        
            

if __name__ == "__main__":
    from network import ResNet
    model = ResNet([3,4,6,3])
    starting_train(model, train_datasize=10000)

# Circle-Project

> The project is aim to build a model which is able to find the circle inside an image using CNN network

## Files

+ `/circle/*`: the folder include all helper functions
+ `circleDataset.py`: the dataset class for initialize the Datasets for training the model. All training data are generated using the `generate_examples` function with default parameter. The following options are included:
  + `FixCircleDataset(img_size, datasize)`: a list of `datasize` number of data is generated and fixed for training the model
+ `network.py`: the model architecture file. Currently the model is using a ResNet architecture. 
  + `ResNet` is the main model class, the architecture is shown in the figure below
  + `ResNetBlock` is the ResNet block module, the model contain a CNN block with calculating the residual as the output.
+ `trainingloop.py`: the model training loop with real-time result showing in tensorboard
  + `starting_train` is the main training function which used to train the model, input parameter `model` to train the model, evaluation after each epoch is perform and the result will show in tensorboard

## Model Design

```flow
graph TD
    subgraph InputLayer[Input (1 channel)]
    end
    
    subgraph Conv1[Conv2d(64, 7x7)]
        -->|BatchNorm2d| BatchNorm2d
        -->|ReLU| ReLU
    end
    
    subgraph MaxPool[MaxPool2d(3x3, 2)]
    end
    
    subgraph ResNetBlock0[ResNetBlock(64)]
        -->|Conv2d| Conv2d
        -->|BatchNorm2d| BatchNorm2d
        -->|ReLU| ReLU
        -->|Conv2d| Conv2d
        -->|BatchNorm2d| BatchNorm2d
        -->|Add Residual Connection| Add
        -->|ReLU| ReLU
    end
    
    subgraph ResNetBlock1[ResNetBlock(64)]
        -->|Conv2d| Conv2d
        -->|BatchNorm2d| BatchNorm2d
        -->|ReLU| ReLU
        -->|Conv2d| Conv2d
        -->|BatchNorm2d| BatchNorm2d
        -->|Add Residual Connection| Add
        -->|ReLU| ReLU
    end
    
    subgraph ResNetBlockN[...more blocks...]
    end
    
    subgraph ResNetBlockLast[ResNetBlock(512)]
        -->|Conv2d| Conv2d
        -->|BatchNorm2d| BatchNorm2d
        -->|ReLU| ReLU
        -->|Conv2d| Conv2d
        -->|BatchNorm2d| BatchNorm2d
        -->|Add Residual Connection| Add
        -->|ReLU| ReLU
    end
    
    subgraph AvgPool[AvgPool2d(7x7)]
    end
    
    subgraph View[View]
    end
    
    subgraph Linear[Linear(num_classes)]
    end
    
    subgraph Output[Output(num_classes)]
    end
    
    InputLayer --> Conv1
    Conv1 --> MaxPool
    MaxPool --> ResNetBlock0
    ResNetBlock0 --> ResNetBlock1
    ResNetBlock1 --> ResNetBlockN
    ResNetBlockN --> ResNetBlockLast
    ResNetBlockLast --> AvgPool
    AvgPool --> View
    View --> Linear
    Linear --> Output
```
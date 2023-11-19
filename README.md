# Circle-Project

> The project is aim to build a model which is able to find the circle inside an image using CNN network

## Files

+ `/circle/*`: the folder include all helper functions
+ `circleDataset.py`: the dataset class for initialize the Datasets for training the model. All training data are generated using the `generate_examples` function with default parameter. The following options are included:
  + `FixCircleDataset(img_size, datasize)`: a list of `datasize` number of data is generated and fixed for training the model
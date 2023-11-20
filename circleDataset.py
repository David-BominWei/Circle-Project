import torch
from circle.circle_detection_data import generate_examples, iou
from torch.utils.data import Dataset
import numpy as np
from torch import tensor
from tqdm import tqdm

class FixCircleDataset(Dataset):
    """
    Generate a list of circles and set as the dataset, the dataset will be fixed and used for training the model
    """
    
    def __init__(self, img_size=100, data_size=10000):
        """
        Args:
            img_size:       The size of image
            data_size:      The number of data in the dataset
        """
        
        super().__init__()
        circle_iter = generate_examples(img_size=img_size)
        circle_list = []
        pos_list = []
        
        #generate the list of circle
        for _ in tqdm(range(data_size)):
            thecircle = next(circle_iter)
            circle_list.append(thecircle[0])
            pos_list.append(thecircle[1])
        
        # set the data as the list circles
        self.circle_data = tensor(np.array(circle_list)).to(torch.float32)
        self.pos_data = tensor(np.array(pos_list))
        self.data_size = data_size
        
    def __getitem__(self, index):
        return self.circle_data[index], self.pos_data[index]
    
    def __len__(self):
        return self.data_size
    
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    test_loader = DataLoader(FixCircleDataset(), batch_size=64, shuffle=True)
    print(next(iter(test_loader))[0].shape)


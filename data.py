import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, root):
        self.dataset = []

        for filename in os.listdir(root):
            if filename[-5] == 'A':
                self.dataset.append(os.path.join(root, filename))
                
    
    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, index):

        img = self.dataset[index]
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        
        mask = self.dataset[index].replace("A.", "C.")
        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        
        nx, ny = img.shape
        if nx & (nx - 1) or ny & (ny - 1):
            ix = 2**int(np.log2(nx))
            iy = 2**int(np.log2(ny))
            rx = np.random.randint(0, nx-ix)
            ry = np.random.randint(0, ny-iy)
            img = img[rx:ix+rx, ry:iy+ry]
            mask = mask[rx:ix+rx, ry:iy+ry]
        
        img = torch.tensor(img/img.max())
        img = torch.unsqueeze(img, dim=0)
        mask = torch.tensor(mask/mask.max())
        
        return img.float(), mask.int()


if __name__ == '__main__':
    data = MyDataset('dataset/train')

    for i in range(0, 3):
        a,c = data[i]
        plt.figure()
        ax1 = plt.subplot(2, 1, 1)
        plt.imshow(np.squeeze(a))
        ax2 = plt.subplot(2, 1, 2)
        plt.imshow(c)
        plt.show()

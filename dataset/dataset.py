from torch.utils.data import Dataset
from torchvision.transforms import transforms as tf
import os
import glob
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, root, transforms=None, istrain=True):
        self.transform = transforms
        self.istrain = istrain
        self.files_A = sorted(glob.glob(os.path.join(root, '%s' % 'trainA') + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s' % 'trainB') + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))  # 因为可能出现两个集合的大小不一致的情况，少的数据集合会不够取

        item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

""" Dataloader for all datasets. """
import os.path as osp
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class DatasetLoader(Dataset):
    """The class to load the dataset"""
    def __init__(self, setname, args, train_aug=False):
        # Set the path according to train, val and test        
        if setname=='train':
            THE_PATH = osp.join(args.dataset_dir, 'train')
            label_list = os.listdir(THE_PATH)
        elif setname=='test':
            THE_PATH = osp.join(args.dataset_dir, 'test')
            label_list = os.listdir(THE_PATH)
        elif setname=='val':
            THE_PATH = osp.join(args.dataset_dir, 'val')
            label_list = os.listdir(THE_PATH)
        else:
            raise ValueError('Wrong setname.') 

        # Generate empty list for data and label           
        data = []
        label = []

        # Get folders' name
        folders = [osp.join(THE_PATH, label) for label in label_list if os.path.isdir(osp.join(THE_PATH, label))]

        # Get the images' paths and labels
        for idx, this_folder in enumerate(folders):
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(idx)

        # Set data, label and class number to be accessable from outside
        self.data = data
        self.label = label
        self.num_class = len(set(label))
        self.image_size = args.image_size

        # Transformation
        if train_aug:
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.RandomCrop(self.image_size, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), 
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), 
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

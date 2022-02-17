import numpy as np

import os

import torch
from torch.utils.data import Dataset


class Simple2DDataset(Dataset):
    def __init__(self, split='train'):
        super().__init__()
        assert split in ['train', 'valid'], f'Split parameters "{split}" must be either "train" or "valid".'
        # Read either train or validation data from disk based on split parameter using np.load.
        # Data is located in the folder "data".

        data_path = os.path.join('data', f'{split}.npz')
        data = np.load(data_path)
        self.samples = data['samples']
        self.annotations = data['annotations']

        '''
        labels_path = os.path.join('data', f'{split}-labels-idx1-ubyte.gz')
        self.data = load_images(data_path)
        self.annotations = load_labels(labels_path)

        simple_2d_classifier_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if split == 'train':
            train = np.load(os.path.join(simple_2d_classifier_directory, 'data', 'train.npz'))
            self.samples = train['samples']
            self.annotations = train['annotations']
        elif split == 'valid':
            valid = np.load(os.path.join(simple_2d_classifier_directory, 'data', 'valid.npz'))
            self.samples = valid['samples']
            self.annotations = valid['annotations']
        '''

        # Hint: you can use os.path.join to obtain a path in a subfolder.
        # Save samples and annotations to class members self.samples and self.annotations respectively.
        # Samples should be an Nx2 numpy array. Annotations should be Nx1.
            
    def __len__(self):
        # Returns the number of samples in the dataset.
        return self.samples.shape[0]
    
    def __getitem__(self, idx):
        # Returns the sample and annotation with index idx.
        sample = self.samples[idx]
        annotation = self.annotations[idx]
        
        # Convert to tensor.
        return {
            'input': torch.from_numpy(sample).float(),
            'annotation': torch.from_numpy(annotation[np.newaxis]).float()
        }


class Simple2DTransformDataset(Dataset):
    def __init__(self, split='train'):
        super().__init__()
        assert split in ['train', 'valid'], f'Split parameters "{split}" must be either "train" or "valid".'
        # Read either train or validation data from disk based on split parameter.
        # Data is located in the folder "data".

        data_path = os.path.join('data', f'{split}.npz')
        data = np.load(data_path)
        self.samples = data['samples']
        self.annotations = data['annotations']

        '''
        simple_2d_classifier_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if split == 'train':
            train = np.load(os.path.join(simple_2d_classifier_directory, 'data', 'train.npz'))
            self.samples = train['samples']
            self.annotations = train['annotations']
        elif split == 'valid':
            valid = np.load(os.path.join(simple_2d_classifier_directory, 'data', 'valid.npz'))
            self.samples = valid['samples']
            self.annotations = valid['annotations']
        '''

        # Hint: you can use os.path.join to obtain a path in a subfolder.
        # Save samples and annotations to class members self.samples and self.annotations respectively.
        # Samples should be an Nx2 numpy array. Annotations should be Nx1.

    def __len__(self):
        # Returns the number of samples in the dataset.
        return self.samples.shape[0]
    
    def __getitem__(self, idx):
        # Returns the sample and annotation with index idx.
        sample = self.samples[idx]
        annotation = self.annotations[idx]
        
        # Transform the sample to a different coordinate system.
        sample = transform(sample)

        # Convert to tensor.
        return {
            'input': torch.from_numpy(sample).float(),
            'annotation': torch.from_numpy(annotation[np.newaxis]).float()
        }


def transform(sample):
    r = np.sqrt(sample[0]**2 + sample[1]**2)
    if r == 0:
        phi = 0
    elif sample[1] >= 0:
        phi = np.arccos(sample[0]/r)
    else:
        phi = 2*np.pi - np.arccos(sample[0]/r)

    new_sample = np.array([r, phi])
    return new_sample

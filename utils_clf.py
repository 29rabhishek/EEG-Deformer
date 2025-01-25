import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import csv
import os
import re


class EEGImageDataset(Dataset):
    def __init__(self, eeg_data, image_paths, labels, transform=None):
        self.eeg_data = eeg_data
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.base_path = './imageNet_images'

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        eeg = torch.tensor(self.eeg_data[idx], dtype=torch.float32)
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return eeg, image
    



def create_dataset(train_batch_size, train_data_path='processed_data/train_data.npz', test_data_path='processed_data/test_data.npz', 
                   train_csv_path='processed_data/train_image_data.csv', test_csv_path='processed_data/test_image_data.csv', 
                   transform = None):
    # Load EEG data
    train_data = np.load(train_data_path, mmap_mode="r",allow_pickle=True)
    test_data = np.load(test_data_path,  mmap_mode="r", allow_pickle=True)
    
    # Load image paths and labels from CSV
    train_image_data = pd.read_csv(train_csv_path)
    test_image_data = pd.read_csv(test_csv_path)

    # Create datasets
    train_dataset = EEGImageDataset(
        eeg_data=train_data['eeg'],
        image_paths=train_image_data['image_paths'].tolist(),
        labels=train_data['labels'],
        transform=transform
    )

    test_dataset = EEGImageDataset(
        eeg_data=test_data['eeg'],
        image_paths=test_image_data['image_paths'].tolist(),
        labels=test_data['labels'],
        transform=transform
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers = 4)
    test_loader = DataLoader(test_dataset, batch_size=train_batch_size, shuffle=False, drop_last=True, num_workers = 4)
    return train_loader, test_loader


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)


def get_task_chunk(subjects, step):
    return np.array_split(subjects, len(subjects) // step)


def log2txt(text_file, content):
    file = open(text_file, 'a')
    file.write(str(content) + '\n')
    file.close()


def log2csv(csv_file, content):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header row (column names)
        writer.writerow(["Metric", "Value"])

        # Write the data from the dictionary
        for key, value in content.items():
            writer.writerow([key, value])

    print(f"Data has been logged to {csv_file}")


def get_checkpoints(path):
    all_files = os.listdir(path)
    ckpt_files = [file for file in all_files if file.endswith(".ckpt")]
    return ckpt_files


def get_epoch_from_ckpt(ckpt_file):
    epoch_match = re.search(r'epoch=(\d+)', ckpt_file)
    epoch_number = int(epoch_match.group(1))
    return epoch_number


class LabelSmoothing(torch.nn.Module):
    """NLL loss with label smoothing.
       refer to: https://github.com/NVIDIA/DeepLearningExamples/blob/8d8b21a933fff3defb692e0527fca15532da5dc6/PyTorch/Classification/ConvNets/image_classification/smoothing.py#L18
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()



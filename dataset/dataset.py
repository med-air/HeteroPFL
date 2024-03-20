import sys, os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset

class Digits5(Dataset):
    """
    Five digits dataset (MNIST, SVHN, USPS, SynthDigits, MNIST_M)
    """

    def __init__(self, site, base_path=None, split="train", transform=None):
        channels = {"MNIST": 1, "SVHN": 3, "USPS": 1, "SynthDigits": 3, "MNIST_M": 3}
        assert split in ["train", "test"]
        assert site in list(channels.keys())
        
        
        base_path = (
            base_path if base_path is not None else "/Dataset/Digits5"
        )
        self.images, self.labels = np.load(
            os.path.join(base_path, f"{site}/{split}_stratified.pkl"), allow_pickle=True
        )

        self.transform = transform
        self.channels = channels[site]
        self.labels = self.labels.astype(np.int64).squeeze()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.channels == 1:
            image = Image.fromarray(image, mode="L")
            image = image.convert("RGB")
        elif self.channels == 3:
            image = Image.fromarray(image, mode="RGB")
        else:
            raise ValueError("{} channel is not allowed.".format(self.channels))

        if self.transform is not None:
            image = self.transform(image)

        return {
            "Image": image,
            "Label": label,
            'Image_idx': idx
        }


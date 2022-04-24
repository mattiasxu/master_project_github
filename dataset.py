from pytorch_lightning import LightningDataModule
import torch
from tqdm import tqdm
import os
from PIL import ImageFile
import PIL
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from collections import namedtuple
import lmdb
import pickle
from torch.nn.functional import one_hot


ImageFile.LOAD_TRUNCATED_IMAGES = True
# PATH = "./tensor_data"

Data = namedtuple('Data', ['top', 'bottom'])


class LatentDataset(Dataset):
    def __init__(self, path, n_embed=512):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
        self.n_embed = n_embed

        if not self.env:
            raise IOError('Cannot open dataset', path)
        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = str(index).encode('utf-8')
            data = pickle.loads(txn.get(key))
        top = one_hot(torch.from_numpy(data.top), self.n_embed).permute(3, 0, 1, 2).type(torch.FloatTensor)
        bot = one_hot(torch.from_numpy(data.bottom), self.n_embed).permute(3, 0, 1, 2).type(torch.FloatTensor)
        return top, bot


class DrivingDataset(Dataset):
    def __init__(self, path, frames=16, skip=8):
        self.path = path
        self.data = os.listdir(path)
        self.frames = frames
        self.skip = skip
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5526, 0.5384, 0.5170),
                (0.2135, 0.2026, 0.1969)
            )
        ])

    def to_tensor(self, idx):
        img = PIL.Image.open(
            self.path + f"/{idx}.png"
        ).convert('RGB')
        return self.transforms(img)

    def __len__(self):
        return 1 + (len(self.data) - self.frames) // self.skip

    def __getitem__(self, idx):
        frames = []
        for i in range(idx * self.skip, idx * self.skip + self.frames):
            frames.append(self.to_tensor(i))
        return torch.stack(frames, 1)

class DataModule(LightningDataModule): # For VQVAE
    def __init__(self, hparams):
        super().__init__()
        self.path = hparams.path
        self.batch_size = hparams.batch_size
    def setup(self, stage=None):
        self.train_set = DrivingDataset(f"{self.path}/train", frames=16, skip=8)
        self.val_set = DrivingDataset(f"{self.path}/val", frames=16, skip=8)
        self.test_set = DrivingDataset(f"{self.path}/test", frames=16, skip=8)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=2)

class LatentDataModule(LightningDataModule): # For SNAILS
    def __init__(self, hparams):
        super().__init__()
        self.path = hparams.path
        self.batch_size = hparams.batch_size

    def setup(self, stage=None):
        self.train_set = LatentDataset(f"{self.path}/train")
        self.val_set = LatentDataset(f"{self.path}/val")

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=2)

if __name__ == "__main__":
    test = LatentDataset("./latent_data/train")
    print(len(test))

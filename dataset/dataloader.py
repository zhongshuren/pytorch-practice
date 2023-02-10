import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def dataloader(args, mode):
    dataset = MNISTDataset(args, mode)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True)


class MNISTDataset(Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode
        self.sample_size = args.train_sample_size if mode == 'train' else args.eval_sample_size

        self.images, self.labels = self.read_data()

    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        sample = {'image': torch.from_numpy(image).unsqueeze(0), 'label': label}
        return sample

    def read_data(self):
        if self.mode == 'train':
            image_file = 'train-images.idx3-ubyte'
            label_file = 'train-labels.idx1-ubyte'
        else:
            image_file = 't10k-images.idx3-ubyte'
            label_file = 't10k-labels.idx1-ubyte'

        with open(f'{self.args.dataset_dir}/{image_file}', 'rb') as f:
            image_set = f.read()
        with open(f'{self.args.dataset_dir}/{label_file}', 'rb') as f:
            label_set = f.read()
        image = []
        label = []
        for i in range(0, self.sample_size):
            tmp_image = np.array([item for item in image_set[16 + i * 784: 16 + 784 + i * 784]],
                                 dtype=np.float32).reshape(28, 28)
            tmp_image /= 255
            tmp_label = int(label_set[8 + i: 8 + 1 + i].hex(), 16)
            image.append(tmp_image)
            label.append(tmp_label)
        return image, label

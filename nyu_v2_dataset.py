import torch.utils.data as data
from path import Path
import numpy as np
from pandas import read_csv
from torchvision.io import read_image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, ToPILImage
from PIL import Image


class NYUv2Dataset(data.Dataset):
    def __init__(self, train=True):

        self.root = Path("/Users/sairaj/Desktop/acads/design_project/")
        self.train = train

        if train:
            csv_data_train = read_csv(self.root + "/data/nyu2_train.csv", header=None)
            self.rgb_paths = csv_data_train[0]
        else:
            csv_data_test = read_csv(self.root + "/data/nyu2_test.csv", header=None)
            self.rgb_paths = csv_data_test[0]

        self.length = len(self.rgb_paths)
        self.transform = Compose(
            [
                Resize((240, 320)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.target_transform = Compose(
            [
                Resize((120, 160)),
                ToTensor(),
            ]
        )

    def __getitem__(self, index):
        path = self.root + self.rgb_paths[index]

        rgb = Image.open(path)
        depth = Image.open(path.replace(".jpg", ".png"))

        return self.transform(rgb), self.target_transform(depth)

    def __len__(self):
        return self.length


if __name__ == "__main__":
    # Testing
    for i in range(0, 10):
        if i > 4:
            continue
        print("for ", i)
    else:
        print("else block")

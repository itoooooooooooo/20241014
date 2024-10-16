import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.files = [f for f in os.listdir(data_path) if f.endswith(".png")]

        # 画像をTensorに変換し、標準化する変換
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),  # (H, W, C) -> (C, H, W) and normalize to [0, 1]
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = os.path.join(self.data_path, self.files[idx])
        
        # 画像をRGBで読み込み（既にRGBなのでconvertは不要）
        image = Image.open(file_name)
        image_tensor = self.transform(image)  # (3, H, W)のTensorに変換

        # ラベルの決定（ファイル名に"normal"が含まれるかどうかで分類）
        label = 0 if "normal" in file_name else 1
        return image_tensor, label

def get_dataloader(data_path, batch_size):
    dataset = ImageDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

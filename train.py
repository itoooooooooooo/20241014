import os
os.environ["OMP_NUM_THREADS"] = "2"
import torch
import torch.optim as optim
import torch.nn as nn
from model import UNet, Diffuser
from data_loader import get_dataloader
from tqdm import tqdm
import yaml
from PIL import Image
import torchvision.transforms as transforms

# YAMLの読み込み
with open("ae.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = get_dataloader(config['train_data_path'], config['batch_size'])
model = UNet(in_ch=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
diffuser = Diffuser(num_timesteps=1000, device=device)
criterion = nn.MSELoss()

# データを5件保存するディレクトリ
save_dir = "sample_images"
os.makedirs(save_dir, exist_ok=True)

# データの可視化と保存
data_iter = iter(train_loader)
for i in range(5):
    data, _ = next(data_iter)  #data=(batchサイズ、3,256,256)
    image = transforms.ToPILImage()(data[0])  # 最初の画像を取り出し、PIL形式に変換
    image.save(f"{save_dir}/sample_image_{i+1}.png")
    print(f"Saved {save_dir}/sample_image_{i+1}.png")

# 学習ループ
for epoch in range(config['epochs']):
    model.train()
    running_loss = 0.0
    for data, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
        data = data.to(device)
        optimizer.zero_grad()

        t = torch.randint(1, 1000+1, (len(data),), device=device)
        x_t, noise = diffuser.add_noise(data, t)
        noise_pred = model(x_t, t)
        loss = criterion(noise, noise_pred)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {running_loss / len(train_loader)}")

# モデルの保存
torch.save(model.state_dict(), config['model_directory'] + "/autoencoder_with_diffusion.pth")

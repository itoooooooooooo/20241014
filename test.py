import os
os.environ["OMP_NUM_THREADS"] = "4"
import torch
import torch.nn as nn
from model import UNet, Diffuser
from data_loader import get_dataloader
from sklearn.metrics import roc_auc_score, roc_curve, auc
import yaml
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

# YAMLの読み込み
with open("ae.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_loader = get_dataloader(config['test_data_path'], config['batch_size'])
model = UNet(in_ch=3).to(device)
model.load_state_dict(torch.load(config['model_directory'] + "/autoencoder_with_diffusion.pth"))
model.eval()
diffuser = Diffuser(num_timesteps=1000, device=device)

criterion = nn.MSELoss(reduction='none')

timestep = 50

results = []

with torch.no_grad():
    for data, labels in test_loader:
        data = data.to(device)

        # テスト時はステップ数を固定
        #fixed_timestep = 700
        #reconstructed, latent, noisy_latent, denoised_latent, noise, noise_pred = model(data, fixed_timestep=fixed_timestep)
        
        # サンプルごとの損失を計算
        t = torch.full((data.size(0),), timestep, device=device, dtype=torch.long)
        x_t, noise = diffuser.add_noise(data, t)
        #reconstructed = torch.squeeze(diffuser.denoise(model, x_t, t))

        noise_pred = model(x_t, t)
        loss = criterion(noise, noise_pred)

        # loss = criterion(noise, noise_pred)

        # 損失とラベルをリストに追加（バッチごと）
        for i in range(data.size(0)):
            results.append([loss[i].mean().item(), labels[i].item()])

# 結果の保存
results = np.array(results)
np.savetxt(config['result_directory'] + "/results.csv", results, delimiter=",", header="loss,label")

# AUC, pAUCの計算
y_true = results[:, 1]
y_scores = results[:, 0]

# AUCの計算
auc_value = roc_auc_score(y_true, y_scores)

# ROC曲線を計算
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# pAUCの計算 (0 <= FPR <= 0.1 の範囲でのAUC)
fpr_limit = 0.1  # pAUCを計算するFPRの範囲
fpr_pauc = fpr[fpr <= fpr_limit]  # FPRが0.1以下の範囲
tpr_pauc = tpr[:len(fpr_pauc)]    # 対応するTPR
pauc_value = auc(fpr_pauc, tpr_pauc) / fpr_limit  # 正規化してpAUCを0-1スケールに

# AUCとpAUCの出力
print(f"AUC: {auc_value}")
print(f"pAUC (FPR <= {fpr_limit}): {pauc_value}")


# #ここからのコードは生成されたサンプルの確認用コード

# 元の画像と再構成画像の保存用ディレクトリを設定
original_image_dir = os.path.join(config['result_directory'], "original_images")
reconstructed_image_dir = os.path.join(config['result_directory'], "reconstructed_images")
os.makedirs(original_image_dir, exist_ok=True)
os.makedirs(reconstructed_image_dir, exist_ok=True)

# データの可視化と保存
data_iter = iter(test_loader)
for i in range(16):
    data, label = next(data_iter)  #data=(batchサイズ、3,256,256)

    # データをデバイスに移動
    data = data.to(device)

    # ノイズを加えて再構成
    t = torch.full((data.size(0),), timestep, device=device, dtype=torch.long)
    x_t, noise = diffuser.add_noise(data, t)
    reconstructed = torch.squeeze(diffuser.denoise(model, x_t, t))
    
    image = transforms.ToPILImage()(data[0])  # 最初の画像を取り出し、PIL形式に変換
    image2 = transforms.ToPILImage()(reconstructed[0])
    image.save(f"{original_image_dir}/{label[0]}label_{i+1}.png")
    image2.save(f"{reconstructed_image_dir}/{label[0]}label_{i+1}.png")  
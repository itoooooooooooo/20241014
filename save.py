import os
os.environ["OMP_NUM_THREADS"] = "4"
import librosa
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def save_log_mel_spectrograms(data_path, save_dir, n_fft=1024, hop_length=512, n_mels=128, power=2.0):
    # 保存先ディレクトリの作成
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # ディレクトリ内のすべての.wavファイルを処理
    files = [f for f in os.listdir(data_path) if f.endswith(".wav")]
    for file_name in files:
        # 音声ファイルの読み込み
        file_path = os.path.join(data_path, file_name)
        y, sr = librosa.load(file_path, sr=None)
        
        # メルスペクトログラムの計算
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, 
                                                         hop_length=hop_length, 
                                                         n_mels=n_mels, 
                                                         power=power)
        # 対数メルスペクトログラムの計算（クリッピングして範囲を制御）
        log_mel_spectrogram = 20.0 * np.log10(np.maximum(mel_spectrogram, 1e-5))

        # 正方形にパディングしてからリサイズ保存
        save_file_name = os.path.join(save_dir, f"{os.path.splitext(file_name)[0]}.png")
        save_spectrogram_image_with_color_and_padding(log_mel_spectrogram, save_file_name)

def save_spectrogram_image_with_color_and_padding(spectrogram, file_path):
    # カラーマップを使用してカラー画像を作成し、一時保存
    plt.figure()
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
    plt.axis('off')
    plt.savefig("temp.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    # パディングして正方形化、再リサイズ
    with Image.open("temp.png") as img:
        # 長辺に合わせて正方形パディング
        max_dim = max(img.size)
        padded_img = Image.new("RGB", (max_dim, max_dim))
        padded_img.paste(img, ((max_dim - img.width) // 2, (max_dim - img.height) // 2))
        
        # (256, 256)にリサイズして保存
        resized_img = padded_img.resize((256, 256))
        resized_img.save(file_path)

        # 画像サイズの出力
        print(f"Saved {file_path} with size {resized_img.size}")

    # 一時ファイルを削除
    os.remove("temp.png")

# 実行
train_data_path = ""
save_dir = ""
save_log_mel_spectrograms(train_data_path, save_dir)

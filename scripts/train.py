import os
import math
import argparse
import random
import numpy as np
from scipy.io import wavfile
from scipy import signal
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.detector import build_model


def pre_emphasis(x, coef=0.97):
    x = np.append(x[0], x[1:] - coef * x[:-1])
    return x


def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filter_bank(sr, n_fft, n_mels, fmin, fmax):
    n_freqs = n_fft // 2 + 1
    m_min = hz_to_mel(fmin)
    m_max = hz_to_mel(fmax)
    m_points = np.linspace(m_min, m_max, n_mels + 2)
    f_points = mel_to_hz(m_points)
    bins = np.floor((n_fft + 1) * f_points / sr).astype(int)
    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        l = bins[i]
        c = bins[i + 1]
        r = bins[i + 2]
        if c > l:
            fb[i, l:c] = (np.arange(l, c) - l) / float(c - l)
        if r > c:
            fb[i, c:r] = (r - np.arange(c, r)) / float(r - c)
    return torch.tensor(fb)


def logmel(x, sr=16000, n_fft=512, win_length=400, hop_length=160, n_mels=64, fmin=20, fmax=8000):
    x = pre_emphasis(x)
    x = torch.tensor(x, dtype=torch.float32)
    w = torch.hann_window(win_length)
    stft = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=w, center=True, pad_mode="reflect", return_complex=True)
    spec = (stft.real ** 2 + stft.imag ** 2)
    fb = mel_filter_bank(sr, n_fft, n_mels, fmin, fmax)
    mel = torch.matmul(fb, spec)
    mel = torch.log1p(mel)
    m = mel.mean()
    s = mel.std()
    mel = (mel - m) / (s + 1e-6)
    return mel


def augment_audio(x, sr):
    if random.random() < 0.5:
        rate = random.uniform(0.95, 1.05)
        x = signal.resample_poly(x, int(rate * 100), 100)
    if random.random() < 0.5:
        noise = np.random.randn(x.shape[0]).astype(np.float32)
        snr = random.uniform(15.0, 30.0)
        sig_p = np.mean(x ** 2) + 1e-12
        noise_p = sig_p / (10 ** (snr / 10.0))
        x = x + noise * math.sqrt(noise_p)
    if random.random() < 0.3:
        gain = random.uniform(0.8, 1.2)
        x = x * gain
    if random.random() < 0.2:
        shift = random.randint(-int(0.05 * sr), int(0.05 * sr))
        if shift > 0:
            x = np.pad(x, (shift, 0))[: x.shape[0]]
        elif shift < 0:
            x = np.pad(x, (0, -shift))[ -shift : ]
    x = np.clip(x, -1.0, 1.0)
    return x


class AudioDataset(Dataset):
    def __init__(self, list_file, sr=16000, n_mels=64, max_frames=1000, augment=False):
        super().__init__()
        self.sr = sr
        self.n_mels = n_mels
        self.max_frames = max_frames
        self.augment = augment
        with open(list_file, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        self.items = []
        for l in lines:
            p, y = l.split(",")
            self.items.append((p, int(y)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, y = self.items[idx]
        sr, x = wavfile.read(path)
        if x.dtype != np.float32:
            x = x.astype(np.float32)
            if x.dtype == np.int16:
                x = x / 32768.0
            elif x.dtype == np.int32:
                x = x / 2147483648.0
        if sr != self.sr:
            g = math.gcd(sr, self.sr)
            up = self.sr // g
            down = sr // g
            x = signal.resample_poly(x, up, down).astype(np.float32)
        if self.augment:
            x = augment_audio(x, self.sr)
        M = logmel(x, sr=self.sr, n_mels=self.n_mels)
        T = M.shape[1]
        if T >= self.max_frames:
            s = random.randint(0, T - self.max_frames)
            M = M[:, s : s + self.max_frames]
        else:
            pad = self.max_frames - T
            M = torch.nn.functional.pad(M, (0, pad))
        X = M.unsqueeze(0)
        return X, y


class ItemsDataset(Dataset):
    def __init__(self, items, sr=16000, n_mels=64, max_frames=1000, augment=False):
        super().__init__()
        self.items = items
        self.base = AudioDataset.__init__
        self.sr = sr
        self.n_mels = n_mels
        self.max_frames = max_frames
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, y = self.items[idx]
        sr, x = wavfile.read(path)
        if x.dtype != np.float32:
            x = x.astype(np.float32)
            if x.dtype == np.int16:
                x = x / 32768.0
            elif x.dtype == np.int32:
                x = x / 2147483648.0
        if sr != self.sr:
            g = math.gcd(sr, self.sr)
            up = self.sr // g
            down = sr // g
            x = signal.resample_poly(x, up, down).astype(np.float32)
        if self.augment:
            x = augment_audio(x, self.sr)
        M = logmel(x, sr=self.sr, n_mels=self.n_mels)
        T = M.shape[1]
        if T >= self.max_frames:
            s = random.randint(0, T - self.max_frames)
            M = M[:, s : s + self.max_frames]
        else:
            pad = self.max_frames - T
            M = torch.nn.functional.pad(M, (0, pad))
        X = M.unsqueeze(0)
        return X, y


class CenterLoss(nn.Module):
    def __init__(self, num_classes=2, feat_dim=128):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, feats, labels):
        return ((feats - self.centers[labels]) ** 2).sum(dim=1).mean()


def forward_emb(model, x):
    x = model.stem(x)
    x = model.block1(x)
    x = model.block2(x)
    x = model.block3(x)
    x = model.block4(x)
    x = model.pool(x)
    return x


def eer(scores, labels):
    scores = np.array(scores)
    labels = np.array(labels)
    order = np.argsort(scores)
    scores = scores[order]
    labels = labels[order]
    P = labels.sum()
    N = len(labels) - P
    fnr = np.cumsum(labels) / (P + 1e-12)
    fpr = (np.arange(len(labels)) - np.cumsum(labels)) / (N + 1e-12)
    d = np.abs(fpr - fnr)
    i = int(np.argmin(d))
    return float(max(fpr[i], fnr[i]))


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(n_mels=args.n_mels, base_ch=args.base_ch, num_classes=2).to(device)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    cl = CenterLoss(num_classes=2, feat_dim=model.head[0].in_features).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.train_list and args.dev_list:
        train_ds = AudioDataset(args.train_list, sr=16000, n_mels=args.n_mels, max_frames=args.max_frames, augment=True)
        dev_ds = AudioDataset(args.dev_list, sr=16000, n_mels=args.n_mels, max_frames=args.max_frames, augment=False)
    else:
        real_train = sorted(glob(os.path.join(args.train_real_dir, "**", "*.wav"), recursive=True))
        fake_train = sorted(glob(os.path.join(args.train_fake_dir, "**", "*.wav"), recursive=True))
        train_items = [(p, 0) for p in real_train] + [(p, 1) for p in fake_train]
        if args.dev_real_dir and args.dev_fake_dir:
            real_dev = sorted(glob(os.path.join(args.dev_real_dir, "**", "*.wav"), recursive=True))
            fake_dev = sorted(glob(os.path.join(args.dev_fake_dir, "**", "*.wav"), recursive=True))
            dev_items = [(p, 0) for p in real_dev] + [(p, 1) for p in fake_dev]
        else:
            random.Random(42).shuffle(train_items)
            n_dev = max(1, int(len(train_items) * args.val_split))
            dev_items = train_items[:n_dev]
            train_items = train_items[n_dev:]
        train_ds = ItemsDataset(train_items, sr=16000, n_mels=args.n_mels, max_frames=args.max_frames, augment=True)
        dev_ds = ItemsDataset(dev_items, sr=16000, n_mels=args.n_mels, max_frames=args.max_frames, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    best_eer = 1.0
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            emb = forward_emb(model, xb)
            loss = ce(logits, yb) + args.center_w * cl(emb, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        all_scores = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in dev_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.softmax(logits, dim=1)[:, 1]
                all_scores.extend(probs.cpu().numpy().tolist())
                all_labels.extend(yb.numpy().tolist())
        e = eer(all_scores, all_labels)
        if e < best_eer:
            best_eer = e
            os.makedirs(args.out_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pt"))
    print("best_eer", round(best_eer, 4))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_list", type=str, default="")
    p.add_argument("--dev_list", type=str, default="")
    p.add_argument("--train_real_dir", type=str, default="")
    p.add_argument("--train_fake_dir", type=str, default="")
    p.add_argument("--dev_real_dir", type=str, default="")
    p.add_argument("--dev_fake_dir", type=str, default="")
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--out_dir", type=str, default="runs")
    p.add_argument("--n_mels", type=int, default=64)
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--center_w", type=float, default=0.1)
    p.add_argument("--max_frames", type=int, default=1000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

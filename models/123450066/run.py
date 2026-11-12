from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))

from src.protokol import data

NIM = "123450066"
JALUR = "rnn"

DEVICE = torch.device("cpu")


def atur_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def persiapkan_fitur(X, nama):
    X = np.asarray(X, dtype=np.float32)

    if X.ndim < 3:
        raise ValueError(
            f"{nama} harus berbentuk [N, T, F] atau [N, T, ...]. "
            f"Bentuk yang diterima: {X.shape}"
        )

    X = X.reshape(X.shape[0], X.shape[1], -1)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X


def normalisasi(X_train, X_val, X_test=None):
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    if X_test is not None:
        X_test = (X_test - mean) / std

    return X_train, X_val, X_test


class ModelLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        jumlah_kelas,
        dropout
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            hidden_size * 2,
            jumlah_kelas
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        fitur = output[:, -1, :]
        fitur = self.dropout(fitur)

        return self.classifier(fitur)


def hitung_macro_f1(y_true, y_pred):
    skor = []

    for kelas in np.unique(np.concatenate([y_true, y_pred])):
        tp = np.sum((y_true == kelas) & (y_pred == kelas))
        fp = np.sum((y_true != kelas) & (y_pred == kelas))
        fn = np.sum((y_true == kelas) & (y_pred != kelas))

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        skor.append(f1)

    return float(np.mean(skor))


def latih_dan_prediksi(seed, final=False):
    atur_seed(seed)

    Xtr, ytr, Xva, yva, Xte, yte = data.muat()

    Xtr = persiapkan_fitur(Xtr, "X_train")
    Xva = persiapkan_fitur(Xva, "X_val")

    Xte = (
        persiapkan_fitur(Xte, "X_test")
        if final and Xte is not None else None
    )

    ytr = np.asarray(ytr, dtype=np.int64).reshape(-1)
    yva = np.asarray(yva, dtype=np.int64).reshape(-1)

    if len(Xtr) != len(ytr) or len(Xva) != len(yva):
        raise ValueError("Jumlah data fitur dan label tidak sama.")

    kelas = np.unique(ytr)
    jumlah_kelas = len(kelas)

    if jumlah_kelas < 2:
        raise ValueError("Data latih harus memiliki minimal dua kelas.")

    if not np.isin(yva, kelas).all():
        raise ValueError(
            "Terdapat kelas validasi yang tidak ditemukan pada data latih."
        )

    Xtr, Xva, Xte = normalisasi(Xtr, Xva, Xte)

    X_train = torch.tensor(Xtr, dtype=torch.float32)
    y_train = torch.tensor(ytr, dtype=torch.long)

    X_valid = torch.tensor(Xva, dtype=torch.float32)

    dataset = TensorDataset(X_train, y_train)

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        generator=generator
    )

    model = ModelLSTM(
        input_size=Xtr.shape[2],
        hidden_size=64,
        jumlah_kelas=jumlah_kelas,
        dropout=0.3
    ).to(DEVICE)

    frekuensi = np.bincount(
        np.searchsorted(kelas, ytr),
        minlength=jumlah_kelas
    )

    bobot = len(ytr) / (
        jumlah_kelas * np.maximum(frekuensi, 1)
    )

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(bobot, dtype=torch.float32)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )

    epoch_maksimal = 40
    patience = 8
    terbaik = -1.0
    epoch_tanpa_perbaikan = 0
    bobot_terbaik = None

    for epoch in range(epoch_maksimal):
        model.train()

        for xb, yb in loader:
            optimizer.zero_grad()

            output = model(xb)
            loss = criterion(output, yb)

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

        model.eval()

        with torch.no_grad():
            prediksi = model(X_valid).argmax(dim=1).numpy()

        skor = hitung_macro_f1(yva, kelas[prediksi])

        print(
            f"Epoch {epoch + 1:02d}/{epoch_maksimal} "
            f"| Macro-F1: {skor:.4f}"
        )

        if skor > terbaik:
            terbaik = skor
            bobot_terbaik = {
                k: v.detach().clone()
                for k, v in model.state_dict().items()
            }
            epoch_tanpa_perbaikan = 0
        else:
            epoch_tanpa_perbaikan += 1

        if epoch_tanpa_perbaikan >= patience:
            print("Early stopping aktif.")
            break

    model.load_state_dict(bobot_terbaik)
    model.eval()

    with torch.no_grad():
        pred_val = kelas[
            model(X_valid).argmax(dim=1).numpy()
        ]

        pred_test = None

        if final and Xte is not None:
            X_test = torch.tensor(Xte, dtype=torch.float32)
            pred_test = kelas[
                model(X_test).argmax(dim=1).numpy()
            ]

    return pred_val, pred_test


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--perangkat", default="cpu laptop")

    args = parser.parse_args()

    waktu_mulai = time.time()

    pred_val, pred_test = latih_dan_prediksi(
        args.seed,
        args.final
    )

    durasi = time.time() - waktu_mulai

    if durasi > 1200:
        print("PERINGATAN: Durasi pelatihan melebihi 20 menit.")

    output = AKAR / "results" / NIM
    output.mkdir(parents=True, exist_ok=True)

    path_val = output / "pred_val.npy"
    np.save(path_val, pred_val)

    perintah = [
        sys.executable,
        "-m",
        "src.common.evaluate",
        "--nim", NIM,
        "--model", JALUR,
        "--seed", str(args.seed),
        "--pred-val", str(path_val),
        "--durasi", f"{durasi:.1f}",
        "--perangkat", args.perangkat
    ]

    if pred_test is not None:
        path_test = output / "pred_test.npy"
        np.save(path_test, pred_test)
        perintah.extend(["--pred-test", str(path_test)])

    subprocess.run(
        perintah,
        check=True,
        cwd=AKAR
    )


if __name__ == "__main__":
    main()

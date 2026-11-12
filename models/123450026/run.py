
from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import torch


# ============================================================
# KONFIGURASI
# ============================================================

AKAR = Path(__file__).resolve().parents[2]
DIREKTORI_MODEL = Path(__file__).resolve().parent
NIM = DIREKTORI_MODEL.name

JALUR_MODEL = DIREKTORI_MODEL / "model.py"

JUMLAH_SAMPEL = 4
PANJANG_SEKUENS = 90
JUMLAH_FITUR = 99
JUMLAH_KELAS = 6

HIDDEN_SIZE = 64
DROPOUT = 0.3


# ============================================================
# PENGATURAN SEED
# ============================================================

def atur_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# MEMUAT MODEL
# ============================================================

def muat_modul_model():
    if not JALUR_MODEL.exists():
        raise FileNotFoundError(
            f"File model.py tidak ditemukan: {JALUR_MODEL}"
        )

    spesifikasi = importlib.util.spec_from_file_location(
        "model_123450066",
        JALUR_MODEL
    )

    if spesifikasi is None or spesifikasi.loader is None:
        raise ImportError("Modul model.py tidak dapat dimuat.")

    modul = importlib.util.module_from_spec(spesifikasi)
    sys.modules[spesifikasi.name] = modul
    spesifikasi.loader.exec_module(modul)

    return modul


def buat_model(modul, nama_model: str):
    daftar_model = {
        "rnn": "ModelRNN",
        "lstm": "ModelLSTM",
        "gru": "ModelGRU",
    }

    nama_kelas = daftar_model[nama_model]

    if not hasattr(modul, nama_kelas):
        raise AttributeError(
            f"{nama_kelas} tidak ditemukan di model.py. "
            f"Pastikan kelas tersebut sudah didefinisikan."
        )

    kelas_model = getattr(modul, nama_kelas)

    return kelas_model(
        input_size=JUMLAH_FITUR,
        hidden_size=HIDDEN_SIZE,
        jumlah_kelas=JUMLAH_KELAS,
        dropout=DROPOUT
    )


# ============================================================
# SMOKE TEST
# ============================================================

def smoke_test(nama_model: str, seed: int) -> bool:
    atur_seed(seed)

    perangkat = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print(f"SMOKE TEST MODEL: {nama_model.upper()}")
    print("=" * 60)

    modul = muat_modul_model()
    model = buat_model(modul, nama_model).to(perangkat)

    model.eval()

    data_dummy = torch.randn(
        JUMLAH_SAMPEL,
        PANJANG_SEKUENS,
        JUMLAH_FITUR,
        device=perangkat
    )

    with torch.no_grad():
        prediksi = model(data_dummy)

    bentuk_diharapkan = (JUMLAH_SAMPEL, JUMLAH_KELAS)

    if tuple(prediksi.shape) != bentuk_diharapkan:
        raise ValueError(
            f"Bentuk output tidak sesuai. "
            f"Diharapkan {bentuk_diharapkan}, "
            f"tetapi diperoleh {tuple(prediksi.shape)}."
        )

    if not torch.isfinite(prediksi).all():
        raise ValueError("Output model mengandung NaN atau Inf.")

    print(f"Model       : {nama_model.upper()}")
    print(f"Perangkat   : {perangkat}")
    print(f\Input       : {tuple(data_dummy.shape)}")
    print(f"Output      : {tuple(prediksi.shape)}")
    print("Status      : BERHASIL")
    print("=" * 60)

    return True


# ============================================================
# EKSEKUSI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Runner model RNN - Capstone Deep Learning"
    )

    parser.add_argument(
        "--model",
        choices=["rnn", "lstm", "gru", "all"],
        default="rnn",
        help="Model yang akan diuji."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed untuk reproduksibilitas."
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Menjalankan pengujian bentuk input dan output model."
    )

    args = parser.parse_args()

    if not args.smoke_test:
        print(
            "Pelatihan belum dijalankan karena pipeline dataset "
            "belum siap."
        )
        print(
            "Implementasikan src/protokol/data.py terlebih dahulu, "
            "kemudian hubungkan proses training dan evaluasi."
        )
        sys.exit(1)

    daftar_model = (
        ["rnn", "lstm", "gru"]
        if args.model == "all"
        else [args.model]
    )

    for nama_model in daftar_model:
        smoke_test(nama_model, args.seed)

    print("\nSeluruh smoke test berhasil.")


if __name__ == "__main__":
    main()

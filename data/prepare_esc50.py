"""
prepare_esc50.py — Preparazione del dataset ESC-50 per il fine-tuning MAE-AST.

ESC-50 va scaricato da: https://github.com/karolpiczak/ESC-50
Struttura attesa:
    ESC-50/
    ├── audio/           ← 2000 file .wav (5 secondi ciascuno, 44.1kHz)
    └── meta/
        └── esc50.csv    ← metadata: filename, fold, target, category, ...

Questo script:
  1. Legge il CSV metadata di ESC-50
  2. Verifica che i file .wav esistano su disco
  3. Genera JSON per la 5-fold cross-validation:
     - esc50_train_fold{1-5}.json  → 4 fold per training
     - esc50_eval_fold{1-5}.json   → 1 fold per test
  4. Stampa statistiche di riepilogo (classi, distribuzione fold)

Formato JSON output:
{
    "data": [
        {"wav": "/path/to/audio/1-100032-A-0.wav", "labels": "0"},
        ...
    ]
}

Le label sono l'indice numerico della classe (0-49).

Valutazione: per ogni fold k (1-5), si allena sui 4 fold rimanenti
e si testa sul fold k. Il risultato finale è la media delle 5 accuracy.

Uso:
    python prepare_esc50.py --esc50_path /path/to/ESC-50 --output_dir ./datafiles
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from collections import defaultdict


def read_esc50_metadata(csv_path: str) -> list[dict]:
    """
    Legge il CSV metadata di ESC-50.

    Formato CSV:
        filename,fold,target,category,esc10,src_file,take
        1-100032-A-0.wav,1,0,dog,True,100032,A

    Returns:
        Lista di dict con chiavi: filename, fold, target, category
    """
    entries = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append({
                "filename": row["filename"],
                "fold": int(row["fold"]),
                "target": int(row["target"]),
                "category": row["category"],
            })
    return entries


def check_audio_files(entries: list[dict], audio_dir: str) -> tuple[list[dict], list[dict]]:
    """
    Verifica quali file .wav esistono su disco.

    Returns:
        (found, missing): tuple di due liste di entries
    """
    found = []
    missing = []
    for entry in entries:
        wav_path = os.path.join(audio_dir, entry["filename"])
        if os.path.isfile(wav_path):
            entry_with_path = entry.copy()
            entry_with_path["wav_path"] = wav_path
            found.append(entry_with_path)
        else:
            missing.append(entry)
    return found, missing


def build_fold_jsons(entries: list[dict], num_folds: int = 5) -> dict:
    """
    Genera i JSON per la 5-fold cross-validation.

    Per ogni fold k:
      - train: tutti gli entries con fold != k
      - eval: tutti gli entries con fold == k

    Returns:
        Dict con chiavi "train_fold{k}" e "eval_fold{k}" per k=1..5,
        ogni valore è un dict nel formato {"data": [...]}.
    """
    jsons = {}

    for k in range(1, num_folds + 1):
        train_data = []
        eval_data = []

        for entry in sorted(entries, key=lambda x: x["filename"]):  # sorted per riproducibilità
            item = {
                "wav": entry["wav_path"],
                "labels": str(entry["target"]),
            }
            if entry["fold"] == k:
                eval_data.append(item)
            else:
                train_data.append(item)

        jsons[f"train_fold{k}"] = {"data": train_data}
        jsons[f"eval_fold{k}"] = {"data": eval_data}

    return jsons


def save_json(manifest: dict, output_path: str):
    """Salva il manifest JSON su disco."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Salvato: {output_path} ({len(manifest['data'])} clip)")


def print_dataset_stats(entries: list[dict]):
    """Stampa statistiche sul dataset."""

    # Distribuzione per fold
    fold_counts = defaultdict(int)
    for entry in entries:
        fold_counts[entry["fold"]] += 1

    print("  Distribuzione per fold:")
    for fold in sorted(fold_counts):
        print(f"    Fold {fold}: {fold_counts[fold]} clip")

    # Numero classi
    categories = set(entry["category"] for entry in entries)
    targets = set(entry["target"] for entry in entries)
    print(f"  Classi: {len(targets)} (target 0-{max(targets)})")

    # Clip per classe
    class_counts = defaultdict(int)
    for entry in entries:
        class_counts[entry["category"]] += 1

    counts = list(class_counts.values())
    print(f"  Clip per classe: {min(counts)} min, {max(counts)} max, {sum(counts)/len(counts):.0f} media")


def main():
    parser = argparse.ArgumentParser(
        description="Prepara il dataset ESC-50 per il fine-tuning MAE-AST"
    )
    parser.add_argument(
        "--esc50_path",
        type=str,
        required=True,
        help="Percorso root di ESC-50 (contiene audio/ e meta/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./datafiles",
        help="Cartella output per i file JSON (default: ./datafiles)",
    )
    args = parser.parse_args()

    esc50_path = Path(args.esc50_path)
    output_dir = Path(args.output_dir)

    # ------------------------------------------------------------------
    # Verifica che le cartelle necessarie esistano
    # ------------------------------------------------------------------
    audio_dir = esc50_path / "audio"
    meta_csv = esc50_path / "meta" / "esc50.csv"

    print("=" * 60)
    print("Preparazione ESC-50 per MAE-AST")
    print("=" * 60)
    print(f"  ESC-50 path:  {esc50_path}")
    print(f"  Output dir:   {output_dir}")
    print()

    errors = []
    if not audio_dir.exists():
        errors.append(f"  MANCANTE: {audio_dir}")
    if not meta_csv.exists():
        errors.append(f"  MANCANTE: {meta_csv}")

    if errors:
        print("[ERRORE] File o cartelle mancanti:")
        for e in errors:
            print(e)
        print()
        print("Scarica ESC-50 da: https://github.com/karolpiczak/ESC-50")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Leggi il CSV metadata
    # ------------------------------------------------------------------
    print("[1/4] Lettura metadata CSV...")
    entries = read_esc50_metadata(str(meta_csv))
    print(f"  Trovate {len(entries)} entries nel CSV")
    print()

    # ------------------------------------------------------------------
    # 2. Verifica file audio
    # ------------------------------------------------------------------
    print("[2/4] Verifica file audio su disco...")
    found, missing = check_audio_files(entries, str(audio_dir))
    print(f"  Trovati: {len(found)}, mancanti: {len(missing)}")

    if missing:
        print(f"  [WARN] {len(missing)} file mancanti. Primi 5:")
        for entry in missing[:5]:
            print(f"    {entry['filename']}")
    print()

    # ------------------------------------------------------------------
    # 3. Statistiche dataset
    # ------------------------------------------------------------------
    print("[3/4] Statistiche dataset:")
    print_dataset_stats(found)
    print()

    # ------------------------------------------------------------------
    # 4. Genera i file JSON per 5-fold cross-validation
    # ------------------------------------------------------------------
    print("[4/4] Generazione file JSON (5-fold cross-validation)...")
    fold_jsons = build_fold_jsons(found)

    for name, manifest in fold_jsons.items():
        json_path = output_dir / f"esc50_{name}.json"
        save_json(manifest, str(json_path))

    # ------------------------------------------------------------------
    # Riepilogo finale
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("RIEPILOGO")
    print("=" * 60)
    print(f"  Clip totali: {len(found)}")
    print(f"  Classi: 50")
    print(f"  Fold: 5 (400 clip ciascuno)")
    print(f"  File generati: 10 JSON (5 train + 5 eval)")
    print()
    print("  Per ogni fold k:")
    sample_train = fold_jsons["train_fold1"]
    sample_eval = fold_jsons["eval_fold1"]
    print(f"    Train: {len(sample_train['data'])} clip (4 fold)")
    print(f"    Eval:  {len(sample_eval['data'])} clip (1 fold)")
    print()
    print("Prossimo passo: configurazione iperparametri in configs/config.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
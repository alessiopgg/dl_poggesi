"""
prepare_fsd50k.py — Preparazione del dataset FSD50K per il pretraining MAE-AST.

FSD50K va scaricato manualmente da Zenodo: https://zenodo.org/records/4060432
Servono:
  - FSD50K.dev_audio/       → 40966 clip (~80h) — usato per pretraining
  - FSD50K.eval_audio/      → 10231 clip (~28h) — usato per validazione
  - FSD50K.ground_truth/    → dev.csv, eval.csv, vocabulary.csv

Questo script:
  1. Legge i CSV ground truth per ottenere la lista dei filename
  2. Verifica che i file .wav esistano su disco
  3. (Opzionale) Calcola la durata di ogni clip per statistiche
  4. Genera due JSON in formato SSAST-style:
     - fsd50k_train.json  (da dev_audio, per pretraining)
     - fsd50k_eval.json   (da eval_audio, per validazione)
  5. Stampa statistiche di riepilogo

Formato JSON output:
{
    "data": [
        {"wav": "/path/to/FSD50K.dev_audio/64760.wav", "labels": "dummy"},
        ...
    ]
}

Per il pretraining self-supervised le label sono "dummy" — non servono.

Uso:
    python prepare_fsd50k.py --fsd50k_path /path/to/FSD50K --output_dir ./datafiles
    python prepare_fsd50k.py --fsd50k_path /path/to/FSD50K --output_dir ./datafiles --compute_stats
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path


def read_ground_truth_csv(csv_path: str) -> list[str]:
    """
    Legge un CSV ground truth di FSD50K e restituisce la lista di fname (senza estensione).

    Formato CSV FSD50K:
        fname,labels,mids,split
        64760,"Fart","/m/07pjjrj","train"

    Returns:
        Lista di stringhe fname, es: ["64760", "16399", ...]
    """
    fnames = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fnames.append(row["fname"])
    return fnames


def check_audio_files(fnames: list[str], audio_dir: str) -> tuple[list[str], list[str]]:
    """
    Verifica quali file .wav esistono effettivamente su disco.

    Args:
        fnames: lista di filename (senza estensione)
        audio_dir: percorso della cartella audio (es: FSD50K.dev_audio/)

    Returns:
        (found, missing): tuple di due liste di path completi
    """
    found = []
    missing = []
    for fname in fnames:
        wav_path = os.path.join(audio_dir, f"{fname}.wav")
        if os.path.isfile(wav_path):
            found.append(wav_path)
        else:
            missing.append(wav_path)
    return found, missing


def compute_duration_stats(wav_paths: list[str]) -> dict:
    """
    Calcola statistiche sulle durate dei file audio.
    Richiede soundfile o torchaudio.

    Returns:
        Dict con min, max, mean, total durate in secondi.
    """
    try:
        import soundfile as sf
    except ImportError:
        print("[WARN] soundfile non installato. Installa con: pip install soundfile")
        print("       Skippo il calcolo delle statistiche di durata.")
        return None

    durations = []
    for i, wav_path in enumerate(wav_paths):
        try:
            info = sf.info(wav_path)
            durations.append(info.duration)
        except Exception as e:
            print(f"[WARN] Errore leggendo {wav_path}: {e}")

        # Progress ogni 5000 file
        if (i + 1) % 5000 == 0:
            print(f"  Analizzati {i + 1}/{len(wav_paths)} file...")

    if not durations:
        return None

    return {
        "num_clips": len(durations),
        "total_hours": sum(durations) / 3600,
        "mean_sec": sum(durations) / len(durations),
        "min_sec": min(durations),
        "max_sec": max(durations),
    }


def build_json_manifest(wav_paths: list[str], use_dummy_labels: bool = True) -> dict:
    """
    Costruisce il manifest JSON in formato SSAST-style.

    Args:
        wav_paths: lista di percorsi completi ai file .wav
        use_dummy_labels: se True, tutte le label sono "dummy" (per pretraining)

    Returns:
        Dict nel formato {"data": [{"wav": "...", "labels": "..."}, ...]}
    """
    data = []
    for wav_path in sorted(wav_paths):  # sorted per riproducibilità
        entry = {
            "wav": wav_path,
            "labels": "dummy" if use_dummy_labels else "",
        }
        data.append(entry)
    return {"data": data}


def save_json(manifest: dict, output_path: str):
    """Salva il manifest JSON su disco."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Salvato: {output_path} ({len(manifest['data'])} clip)")


def main():
    parser = argparse.ArgumentParser(
        description="Prepara il dataset FSD50K per il pretraining MAE-AST"
    )
    parser.add_argument(
        "--fsd50k_path",
        type=str,
        required=True,
        help="Percorso root di FSD50K (contiene FSD50K.dev_audio/, FSD50K.eval_audio/, FSD50K.ground_truth/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./datafiles",
        help="Cartella output per i file JSON (default: ./datafiles)",
    )
    parser.add_argument(
        "--compute_stats",
        action="store_true",
        help="Se presente, calcola statistiche sulle durate (richiede soundfile)",
    )
    args = parser.parse_args()

    fsd50k_path = Path(args.fsd50k_path)
    output_dir = Path(args.output_dir)

    # ------------------------------------------------------------------
    # Verifica che le cartelle necessarie esistano
    # ------------------------------------------------------------------
    dev_audio_dir = fsd50k_path / "FSD50K.dev_audio"
    eval_audio_dir = fsd50k_path / "FSD50K.eval_audio"
    gt_dir = fsd50k_path / "FSD50K.ground_truth"

    dev_csv = gt_dir / "dev.csv"
    eval_csv = gt_dir / "eval.csv"

    print("=" * 60)
    print("Preparazione FSD50K per MAE-AST")
    print("=" * 60)
    print(f"  FSD50K path:  {fsd50k_path}")
    print(f"  Output dir:   {output_dir}")
    print()

    # Controlla che esistano le cartelle
    errors = []
    for path, desc in [
        (dev_audio_dir, "FSD50K.dev_audio/"),
        (eval_audio_dir, "FSD50K.eval_audio/"),
        (dev_csv, "FSD50K.ground_truth/dev.csv"),
        (eval_csv, "FSD50K.ground_truth/eval.csv"),
    ]:
        if not path.exists():
            errors.append(f"  MANCANTE: {path} ({desc})")

    if errors:
        print("[ERRORE] File o cartelle mancanti:")
        for e in errors:
            print(e)
        print()
        print("Scarica FSD50K da: https://zenodo.org/records/4060432")
        print("Servono: FSD50K.dev_audio.z01-z08 + .zip, FSD50K.eval_audio.zip, FSD50K.ground_truth.zip")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Leggi i CSV ground truth
    # ------------------------------------------------------------------
    print("[1/4] Lettura CSV ground truth...")
    dev_fnames = read_ground_truth_csv(str(dev_csv))
    eval_fnames = read_ground_truth_csv(str(eval_csv))
    print(f"  dev.csv:  {len(dev_fnames)} entries")
    print(f"  eval.csv: {len(eval_fnames)} entries")
    print()

    # ------------------------------------------------------------------
    # 2. Verifica che i file audio esistano
    # ------------------------------------------------------------------
    print("[2/4] Verifica file audio su disco...")
    dev_found, dev_missing = check_audio_files(dev_fnames, str(dev_audio_dir))
    eval_found, eval_missing = check_audio_files(eval_fnames, str(eval_audio_dir))

    print(f"  Dev  — trovati: {len(dev_found)}, mancanti: {len(dev_missing)}")
    print(f"  Eval — trovati: {len(eval_found)}, mancanti: {len(eval_missing)}")

    if dev_missing:
        print(f"  [WARN] {len(dev_missing)} file dev mancanti. Primi 5:")
        for p in dev_missing[:5]:
            print(f"    {p}")
    if eval_missing:
        print(f"  [WARN] {len(eval_missing)} file eval mancanti. Primi 5:")
        for p in eval_missing[:5]:
            print(f"    {p}")
    print()

    # ------------------------------------------------------------------
    # 3. (Opzionale) Calcola statistiche durate
    # ------------------------------------------------------------------
    if args.compute_stats:
        print("[3/4] Calcolo statistiche durate (può richiedere qualche minuto)...")
        print("  Analisi dev set:")
        dev_stats = compute_duration_stats(dev_found)
        if dev_stats:
            print(f"    Clip: {dev_stats['num_clips']}")
            print(f"    Durata totale: {dev_stats['total_hours']:.1f} ore")
            print(f"    Media: {dev_stats['mean_sec']:.1f}s | Min: {dev_stats['min_sec']:.1f}s | Max: {dev_stats['max_sec']:.1f}s")

        print("  Analisi eval set:")
        eval_stats = compute_duration_stats(eval_found)
        if eval_stats:
            print(f"    Clip: {eval_stats['num_clips']}")
            print(f"    Durata totale: {eval_stats['total_hours']:.1f} ore")
            print(f"    Media: {eval_stats['mean_sec']:.1f}s | Min: {eval_stats['min_sec']:.1f}s | Max: {eval_stats['max_sec']:.1f}s")
        print()
    else:
        print("[3/4] Statistiche durate skippate (usa --compute_stats per calcolarle)")
        print()

    # ------------------------------------------------------------------
    # 4. Genera i file JSON
    # ------------------------------------------------------------------
    print("[4/4] Generazione file JSON...")

    # Train JSON (da dev set, per pretraining self-supervised)
    train_manifest = build_json_manifest(dev_found, use_dummy_labels=True)
    train_json_path = output_dir / "fsd50k_train.json"
    save_json(train_manifest, str(train_json_path))

    # Eval JSON (da eval set, per validazione)
    eval_manifest = build_json_manifest(eval_found, use_dummy_labels=True)
    eval_json_path = output_dir / "fsd50k_eval.json"
    save_json(eval_manifest, str(eval_json_path))

    # ------------------------------------------------------------------
    # Riepilogo finale
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("RIEPILOGO")
    print("=" * 60)
    print(f"  Training set (dev):    {len(dev_found)} clip → {train_json_path}")
    print(f"  Validation set (eval): {len(eval_found)} clip → {eval_json_path}")
    if dev_missing or eval_missing:
        print(f"  File mancanti:         {len(dev_missing) + len(eval_missing)} totali")
    print(f"  Label:                 'dummy' (pretraining self-supervised)")
    print()
    print("Prossimo passo: prepara ESC-50 con prepare_esc50.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
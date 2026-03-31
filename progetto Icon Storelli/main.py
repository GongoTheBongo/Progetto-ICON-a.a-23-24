"""
Main per l'esecuzione degli script del progetto.
Esegue gli script principali come processi separati (per evitare conflitti di variabili e plotting simultaneo).
Uso:
    python main.py

Esegue in ordine:
1) Supervisionato con dataset standard
2) Fuzzy C-Means per factor loadings (salva CSV fattori)
3) Supervisionato con dataset fattori

Nota: gli script vengono eseguiti come nuovi processi; per visualizzare i plot, chiudere le figure o utilizzare l'output nei singoli script.
"""

from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
# Percorso di default all'interprete virtualenv (aggiornato quando possibile)
DEFAULT_PYTHON = PROJECT_ROOT / "env" / "Scripts" / "python.exe"
DEFAULT_DATASET = PROJECT_ROOT / "Ontologia" / "Autism-Dataset.csv"
DEFAULT_FACTORS_OUT = PROJECT_ROOT / "Clustering" / "Autism-Dataset-factors.csv"

SUPERVISED_SCRIPTS = [
    PROJECT_ROOT / 'Supervised_learning' / 'KNN.py',
    PROJECT_ROOT / 'Supervised_learning' / 'SVM.py',
    PROJECT_ROOT / 'Supervised_learning' / 'RandomForest.py'
]

def _maybe_reexec_with_venv() -> None:
    """Riavvia lo script con il Python del virtualenv se disponibile."""
    if os.environ.get("SKIP_VENV_REEXEC") == "1":
        return
    if not DEFAULT_PYTHON.exists():
        return
    try:
        current = Path(sys.executable).resolve()
    except Exception:
        current = Path(sys.executable)
    if current != DEFAULT_PYTHON.resolve():
        os.environ["SKIP_VENV_REEXEC"] = "1"
        print(f"[INFO] Riavvio con virtualenv: {DEFAULT_PYTHON}")
        os.execv(str(DEFAULT_PYTHON), [str(DEFAULT_PYTHON), *sys.argv])


def run_script(python_exe: str, script_path: Path, extra_env: dict | None = None) -> int:
    """Esegue lo script con l'interprete specificato e mostra l'output in tempo reale."""
    if not script_path.exists():
        print(f" Script non trovato: {script_path.name}")
        return 2

    print(f"\n Avvio: {script_path.name} con {python_exe}")
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    proc = subprocess.Popen(
        [python_exe, str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )

    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            print(line, end='')
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
        print('\n Esecuzione interrotta dall\'utente')
        return 130

    ret = proc.wait()
    if ret == 0:
        print(f"Completato: {script_path.name} (exit {ret})")
    else:
        print(f"Errore ({ret}) durante l'esecuzione di {script_path.name}")
    return ret


def run_supervised_pipeline(python_exe: str, dataset_env: dict | None = None) -> list[tuple[Path, int]]:
    results: list[tuple[Path, int]] = []
    for script in SUPERVISED_SCRIPTS:
        code = run_script(python_exe, script, extra_env=dataset_env)
        results.append((script, code))
    return results


def run_supervised_fcm_pipeline(python_exe: str, dataset_path: Path, factors_out: Path) -> int:
    from Clustering.Fuzzy_C_Means import build_factor_dataset

    print("\n=== STEP 1: Esecuzione supervisionato (dataset originale) ===")
    step1 = run_supervised_pipeline(python_exe, dataset_env={"FAST_MODE": "1"})

    print("\n=== STEP 2: Fuzzy C-Means per factor loadings ===")
    factors_path, best_k = build_factor_dataset(dataset_path, factors_out)
    print(f"[INFO] Fattori generati: {best_k}")

    print("\n=== STEP 3: Esecuzione supervisionato (dataset fattori) ===")
    env = {
        "DATASET_PATH": str(factors_path),
        "DATASET_FORMAT": "factors",
        "FAST_MODE": "1"
    }
    step3 = run_supervised_pipeline(python_exe, dataset_env=env)

    print("\n--- Riepilogo pipeline supervisionato+FCM ---")
    for script, code in step1 + step3:
        status = "OK" if code == 0 else f"FAILED (code {code})"
        print(f"{script.name}: {status}")

    for _, code in step1 + step3:
        if code != 0:
            return code
    return 0


def main() -> int:
    _maybe_reexec_with_venv()
    python_exe = str(DEFAULT_PYTHON) if DEFAULT_PYTHON.exists() else sys.executable
    return run_supervised_fcm_pipeline(
        python_exe=python_exe,
        dataset_path=DEFAULT_DATASET,
        factors_out=DEFAULT_FACTORS_OUT
    )


if __name__ == '__main__':
    raise SystemExit(main())

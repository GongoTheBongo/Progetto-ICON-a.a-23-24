"""
Main per l'esecuzione degli script del progetto.
Esegue gli script principali come processi separati (per evitare conflitti di variabili e plotting simultaneo).
Uso:
    python main.py --all
    python main.py --knn --svm
    python main.py --script Supervised_learning/KNN.py

Opzioni principali:
    --all       : esegue tutti gli script predefiniti
    --knn       : esegue `Supervised_learning/KNN.py`
    --svm       : esegue `Supervised_learning/SVM.py`
    --rf        : esegue `Supervised_learning/RandomForest.py`
    --kmeans    : esegue `Clustering/KMeans.py`
    --grafici   : esegue `Ontologia/grafico_1.py`
    --ontologia : esegue `Ontologia/Query_ontologia.py`
    --script    : esegue uno script qualsiasi passando il path
    --python    : percorso all'interprete Python da usare (di default usa l'interprete dell'ambiente venv del progetto)

Nota: gli script vengono eseguiti come nuovi processi; per visualizzare i plot, chiudere le figure o utilizzare l'output nei singoli script.
"""

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

# Percorso di default all'interprete virtualenv (aggiornato quando possibile)
DEFAULT_PYTHON = r"C:/Users/store/OneDrive/Desktop/File per l' università/Terzo anno/Ingegneria della conoscenza/progetto Icon Storelli/env/Scripts/python.exe"

PROJECT_ROOT = Path(__file__).parent

def list_py_scripts(dir_path: Path) -> List[Path]:
    """Ritorna i file .py (ordinati) in una directory, escludendo file privati e __init__."""
    return sorted([p for p in dir_path.glob('*.py') if p.is_file() and not p.name.startswith('_')])

SCRIPTS = {
    'grafici_1': PROJECT_ROOT / 'Ontologia' / 'grafico_1.py',
    'ontologia': PROJECT_ROOT / 'Ontologia' / 'Query_ontologia.py',
    'knn': PROJECT_ROOT / 'Supervised_learning' / 'KNN.py',
    'svm': PROJECT_ROOT / 'Supervised_learning' / 'SVM.py',
    'rf': PROJECT_ROOT / 'Supervised_learning' / 'RandomForest.py',
    'kmeans': PROJECT_ROOT / 'Clustering' / 'KMeans.py'
} 


def run_script(python_exe: str, script_path: Path) -> int:
    """Esegue lo script con l'interprete specificato e mostra l'output in tempo reale."""
    if not script_path.exists():
        print(f"⚠️  Script non trovato: {script_path.name}")
        return 2

    print(f"\n✅ Avvio: {script_path.name} con {python_exe}")
    proc = subprocess.Popen([python_exe, str(script_path)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            print(line, end='')
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
        print('\n✋ Esecuzione interrotta dall\'utente')
        return 130

    ret = proc.wait()
    if ret == 0:
        print(f"✅ Completato: {script_path.name} (exit {ret})")
    else:
        print(f"❌ Errore ({ret}) durante l'esecuzione di {script_path.name}")
    return ret


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Main per l\'esecuzione degli script del progetto')
    parser.add_argument('--all', action='store_true', help='Esegui tutti gli script principali')
    parser.add_argument('--knn', action='store_true')
    parser.add_argument('--svm', action='store_true')
    parser.add_argument('--rf', action='store_true')
    parser.add_argument('--kmeans', action='store_true')
    parser.add_argument('--grafici', action='store_true', help='Esegue grafico_1')
    parser.add_argument('--ontologia', action='store_true')
    parser.add_argument('--script', type=str, help='Esegui uno script specifico (path relativo al progetto)')
    parser.add_argument('--python', type=str, default=DEFAULT_PYTHON, help='Percorso all\'interprete Python da usare')

    args = parser.parse_args(argv)

    python_exe = args.python

    # Se non viene passato nulla, esegui tutti
    to_run: List[Path] = []
    if args.all or not any([args.knn, args.svm, args.rf, args.kmeans, args.grafici, args.ontologia, args.script]):
        # Esegui tutti: prima `Ontologia`, poi `Clustering`, poi `Supervised_learning`
        to_run.extend(list_py_scripts(PROJECT_ROOT / 'Ontologia'))
        to_run.extend(list_py_scripts(PROJECT_ROOT / 'Clustering'))
        to_run.extend(list_py_scripts(PROJECT_ROOT / 'Supervised_learning'))
    else:
        if args.knn:
            to_run.append(SCRIPTS['knn'])
        if args.svm:
            to_run.append(SCRIPTS['svm'])
        if args.rf:
            to_run.append(SCRIPTS['rf'])
        if args.kmeans:
            to_run.append(SCRIPTS['kmeans'])
        if args.grafici:
            # Mantiene compatibilità con --grafici (esegue `grafico_1.py`)
            to_run.append(SCRIPTS['grafici_1'])
        if args.ontologia:
            to_run.append(SCRIPTS['ontologia'])
        if args.script:
            to_run.append(PROJECT_ROOT / args.script)

    # Scambia l'ordine di esecuzione di SVM e run_experiments_top3 se entrambi presenti
    svm_path = SCRIPTS['svm']
    run_top3_path = PROJECT_ROOT / 'Supervised_learning' / 'run_experiments_top3.py'
    if svm_path in to_run and run_top3_path in to_run:
        i_svm = to_run.index(svm_path)
        i_top3 = to_run.index(run_top3_path)
        # Swap positions
        to_run[i_svm], to_run[i_top3] = to_run[i_top3], to_run[i_svm]
        

    # Esecuzione sequenziale per ridurre conflitti su risorse e plotting
    exit_codes = []
    for script in to_run:
        code = run_script(python_exe, script)
        exit_codes.append((script, code))

    print('\n--- Riepilogo esecuzioni ---')
    for script, code in exit_codes:
        status = 'OK' if code == 0 else f'FAILED (code {code})'
        print(f"{script.name}: {status}")

    # Se qualcuno è fallito, ritorna il codice non-zero del primo fallito
    for _, code in exit_codes:
        if code != 0:
            return code
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

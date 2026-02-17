# Progetto di Machine Learning - Riconoscimento Autismo

Progetto universitario di Machine Learning per la classificazione dello spettro autistico (ASD) utilizzando diversi algoritmi di supervised learning.

## Struttura del Progetto

```
├── Supervised_learning/      # Algoritmi di classificazione
│   ├── SVM.py               # Support Vector Machine
│   ├── KNN.py               # K-Nearest Neighbors
│   ├── RandomForest.py      # Random Forest
│   ├── run_experiments_top3.py  # Esperimenti comparativi
│   └── experiments_top3_results.csv  # Risultati esperimenti
├── Clustering/              # Algoritmi di clustering
│   └── KMeans.py            # K-Means clustering
├── Ontologia/               # Ontologia e dataset
│   ├── ontologia.owl        # File ontologia OWL
│   ├── Autism-Dataset.csv   # Dataset
│   ├── Query_ontologia.py   # Query SPARQL
│   └── grafico_1.py        # Visualizzazioni
├── main.py                  # Script principale
├── requirements.txt         # Dipendenze Python
└── README.md               # Questo file
```

## Algoritmi Utilizzati

### Supervised Learning
- **SVM (Support Vector Machine)**: Classificatore con kernel RBF e lineare
- **KNN (K-Nearest Neighbors)**: Classificatore basato sui vicini
- **Random Forest**: Ensemble di alberi decisionali

### Preprocessing
- **SMOTE**: Per il bilanciamento delle classi
- **One-Hot Encoding**: Per le feature categoriche
- **Gestione missing values**: Con la moda per ogni colonna

### Metriche di Valutazione
- Accuracy
- Precision
- Recall
- F1-Score

## Installazione

1. Clona il repository:
```bash
git clone <url-repository>
```

2. Crea un ambiente virtuale (opzionale ma consigliato):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Installa le dipendenze:
```bash
pip install -r requirements.txt
```

## Utilizzo

### Eseguire un classificatore specifico

```bash
# SVM
python Supervised_learning/SVM.py

# KNN
python Supervised_learning/KNN.py

# Random Forest
python Supervised_learning/RandomForest.py
```

### Eseguire gli esperimenti comparativi

```bash
python Supervised_learning/run_experiments_top3.py
```

Questo script:
- Trova le top-3 feature più importanti usando Random Forest
- Addestra KNN, SVM e Random Forest su tutte le feature e solo sulle top-3
- Genera grafici comparativi delle performance

## Dipendenze

- numpy
- pandas
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn
- kneed

Vedi `requirements.txt` per le versioni specifiche.

## Autori

Progetto sviluppato per il corso di Ingegneria della Conoscenza.

## Licenza

MIT License

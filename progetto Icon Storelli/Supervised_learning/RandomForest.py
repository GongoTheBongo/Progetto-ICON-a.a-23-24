"""
Questo script analizza un dataset sull'autismo utilizzando il classificatore Random Forest.
Risolti i problemi di data leakage usando Pipeline e automatizzata la ricerca degli iperparametri con GridSearchCV.
"""

import numpy as np
import os
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import random
from datetime import datetime
from pathlib import Path

# Configurazione seed per riproducibilità
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
RF_N_JOBS = int(os.environ.get("RF_N_JOBS", "-1"))
def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")

# Modalita' veloce (griglie ridotte + meno fold)
FAST_MODE = _env_flag("FAST_MODE", "1")
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from inspect import signature

# Preprocessing: one-hot SOLO per colonne stringa
def _prepare_xy_from_dataset(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X_raw = dataset.drop(columns=["Class/ASD"], errors="ignore")
    string_cols = []
    for col in X_raw.columns:
        col_numeric = pd.to_numeric(X_raw[col], errors="coerce")
        if col_numeric.notna().sum() == X_raw[col].notna().sum():
            X_raw[col] = col_numeric
        else:
            string_cols.append(col)

    data_dummies = pd.get_dummies(X_raw, columns=string_cols)
    data_dummies = data_dummies.apply(pd.to_numeric, errors='coerce')
    for col in data_dummies.columns:
        mode_val = data_dummies[col].mode()
        data_dummies[col] = data_dummies[col].fillna(mode_val.iloc[0] if len(mode_val) > 0 else 0)
    data_dummies = data_dummies.astype(float)

    y = dataset['Class/ASD'].map({'1': 1, '0': 0, 1: 1, 0: 0})
    mask = y.notna()
    if not mask.all():
        dropped = (~mask).sum()
        print(f"[WARN] Rimosse {dropped} righe con target non valido")
        X = data_dummies.loc[mask].reset_index(drop=True)
        y = y[mask].astype(int).reset_index(drop=True)
    else:
        X = data_dummies
        y = y.astype(int)

    return X, y

# Definizione delle feature e delle feature dummificate
feature = ["A1_Score","A2_Score","A3_Score","A4_Score","A5_Score","A6_Score","A7_Score","A8_Score","A9_Score","A10_Score","age","gender","ethnicity","jundice","is_autistic","screening_score","PDD_parent","Class/ASD"]
feature_dummied = ["A1_Score","A2_Score","A3_Score","A4_Score","A5_Score","A6_Score","A7_Score","A8_Score","A9_Score","A10_Score","age","gender","ethnicity","jundice","is_autistic","screening_score","PDD_parent"]

# Caricamento del dataset da un file CSV
DATASET_PATH = os.environ.get("DATASET_PATH", "Ontologia/Autism-Dataset.csv")
DATASET_FORMAT = os.environ.get("DATASET_FORMAT", "raw").lower()

def _sanitize_tag(tag: str) -> str:
    cleaned = []
    for ch in tag.strip():
        if ch.isalnum() or ch in ("-", "_"):
            cleaned.append(ch)
        else:
            cleaned.append("_")
    out = "".join(cleaned).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out

def _dataset_tag() -> str:
    tag_env = os.environ.get("DATASET_TAG", "").strip()
    if tag_env:
        return _sanitize_tag(tag_env)
    try:
        stem = Path(DATASET_PATH).stem
    except Exception:
        stem = ""
    if stem:
        return _sanitize_tag(stem)
    return _sanitize_tag(DATASET_FORMAT) or "dataset"

DATASET_TAG = _dataset_tag()

def _with_dataset_suffix(filename: str) -> str:
    if not DATASET_TAG:
        return filename
    path = Path(filename)
    return str(path.with_name(f"{path.stem}_{DATASET_TAG}{path.suffix}"))

if DATASET_FORMAT == "factors":
    dataset = pd.read_csv(DATASET_PATH)
    if "Class/ASD" not in dataset.columns:
        raise ValueError("Colonna Class/ASD non trovata nel dataset dei fattori.")
    X, y = _prepare_xy_from_dataset(dataset)
else:
    dataset = pd.read_csv(DATASET_PATH, sep=",")
    if "Class/ASD" not in dataset.columns:
        dataset = pd.read_csv(DATASET_PATH, sep=",", names=feature, 
                              dtype={'A1_Score':object,'A2_Score':object,'A3_Score':object,'A4_Score':object,'A5_Score':object,'A6_Score':object,'A7_Score':object,'A8_Score':object,'A9_Score':object,'A10_Score':object,'age':object,'gender':object,'ethnicity':object,'jundice':object,'is_autistic':object,'screening_score':object,'PDD_parent':object,'Class/ASD':object})
        dataset = dataset[dataset['Class/ASD'].notna() & (dataset['Class/ASD'].astype(str).str.strip().str.lower() != 'class/asd')].reset_index(drop=True) 
    else:
        has_factors = any(c.startswith("Factor_") for c in dataset.columns)
        if not has_factors:
            cols = [c for c in feature if c in dataset.columns]
            dataset = dataset[cols]

    X, y = _prepare_xy_from_dataset(dataset)

# Divisione dei dati in training e test set (PRIMA di applicare SMOTE!)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=RANDOM_SEED, stratify=y)

print(f"Training set: {X_train.shape[0]} campioni")
print(f"Test set: {X_test.shape[0]} campioni")
print(f"Distribuzione classi nel training set: {np.bincount(y_train.to_numpy().astype(int))}")

# Creazione della Pipeline con SMOTE e Random Forest
# Questo evita il data leakage: SMOTE viene applicato solo durante la CV sui fold di training
pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=RANDOM_SEED)),
    ('rf', RandomForestClassifier(random_state=RANDOM_SEED))
])

# Definizione della griglia di iperparametri da testare
param_grid_full = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [16, 18, 20, None],
    'rf__min_samples_leaf': [1, 2, 4]
}
param_grid_fast = {
    'rf__n_estimators': [100],
    'rf__max_depth': [16, None],
    'rf__min_samples_leaf': [1, 2]
}
param_grid = param_grid_fast if FAST_MODE else param_grid_full

# Ricerca degli iperparametri con GridSearchCV e cross-validation stratificata
cv_splits = 3 if FAST_MODE else 5
cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_SEED)
if FAST_MODE:
    grid_size = (
        len(param_grid['rf__n_estimators'])
        * len(param_grid['rf__max_depth'])
        * len(param_grid['rf__min_samples_leaf'])
    )
    print(f"[INFO] Modalita' FAST: {grid_size} combinazioni, {cv_splits}-fold CV")

def _make_grid_search(n_jobs: int) -> GridSearchCV:
    return GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='precision',
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=True
    )

grid_search = _make_grid_search(RF_N_JOBS)

print("\nAvvio ricerca iperparametri con GridSearchCV...")
print("Questo potrebbe richiedere alcuni minuti...")
try:
    grid_search.fit(X_train, y_train)
except PermissionError:
    if RF_N_JOBS != 1:
        print("[WARN] Permessi insufficienti per il multiprocessing; retry con n_jobs=1")
        grid_search = _make_grid_search(1)
        grid_search.fit(X_train, y_train)
    else:
        raise

# Stampa dei risultati della ricerca
print(f"\nMigliori iperparametri trovati:")
print(f"  n_estimators: {grid_search.best_params_['rf__n_estimators']}")
print(f"  max_depth: {grid_search.best_params_['rf__max_depth']}")
print(f"  min_samples_leaf: {grid_search.best_params_['rf__min_samples_leaf']}")
print(f"Miglior precisione in CV: {grid_search.best_score_:.4f}")

# =============================================================================
# VALUTAZIONE SUL TEST SET E ANALISI STATISTICA
# =============================================================================
# Usa direttamente la pipeline completa (SMOTE + RandomForest)
best_pipeline = grid_search.best_estimator_

# Predizione sul test set
pred = best_pipeline.predict(X_test)
precision_test = precision_score(y_test, pred)
recall_test = recall_score(y_test, pred)
f1_test = f1_score(y_test, pred)
accuracy_test = accuracy_score(y_test, pred)

print(f"\nMigliori parametri: {grid_search.best_params_}")
print(f"  Accuracy sul test set: {accuracy_test:.4f}")
print(f"  Precision sul test set: {precision_test:.4f}")
print(f"  Recall sul test set: {recall_test:.4f}")
print(f"  F1-score sul test set: {f1_test:.4f}")

# Matrice di confusione
cm = confusion_matrix(y_test, pred)
print('\nConfusion matrix:')
print(cm)

# Visualizzazione della matrice di confusione come heatmap
plt.figure(figsize=(10, 7))
df_cm = pd.DataFrame(cm, index=[i for i in "01"], columns=[i for i in "01"])
sn.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - Best Model (GridSearchCV)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(_with_dataset_suffix('rf_confusion_matrix.png'), dpi=300)


# Cross-validation sul training set usando la pipeline completa
cv_precision_scores = cross_val_score(best_pipeline, X_train, y_train, cv=cv, scoring='precision')
cv_recall_scores = cross_val_score(best_pipeline, X_train, y_train, cv=cv, scoring='recall')

print('\nStatistiche della cross-validation sul training set:')
print(f'CV Precision scores: {cv_precision_scores}')
print(f'CV Precision mean: {np.mean(cv_precision_scores):.4f}')
print(f'CV Precision variance: {np.var(cv_precision_scores):.4f}')
print(f'CV Precision std: {np.std(cv_precision_scores):.4f}')
print()
print(f'CV Recall scores: {cv_recall_scores}')
print(f'CV Recall mean: {np.mean(cv_recall_scores):.4f}')
print(f'CV Recall variance: {np.var(cv_recall_scores):.4f}')
print(f'CV Recall std: {np.std(cv_recall_scores):.4f}')

# Analisi approfondita dei risultati della GridSearch
results_df = pd.DataFrame(grid_search.cv_results_)

print("\n--- STATISTICHE GLOBALI SU TUTTE LE CONFIGURAZIONI ---")
print(f"Numero totale di configurazioni testate: {len(results_df)}")
print(f"\nPrecision in Cross-Validation:")
print(f"  Media globale: {results_df['mean_test_score'].mean():.4f}")
print(f"  Deviazione standard globale: {results_df['mean_test_score'].std():.4f}")
print(f"  Minimo: {results_df['mean_test_score'].min():.4f}")
print(f"  Massimo: {results_df['mean_test_score'].max():.4f}")

# Analisi per iperparametro
print("\n--- ANALISI PER IPERPARAMETRO ---")
ne_estimators_analysis = results_df.groupby('param_rf__n_estimators')['mean_test_score'].agg(['mean', 'std', 'min', 'max'])
print("\n--- ANALISI PER N_ESTIMATORS ---")
print(ne_estimators_analysis)

print("\n--- ANALISI PER MAX_DEPTH ---")
max_depth_analysis = results_df.groupby('param_rf__max_depth')['mean_test_score'].agg(['mean', 'std', 'min', 'max'])
print(max_depth_analysis)

print("\n--- ANALISI PER MIN_SAMPLES_LEAF ---")
min_samples_leaf_analysis = results_df.groupby('param_rf__min_samples_leaf')['mean_test_score'].agg(['mean', 'std', 'min', 'max'])
print(min_samples_leaf_analysis)

# Analisi dettagliata del miglior modello
print("\n--- ANALISI DEL MIGLIOR MODELLO ---")
best_idx = grid_search.best_index_
best_cv_scores = [results_df.loc[best_idx, f'split{i}_test_score'] for i in range(cv.get_n_splits())]

print(f"Indice nella griglia: {best_idx}")
print(f"Parametri: {grid_search.best_params_}")
print(f"\nPrecision per ogni fold della CV:")
for i, score in enumerate(best_cv_scores):
    print(f"  Fold {i+1}: {score:.4f}")

print(f"\nStatistiche aggregate dei fold:")
print(f"  Media: {np.mean(best_cv_scores):.4f}")
print(f"  Deviazione standard: {np.std(best_cv_scores):.4f}")
print(f"  Varianza: {np.var(best_cv_scores):.4f}")
print(f"  Min: {np.min(best_cv_scores):.4f}")
print(f"  Max: {np.max(best_cv_scores):.4f}")
print(f"  Range: {np.max(best_cv_scores) - np.min(best_cv_scores):.4f}")

cv_coefficient = (np.std(best_cv_scores) / np.mean(best_cv_scores)) * 100
print(f"  Coefficiente di variazione: {cv_coefficient:.2f}%")

# Confronto train vs test
best_train_score = results_df.loc[best_idx, 'mean_train_score']
best_test_score = results_df.loc[best_idx, 'mean_test_score']
overfitting_gap = best_train_score - best_test_score

print("\n--- CONFRONTO TRAIN vs TEST (Analisi Overfitting) ---")
print(f"Precision medio sul training set (CV): {best_train_score:.4f}")
print(f"Precision medio sul test set (CV): {best_test_score:.4f}")
print(f"Gap (potenziale overfitting): {overfitting_gap:.4f}")

if overfitting_gap > 0.05:
    print("[WARNING] ATTENZIONE: Gap significativo - possibile overfitting")
elif overfitting_gap < 0:
    print("[WARNING] ATTENZIONE: Test score > Train score - insolito, verificare dati")
else:
    print("[OK] Gap accettabile - modello generalizza bene")

# Top 5 configurazioni
print("\n--- TOP 5 CONFIGURAZIONI PER PRECISIONE ---")
top_5 = results_df.nlargest(5, 'mean_test_score')[['param_rf__n_estimators', 'param_rf__max_depth', 'param_rf__min_samples_leaf', 'mean_test_score', 'std_test_score']]
print(top_5.to_string(index=False))

# Visualizzazioni rimosse: mantenuti solo ROC, Precision-Recall, Confusion Matrix e Top 10 feature (Random Forest)
# (Istogrammi e grafici di confronto iperparametri eliminati perché non richiesti)
# pass

# Analisi di stabilità
print("\n--- ANALISI DI STABILITÀ DEL MODELLO ---")
print(f"Variabilità tra i fold del miglior modello:")
print(f"  Coefficiente di variazione: {cv_coefficient:.2f}%")
if cv_coefficient < 5:
    print("  OK Stabilità ECCELLENTE (CV < 5%)")
elif cv_coefficient < 10:
    print("  OK Stabilità BUONA (5% <= CV < 10%)")
elif cv_coefficient < 15:
    print("  [WARN] Stabilità MODERATA (10% <= CV < 15%)")
else:
    print("  [WARN] Stabilità BASSA (CV >= 15%) - Considerare più dati o feature engineering")

# Summary finale
print("\n" + "="*70)
print("SUMMARY FINALE")
print("="*70)
print(f"Miglior modello: Random Forest con params={grid_search.best_params_}")
print(f"Performance in CV (Precision): {grid_search.best_score_:.4f} ± {np.std(best_cv_scores):.4f}")
print(f"Precision sul test set: {precision_test:.4f}")
print(f"Recall sul test set: {recall_test:.4f}")
print(f"F1-score sul test set: {f1_test:.4f}")
print(f"Accuracy sul test set: {accuracy_test:.4f}")
print(f"Stabilità del modello: CV={cv_coefficient:.2f}%")
print("="*70)

# Curva ROC
probs = best_pipeline.predict_proba(X_test)
probs = probs[:, 1]

auc = roc_auc_score(y_test, probs)
print(f'\nAUC: {auc:.3f}')

fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], linestyle='--', label='Random')
plt.plot(fpr, tpr, marker='.', label=f'Random Forest (AUC = {auc:.3f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(_with_dataset_suffix('rf_roc_curve.png'), dpi=300)


# Curva Precision-Recall
average_precision = average_precision_score(y_test, probs)
precision, recall, _ = precision_recall_curve(y_test, probs)

step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})

plt.figure(figsize=(8, 6))
plt.step(recall, precision, color='b', alpha=0.2, where='post', label='Precision-Recall curve')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(f'2-class Precision-Recall curve: AP={average_precision:.2f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(_with_dataset_suffix('rf_precision_recall_curve.png'), dpi=300)


print(f'\nAverage Precision: {average_precision:.4f}')

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_pipeline.named_steps['rf'].feature_importances_
}).sort_values('importance', ascending=False)

print('\nTop 10 feature più importanti:')
print(feature_importance.head(10))

# Grafico feature importance
plt.figure(figsize=(10, 6))
top_features = feature_importance.head(10)
plt.barh(range(len(top_features)), top_features['importance'], align='center')
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances - Random Forest')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(_with_dataset_suffix('rf_feature_importance.png'), dpi=300)

# =============================================================================
# SALVATAGGIO RISULTATI ESPERIMENTO (CSV)
# =============================================================================
EXPERIMENT_CSV = os.environ.get(
    "RF_EXPERIMENT_CSV",
    _with_dataset_suffix("rf_experiment_results.csv")
)

results_df = results_df.copy()
experiment_meta = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "dataset_path": DATASET_PATH,
    "dataset_tag": DATASET_TAG,
    "dataset_format": DATASET_FORMAT,
    "n_samples": int(len(y)),
    "n_features": int(X.shape[1]),
    "train_samples": int(len(y_train)),
    "test_samples": int(len(y_test)),
    "best_params": repr(grid_search.best_params_),
    "best_cv_precision": float(grid_search.best_score_),
    "test_accuracy": float(accuracy_test),
    "test_precision": float(precision_test),
    "test_recall": float(recall_test),
    "test_f1": float(f1_test),
    "test_auc": float(auc),
    "test_average_precision": float(average_precision),
    "cv_precision_mean": float(np.mean(cv_precision_scores)),
    "cv_precision_std": float(np.std(cv_precision_scores)),
    "cv_recall_mean": float(np.mean(cv_recall_scores)),
    "cv_recall_std": float(np.std(cv_recall_scores)),
    "best_cv_precision_mean": float(np.mean(best_cv_scores)),
    "best_cv_precision_std": float(np.std(best_cv_scores)),
    "best_train_cv_precision": float(best_train_score),
    "best_test_cv_precision": float(best_test_score),
    "overfitting_gap": float(overfitting_gap),
}

for key, value in experiment_meta.items():
    results_df[key] = value

out_dir = os.path.dirname(EXPERIMENT_CSV)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

results_df.to_csv(EXPERIMENT_CSV, index=False)
print(f"\nRisultati esperimento RandomForest salvati in: {EXPERIMENT_CSV}")

# =============================================================================
# SALVATAGGIO RIEPILOGO ESPERIMENTI (CSV)
# =============================================================================
SUMMARY_CSV = os.environ.get("EXPERIMENTS_SUMMARY_CSV", "experiments_summary.csv")

summary_row = {"model": "RandomForest", **experiment_meta}
summary_df = pd.DataFrame([summary_row])

summary_dir = os.path.dirname(SUMMARY_CSV)
if summary_dir:
    os.makedirs(summary_dir, exist_ok=True)

write_header = not os.path.exists(SUMMARY_CSV)
summary_df.to_csv(SUMMARY_CSV, mode="a", header=write_header, index=False)
print(f"Riepilogo esperimenti aggiornato in: {SUMMARY_CSV}")

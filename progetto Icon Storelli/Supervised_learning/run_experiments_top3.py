"""
Script per: 
- trovare le top-3 feature da RandomForest
- addestrare KNN, SVM, RandomForest sul dataset completo e usando solo le top-3 feature
- valutare i modelli sul test set
- registrare i tempi di addestramento e i risultati in un CSV e in output console
"""

import time
import warnings
import numpy as np
import pandas as pd
import random
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.exceptions import UndefinedMetricWarning
import matplotlib.pyplot as plt
import seaborn as sns

# Sopprime ripetuti UndefinedMetricWarning (nasconde completamente i messaggi)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Disattiva tutte le stampe a terminale (output sarà prodotto come figura/CSV)
def _noop(*a, **k):
    pass
print = _noop

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ---------- Caricamento e preprocessing (coerente con gli script esistenti) ----------
feature = ["A1_Score","A2_Score","A3_Score","A4_Score","A5_Score","A6_Score","A7_Score","A8_Score","A9_Score","A10_Score","age","gender","ethnicity","jundice","is_autistic","screening_score","PDD_parent","Class/ASD"]
feature_dummied = ["A1_Score","A2_Score","A3_Score","A4_Score","A5_Score","A6_Score","A7_Score","A8_Score","A9_Score","A10_Score","age","gender","ethnicity","jundice","is_autistic","screening_score","PDD_parent"]

# Legge il CSV usando l'header presente (evita problemi di shift di colonne)
df = pd.read_csv("Ontologia/Autism-Dataset.csv", sep=",", dtype=str)
# seleziona solo le colonne attese se esistono
expected_cols = ["A1_Score","A2_Score","A3_Score","A4_Score","A5_Score","A6_Score","A7_Score","A8_Score","A9_Score","A10_Score","age","gender","ethnicity","jundice","is_autistic","screening_score","PDD_parent","Class/ASD"]
available = [c for c in expected_cols if c in df.columns]
missing = set(expected_cols) - set(available)
if missing:
    print(f"[WARNING] Mancano colonne attese: {missing}. Procedo con quelle disponibili.")
    
df = df[available].copy()

# rimuovi eventuali righe header o NA nel target
df = df[df['Class/ASD'].notna() & (df['Class/ASD'].str.strip().str.lower() != 'class/asd')].reset_index(drop=True)

# --- Pulizia specifica: is_autistic ---
print("\n[DATA CLEANUP] Valori originali di 'is_autistic':")
print(df['is_autistic'].value_counts(dropna=False).to_string())
# Normalizza e converti in numerico
df['is_autistic'] = df['is_autistic'].str.strip()
df['is_autistic_numeric'] = pd.to_numeric(df['is_autistic'], errors='coerce')
# Rimuove valori non 0/1
invalid_mask = ~df['is_autistic_numeric'].isin([0,1])
if invalid_mask.any():
    print(f"[DATA CLEANUP] Trovati {invalid_mask.sum()} valori non validi in 'is_autistic' (non 0/1). Rimuovo quelle righe.")
    df = df[~invalid_mask].reset_index(drop=True)
# Sostituisci con stringhe '0'/'1' per mantenere coerenza con get_dummies
df['is_autistic'] = df['is_autistic_numeric'].astype(int).astype(str)
df.drop(columns=['is_autistic_numeric'], inplace=True)
print("[DATA CLEANUP] Valori di 'is_autistic' dopo pulizia:")
print(df['is_autistic'].value_counts().to_string())

# prepara dummies
feature_dummied = [c for c in feature_dummied if c in df.columns]
data_dummies = pd.get_dummies(df, columns=feature_dummied)
data_dummies = data_dummies.drop(["Class/ASD"], axis=1)
# converti in numerico riempiendo i NaN con la moda
X = data_dummies.apply(pd.to_numeric, errors='coerce')
for col in X.columns:
    mode_val = X[col].mode()
    X[col] = X[col].fillna(mode_val.iloc[0] if len(mode_val) > 0 else 0)
X = X.astype(float)
# target binario
y = df['Class/ASD'].map({'1': 1, '0': 0})
mask = y.notna()
if not mask.all():
    df = df[mask].reset_index(drop=True)
    X = X.loc[mask].reset_index(drop=True)
    y = y[mask].astype(int).reset_index(drop=True)
else:
    y = y.astype(int)

# Divisione train/test (consistente con RF e KNN usati prima)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=RANDOM_SEED, stratify=y)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# cv
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# helper per allenare una GridSearch e misurare il tempo
def run_grid(pipeline, param_grid, X_tr, y_tr, scoring='precision'):
    gs = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=0, return_train_score=True)
    start = time.perf_counter()
    gs.fit(X_tr, y_tr)
    elapsed = time.perf_counter() - start
    return gs, elapsed

results = []

# ---------- 1) RandomForest su tutte le feature per ottenere feature importances ----------
print('\n=== 1) RandomForest (full features) per feature importances ===')
rf_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=RANDOM_SEED)),
    ('rf', RandomForestClassifier(random_state=RANDOM_SEED))
])
rf_param_grid = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [16, 18, 20, None],
    'rf__min_samples_leaf': [1, 2, 4]
}
rf_gs, rf_time = run_grid(rf_pipeline, rf_param_grid, X_train, y_train)
print(f"RF best params: {rf_gs.best_params_}  (train time: {rf_time:.2f}s)")
# estrai feature importances dal miglior stimatore
best_rf = rf_gs.best_estimator_.named_steps['rf']
importances = best_rf.feature_importances_
feat_importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
feat_importance_df = feat_importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
print('\nTop 10 feature importance:')
print(feat_importance_df.head(10).to_string(index=False))

top3 = feat_importance_df.head(3)['feature'].tolist()
print(f"\nTop-3 features: {top3}")

# Nota: non aggiungiamo ora la voce RF full ai risultati per evitare duplicati
# (RF verrà eseguito e registrato nella fase generale dei modelli).
pred = rf_gs.best_estimator_.predict(X_test)
print(f"RF (full) best params: {rf_gs.best_params_}  (train time: {rf_time:.2f}s)")
print('\nTop 10 feature importance:')
print(feat_importance_df.head(10).to_string(index=False))
print(f"\nTop-3 features: {top3}")

# ---------- Modelli e param grids (ripresi dagli script esistenti) ----------
# Ordine: KNN, RandomForest, SVM (come richiesto)
models = {
    'KNN': {
        'pipeline': ImbPipeline([('smote', SMOTE(random_state=RANDOM_SEED)), ('knn', KNeighborsClassifier())]),
        'param_grid': {
            'knn__n_neighbors': list(range(1, 21)),
            'knn__weights': ['uniform', 'distance'],
            'knn__metric': ['euclidean', 'manhattan', 'minkowski']
        }
    },
    'RandomForest': {
        'pipeline': ImbPipeline([('smote', SMOTE(random_state=RANDOM_SEED)), ('rf', RandomForestClassifier(random_state=RANDOM_SEED))]),
        'param_grid': rf_param_grid
    },
    'SVM': {
        'pipeline': ImbPipeline([('smote', SMOTE(random_state=RANDOM_SEED)), ('svm', svm.SVC(probability=True, random_state=RANDOM_SEED))]),
        'param_grid': {
            'svm__C': [0.1, 1, 10, 100],
            'svm__gamma': [1e-4, 1e-3, 0.01],
            'svm__kernel': ['rbf', 'linear']
        }
    }
}

# ---------- 2) Per ogni modello: GridSearch su tutte le feature e valutazione ----------
for mname, mconf in models.items():
    print(f"\n=== 2) {mname} - training su tutte le feature ===")
    gs, t = run_grid(mconf['pipeline'], mconf['param_grid'], X_train, y_train)
    pred = gs.best_estimator_.predict(X_test)
    results.append({
        'model': mname,
        'features': 'all',
        'train_time_s': round(t, 3),
        'precision_test': precision_score(y_test, pred, zero_division=0),
        'recall_test': recall_score(y_test, pred, zero_division=0),
        'f1_test': f1_score(y_test, pred, zero_division=0),
        'accuracy_test': accuracy_score(y_test, pred),
        'pred_pos': int(pred.sum()),
        'y_test_pos': int(y_test.sum()),
        'best_params': gs.best_params_
    })
    print(f"{mname} (all) best params: {gs.best_params_} - time: {t:.2f}s - precision_test: {precision_score(y_test, pred, zero_division=0):.4f} (pred_pos={int(pred.sum())})")

# ---------- 3) Ripeti usando solo le top-3 feature ----------
print('\n=== 3) Ripetizione esperimenti usando solo le top-3 feature ===')
X_train_top3 = X_train[top3].copy()
X_test_top3 = X_test[top3].copy()

for mname, mconf in models.items():
    print(f"\n=== {mname} - training su top-3 feature ===")
    try:
        gs, t = run_grid(mconf['pipeline'], mconf['param_grid'], X_train_top3, y_train)
        pred = gs.best_estimator_.predict(X_test_top3)
        results.append({
            'model': mname,
            'features': 'top3',
            'train_time_s': round(t, 3),
            'precision_test': precision_score(y_test, pred, zero_division=0),
            'recall_test': recall_score(y_test, pred, zero_division=0),
            'f1_test': f1_score(y_test, pred, zero_division=0),
            'accuracy_test': accuracy_score(y_test, pred),
            'pred_pos': int(pred.sum()),
            'y_test_pos': int(y_test.sum()),
            'best_params': gs.best_params_
        })
        print(f"{mname} (top3) best params: {gs.best_params_} - time: {t:.2f}s - precision_test: {precision_score(y_test, pred, zero_division=0):.4f} (pred_pos={int(pred.sum())})")
    except Exception as e:
        print(f"[ERROR] {mname} on top3 failed: {e}")

# ---------- Salva e mostra i risultati ----------
res_df = pd.DataFrame(results)
# Rimuove eventuali duplicati per modello/features (precauzione)
res_df = res_df.drop_duplicates(subset=['model','features'], keep='first').sort_values(['model','features']).reset_index(drop=True)

# Salva risultati grezzi (CSV)
res_df.to_csv('Supervised_learning/experiments_top3_results.csv', index=False)
# Salva top features in CSV per consultazione
feat_importance_df.head(10).to_csv('Supervised_learning/top_features.csv', index=False)

# --- Figura: confronto modelli (all vs top3) per precision / recall / accuracy + train time ---
metrics = ['precision_test', 'recall_test', 'f1_test', 'accuracy_test', 'train_time_s']
fig, axes = plt.subplots(1, len(metrics), figsize=(26, 5), constrained_layout=True)
for ax, metric in zip(axes, metrics):
    sns.barplot(data=res_df, x='model', y=metric, hue='features', ax=ax)
    ax.set_xlabel('Model')

    # titolo e asse y diversi per il tempo di addestramento
    if metric == 'train_time_s':
        ax.set_title('Train Time (s)')
        ax.set_ylabel('Seconds')
        ymax = res_df['train_time_s'].max() * 1.15 if len(res_df) > 0 else None
        if np.isfinite(ymax):
            ax.set_ylim(0, ymax)
    else:
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)

    # annotate bars (mostra 's' per i tempi)
    for p in ax.patches:
        h = p.get_height()
        if np.isfinite(h):
            if metric == 'train_time_s':
                ax.annotate(f"{h:.3f}s", (p.get_x() + p.get_width() / 2., h),
                            ha='center', va='bottom', fontsize=9, color='black', xytext=(0, 3), textcoords='offset points')
            else:
                ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h),
                            ha='center', va='bottom', fontsize=9, color='black', xytext=(0, 3), textcoords='offset points')

# Legend + title
axes[-1].legend(title='Features', loc='best')
fig.suptitle('Confronto modelli: tutti i feature vs top-3 feature (Precision, Recall, F1-Score, Accuracy, Train Time)', fontsize=14)
# Mostra la figura a video (non salvare su file)
plt.show()
plt.close(fig)

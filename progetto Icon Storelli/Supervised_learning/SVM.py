"""
Questo script analizza un dataset sull'autismo utilizzando il classificatore SVM.
Risolti i problemi di data leakage usando Pipeline e automatizzata la ricerca degli iperparametri con GridSearchCV.
"""

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import random

# Configurazione seed per riproducibilità
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from inspect import signature

# Definizione delle feature e delle feature dummificate
feature = ["A1_Score","A2_Score","A3_Score","A4_Score","A5_Score","A6_Score","A7_Score","A8_Score","A9_Score","A10_Score","age","gender","ethnicity","jundice","is_autistic","screening_score","PDD_parent","Class/ASD"]
feature_dummied = ["A1_Score","A2_Score","A3_Score","A4_Score","A5_Score","A6_Score","A7_Score","A8_Score","A9_Score","A10_Score","age","gender","ethnicity","jundice","is_autistic","screening_score","PDD_parent"]

# Caricamento del dataset da un file CSV
dataset = pd.read_csv("Ontologia/Autism-Dataset.csv", sep=",", names=feature, 
                      dtype={'A1_Score':object,'A2_Score':object,'A3_Score':object,'A4_Score':object,'A5_Score':object,'A6_Score':object,'A7_Score':object,'A8_Score':object,'A9_Score':object,'A10_Score':object,'age':object,'gender':object,'ethnicity':object,'jundice':object,'is_autistic':object,'screening_score':object,'PDD_parent':object,'Class/ASD':object})

# Rimuove eventuali righe dove la colonna target contiene il nome della colonna (es. header importato come riga)
# oppure valori non validi (NA). Questo evita errori di conversione in seguito.
dataset = dataset[dataset['Class/ASD'].notna() & (dataset['Class/ASD'].astype(str).str.strip().str.lower() != 'class/asd')].reset_index(drop=True) 

# Dummificazione delle feature categoriche
data_dummies = pd.get_dummies(dataset, columns=feature_dummied)
data_dummies = data_dummies.drop(["Class/ASD"], axis=1)

# Assicura tipi numerici (SMOTE richiede variabili numeriche, non boolean/object)
# Convertiamo tutte le colonne in numerico, sostituendo NA con la moda
data_dummies = data_dummies.apply(pd.to_numeric, errors='coerce')
for col in data_dummies.columns:
    mode_val = data_dummies[col].mode()
    data_dummies[col] = data_dummies[col].fillna(mode_val.iloc[0] if len(mode_val) > 0 else 0)
data_dummies = data_dummies.astype(float)

# Preparazione dei dati di input (X) e target (y)
X = data_dummies
# Convertiamo la colonna target in 0/1 in modo robusto evitando get_dummies su Series
# (gestisce anche eventuali valori stringa '0'/'1')
y = dataset['Class/ASD'].map({'1': 1, '0': 0})
mask = y.notna()
if not mask.all():
    dropped = (~mask).sum()
    print(f"[WARN] Rimosse {dropped} righe con target non valido")
    dataset = dataset[mask].reset_index(drop=True)
    X = X.loc[mask].reset_index(drop=True)
    y = y[mask].astype(int).reset_index(drop=True)
else:
    y = y.astype(int) 

# Divisione dei dati in training e test set (PRIMA di applicare SMOTE!)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=RANDOM_SEED, stratify=y)

print(f"Training set: {X_train.shape[0]} campioni")
print(f"Test set: {X_test.shape[0]} campioni")
# Assicuriamoci che y_train sia intero per np.bincount
print(f"Distribuzione classi nel training set: {np.bincount(y_train.astype(int))}")

# Creazione della Pipeline con SMOTE e SVM
# Questo evita il data leakage: SMOTE viene applicato solo durante la CV sui fold di training
pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=RANDOM_SEED)),
    ('svm', svm.SVC(probability=True, random_state=RANDOM_SEED))
])

# Definizione della griglia di iperparametri da testare
param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': [1e-4, 1e-3, 0.01],
    'svm__kernel': ['rbf', 'linear']
}

# Ricerca degli iperparametri con GridSearchCV e cross-validation stratificata
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=cv, 
    scoring='precision',
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

print("\nAvvio ricerca iperparametri con GridSearchCV...")
print("Questo potrebbe richiedere alcuni minuti...")
grid_search.fit(X_train, y_train)

# Stampa dei risultati della ricerca
print(f"\nMigliori iperparametri trovati:")
print(f"  C: {grid_search.best_params_['svm__C']}")
print(f"  gamma: {grid_search.best_params_['svm__gamma']}")
print(f"  kernel: {grid_search.best_params_['svm__kernel']}")
print(f"Miglior precisione in CV: {grid_search.best_score_:.4f}")

# =============================================================================
# VALUTAZIONE SUL TEST SET E ANALISI STATISTICA
# =============================================================================
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

plt.figure(figsize=(10, 7))
df_cm = pd.DataFrame(cm, index=[i for i in "01"], columns=[i for i in "01"])
sn.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - Best Model (GridSearchCV)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

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
print("\n--- ANALISI PER C ---")
c_analysis = results_df.groupby('param_svm__C')['mean_test_score'].agg(['mean', 'std', 'min', 'max'])
print(c_analysis)

print("\n--- ANALISI PER GAMMA ---")
gamma_analysis = results_df.groupby('param_svm__gamma')['mean_test_score'].agg(['mean', 'std', 'min', 'max'])
print(gamma_analysis)

print("\n--- ANALISI PER KERNEL ---")
kernel_analysis = results_df.groupby('param_svm__kernel')['mean_test_score'].agg(['mean', 'std', 'min', 'max'])
print(kernel_analysis)

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
top_5 = results_df.nlargest(5, 'mean_test_score')[['param_svm__C', 'param_svm__gamma', 'param_svm__kernel', 'mean_test_score', 'std_test_score']]
print(top_5.to_string(index=False))


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
print(f"Miglior modello: SVM con params={grid_search.best_params_}")
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

from sklearn.metrics import roc_curve, roc_auc_score
# removed unused pyplot import

auc = roc_auc_score(y_test, probs)
print(f'\nAUC: {auc:.3f}')

fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], linestyle='--', label='Random')
plt.plot(fpr, tpr, marker='.', label=f'SVM (AUC = {auc:.3f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Curva Precision-Recall
# Usiamo le probabilità (probs) invece delle etichette binarie per curve più informative
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
plt.show()

print(f'\nAverage Precision: {average_precision:.4f}')



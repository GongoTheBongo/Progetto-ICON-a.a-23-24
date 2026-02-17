"""
Questo script analizza un dataset sull'autismo utilizzando il classificatore dei k-nn.
Risolti i problemi di data leakage usando Pipeline e utilizzato KneeLocator per identificare il gomito della curva.

CORREZIONI APPORTATE:
- Risolto problema di applicazione SMOTE
- Corretta estrazione parametri da GridSearchCV
- Aggiunta sezione di analisi statistica completa
- Corretta cross-validation con pipeline completa
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, roc_auc_score
from inspect import signature
from kneed import KneeLocator

# Definizione delle feature e delle feature dummificate
feature = ["A1_Score","A2_Score","A3_Score","A4_Score","A5_Score","A6_Score","A7_Score","A8_Score","A9_Score","A10_Score","age","gender","ethnicity","jundice","is_autistic","screening_score","PDD_parent","Class/ASD"]
feature_dummied = ["A1_Score","A2_Score","A3_Score","A4_Score","A5_Score","A6_Score","A7_Score","A8_Score","A9_Score","A10_Score","age","gender","ethnicity","jundice","is_autistic","screening_score","PDD_parent"]

# Caricamento del dataset da un file CSV
dataset = pd.read_csv("Ontologia/Autism-Dataset.csv", sep=",", names=feature, 
                      dtype={'A1_Score':object,'A2_Score':object,'A3_Score':object,'A4_Score':object,'A5_Score':object,'A6_Score':object,'A7_Score':object,'A8_Score':object,'A9_Score':object,'A10_Score':object,'age':object,'gender':object,'ethnicity':object,'jundice':object,'is_autistic':object,'screening_score':object,'PDD_parent':object,'Class/ASD':object})

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
y = pd.get_dummies(dataset["Class/ASD"], columns=["Class/ASD"])
y = y["1"]

# Divisione dei dati in training e test set (PRIMA di applicare SMOTE!)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=RANDOM_SEED, stratify=y)

print(f"Training set: {X_train.shape[0]} campioni")
print(f"Test set: {X_test.shape[0]} campioni")
print(f"Distribuzione classi nel training set: {np.bincount(y_train)}")

# =============================================================================
# RICERCA COMPLETA CON GRIDSEARCHCV
# =============================================================================
print("\n" + "="*70)
print("RICERCA IPERPARAMETRI COMPLETA CON GRIDSEARCHCV")
print("="*70)

# Creazione della Pipeline con SMOTE e KNN
pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=RANDOM_SEED)),
    ('knn', KNeighborsClassifier())
])

# Definizione della griglia di iperparametri da testare
param_grid = {
    'knn__n_neighbors': list(range(1, 21)),
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan', 'minkowski']
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
    return_train_score=True  # Per analisi statistica
)

print("\nAvvio ricerca iperparametri con GridSearchCV...")
grid_search.fit(X_train, y_train)

# Stampa dei risultati della ricerca
print(f"\nMigliori iperparametri trovati con GridSearchCV:")
print(f"  n_neighbors: {grid_search.best_params_['knn__n_neighbors']}")
print(f"  weights: {grid_search.best_params_['knn__weights']}")
print(f"  metric: {grid_search.best_params_['knn__metric']}")
print(f"Miglior precisione in CV: {grid_search.best_score_:.4f}")


# =============================================================================
# VALUTAZIONE SUL TEST SET CON IL MIGLIOR MODELLO
# =============================================================================
print("\n" + "="*70)
print("VALUTAZIONE SUL TEST SET")
print("="*70)

# Usa direttamente grid_search.best_estimator_ (che è la pipeline completa)
best_pipeline = grid_search.best_estimator_

# Predizione sul test set
pred_grid = best_pipeline.predict(X_test)
precision_grid = precision_score(y_test, pred_grid)
recall_grid = recall_score(y_test, pred_grid)
f1_grid = f1_score(y_test, pred_grid)
accuracy_grid = accuracy_score(y_test, pred_grid)

print(f"\nMigliori parametri: {grid_search.best_params_}")
print(f"  Accuracy sul test set: {accuracy_grid:.4f}")
print(f"  Precision sul test set: {precision_grid:.4f}")
print(f"  Recall sul test set: {recall_grid:.4f}")
print(f"  F1-score sul test set: {f1_grid:.4f}")

# Matrice di confusione
cm = confusion_matrix(y_test, pred_grid)
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
plt.show()

# =============================================================================
# CROSS-VALIDATION SUL TRAINING SET CON IL MIGLIOR MODELLO
# =============================================================================
print("\n" + "="*70)
print("CROSS-VALIDATION SUL TRAINING SET")
print("="*70)

# Usa la pipeline completa per la cross-validation
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


# =============================================================================
# ANALISI STATISTICA APPROFONDITA
# =============================================================================
print("\n" + "="*70)
print("ANALISI STATISTICA COMPLETA")
print("="*70)

# Estrazione dei risultati della GridSearch
results_df = pd.DataFrame(grid_search.cv_results_)

# 1. ANALISI GLOBALE DI TUTTI I MODELLI
print("\n--- STATISTICHE GLOBALI SU TUTTI I MODELLI TESTATI ---")
print(f"Numero totale di configurazioni testate: {len(results_df)}")
print(f"\nPrecision in Cross-Validation:")
print(f"  Media globale: {results_df['mean_test_score'].mean():.4f}")
print(f"  Deviazione standard globale: {results_df['mean_test_score'].std():.4f}")
print(f"  Minimo: {results_df['mean_test_score'].min():.4f}")
print(f"  Massimo: {results_df['mean_test_score'].max():.4f}")
print(f"  Mediana: {results_df['mean_test_score'].median():.4f}")
print(f"  25° percentile: {results_df['mean_test_score'].quantile(0.25):.4f}")
print(f"  75° percentile: {results_df['mean_test_score'].quantile(0.75):.4f}")

# 2. ANALISI PER IPERPARAMETRO
print("\n--- ANALISI PER VALORE DI K (n_neighbors) ---")
k_analysis = results_df.groupby('param_knn__n_neighbors')['mean_test_score'].agg(['mean', 'std', 'min', 'max'])
print(k_analysis)

print("\n--- ANALISI PER TIPO DI PESO (weights) ---")
weights_analysis = results_df.groupby('param_knn__weights')['mean_test_score'].agg(['mean', 'std', 'min', 'max'])
print(weights_analysis)

print("\n--- ANALISI PER METRICA DI DISTANZA (metric) ---")
metric_analysis = results_df.groupby('param_knn__metric')['mean_test_score'].agg(['mean', 'std', 'min', 'max'])
print(metric_analysis)

# 3. ANALISI DEL MIGLIOR MODELLO
print("\n--- ANALISI DETTAGLIATA DEL MIGLIOR MODELLO ---")
best_idx = grid_search.best_index_
best_cv_scores = [results_df.loc[best_idx, f'split{i}_test_score'] for i in range(5)]

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

# Coefficiente di variazione (misura di stabilità relativa)
cv_coefficient = (np.std(best_cv_scores) / np.mean(best_cv_scores)) * 100
print(f"  Coefficiente di variazione: {cv_coefficient:.2f}%")

# 4. CONFRONTO TRAIN vs TEST
print("\n--- CONFRONTO TRAIN vs TEST (Analisi Overfitting) ---")
best_train_score = results_df.loc[best_idx, 'mean_train_score']
best_test_score = results_df.loc[best_idx, 'mean_test_score']
overfitting_gap = best_train_score - best_test_score

print(f"Precision medio sul training set (CV): {best_train_score:.4f}")
print(f"Precision medio sul test set (CV): {best_test_score:.4f}")
print(f"Gap (potenziale overfitting): {overfitting_gap:.4f}")

if overfitting_gap > 0.05:
    print("ATTENZIONE: Gap significativo - possibile overfitting")
elif overfitting_gap < 0:
    print("ATTENZIONE: Test score > Train score - insolito, verificare dati")
else:
    print("OK - Gap accettabile: il modello generalizza bene")

# 5. TOP 5 CONFIGURAZIONI
print("\n--- TOP 5 CONFIGURAZIONI PER PRECISIONE ---")
top_5 = results_df.nlargest(5, 'mean_test_score')[['param_knn__n_neighbors', 'param_knn__weights', 
                                                     'param_knn__metric', 'mean_test_score', 'std_test_score']]
print(top_5.to_string(index=False))


# 7. ANALISI DI STABILITÀ
print("\n--- ANALISI DI STABILITÀ DEL MODELLO ---")
print("Variabilità tra i fold del miglior modello:")
print(f"  Coefficiente di variazione: {cv_coefficient:.2f}%")
if cv_coefficient < 5:
    print("  OK Stabilità ECCELLENTE (CV < 5%)")
elif cv_coefficient < 10:
    print("  OK Stabilità BUONA (5% <= CV < 10%)")
elif cv_coefficient < 15:
    print("  ATTENZIONE Stabilità MODERATA (10% <= CV < 15%)")
else:
    print("  ATTENZIONE Stabilità BASSA (CV >= 15%) - Considerare più dati o feature engineering")

# 8. SUMMARY FINALE
print("\n" + "="*70)
print("SUMMARY FINALE")
print("="*70)
print(f"Miglior modello: KNN con k={grid_search.best_params_['knn__n_neighbors']}, "
      f"weights={grid_search.best_params_['knn__weights']}, "
      f"metric={grid_search.best_params_['knn__metric']}")
print(f"Performance in CV (Precision): {grid_search.best_score_:.4f} ± {np.std(best_cv_scores):.4f}")
print(f"Precision sul test set: {precision_grid:.4f}")
print(f"Recall sul test set: {recall_grid:.4f}")
print(f"F1-score sul test set: {f1_grid:.4f}")
print(f"Accuracy sul test set: {accuracy_grid:.4f}")
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
plt.plot(fpr, tpr, marker='.', label=f'KNN (AUC = {auc:.3f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

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
plt.show()

print(f'\nAverage Precision: {average_precision:.4f}')
"""
Questo script esegue il clustering K-Means sul dataset dell'autismo.
Utilizza KneeLocator per identificare automaticamente il gomito della curva WCSS
e seleziona il miglior modello tra quelli analizzati.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# Configurazione seed per riproducibilità
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from kneed import KneeLocator

# Caricamento del dataset
dataset0 = pd.read_csv('Ontologia/Autism-Dataset.csv')
dataset = dataset0.copy()

# Applicazione del one-hot encoding sulle feature di tipo stringa
categorical_features = ["age", "ethnicity", "screening_score", "contry_of_res", "test_compiler"]

onehot_encoder = OneHotEncoder(sparse_output=False)
encoded_dataset = onehot_encoder.fit_transform(dataset[categorical_features])

# Combinazione delle feature encodate con il resto del dataset
encoded_dataset = pd.concat([dataset.drop(columns=categorical_features), pd.DataFrame(encoded_dataset)], axis=1)

# Conversione in array numpy
dataset_array = np.array(encoded_dataset)

# =============================================================================
# RICERCA DEL K OTTIMALE CON KNEELOCATOR (METODO DEL GOMITO)
# =============================================================================
print("=" * 70)
print("RICERCA DEL K OTTIMALE CON KNEELOCATOR")
print("=" * 70)

# Esecuzione K-means per un range di K cluster
k_range = range(1, 21)
wcss = []

print("Calcolo WCSS per K da 1 a 20...")

for k in k_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED)
    kmeans.fit(dataset_array)
    wcss.append(kmeans.inertia_)

# Uso KneeLocator per trovare il gomito sulla curva WCSS
# curve='convex' e direction='decreasing' per la curva WCSS tipica
knee_locator = KneeLocator(list(k_range), wcss, curve='convex', direction='decreasing')

# Identificazione del gomito
if knee_locator.knee:
    optimal_k_elbow = int(knee_locator.knee)
    knee_found = True
else:
    # Se non trova un gomito, usa K=3 come default
    optimal_k_elbow = 3
    knee_found = False

print(f"\nRisultati KneeLocator:")
if knee_found:
    print(f"  Gomito identificato: K = {optimal_k_elbow}")
    print(f"  K selezionato: {optimal_k_elbow}")
else:
    print(f"  Gomito NON identificato automaticamente")
    print(f"  K selezionato (fallback su max Silhouette): {optimal_k_elbow}")


# =============================================================================
# SELEZIONE DEL MIGLIOR MODELLO
# =============================================================================
print("\n" + "=" * 70)
print("SELEZIONE DEL MIGLIOR MODELLO")
print("=" * 70)

# Selezione basata solo sul gomito della curva WCSS
best_k = optimal_k_elbow
selection_reason = "KneeLocator (Gomito della curva WCSS)"

print(f"\nMiglior modello selezionato: K = {best_k}")
print(f"Metodo di selezione: {selection_reason}")
print(f"WCSS per K={best_k}: {wcss[best_k-1]:.2f}")

# Calcolo WCSS per alcuni valori di K per confronto
print("\nConfronto WCSS per diversi valori di K:")
for k in [2, 3, 4, best_k]:
    if k <= len(wcss):
        marker = " <-- K selezionato" if k == best_k else ""
        print(f"  K={k}: WCSS = {wcss[k-1]:.2f}{marker}")

# Addestramento del modello finale
kmeans_final = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_SEED)
kmeans_final.fit(dataset_array)
cluster_result = kmeans_final.labels_

# Aggiunta dei cluster al dataset originale
dataset0['cluster'] = cluster_result

# Riordina le colonne del DataFrame
columns_order = list(dataset0.columns)
columns_order.remove('Class/ASD')
columns_order.append('Class/ASD')
columns_order.remove('cluster')
columns_order.insert(-1, 'cluster')
dataset_reordered = dataset0[columns_order]

# Salva il dataset con i cluster
output_path = 'Clustering/Autism-Dataset+clusters.csv'
dataset_reordered.to_csv(output_path, index=False)
print(f"\nDataset con cluster salvato in: {output_path}")

# =============================================================================
# ANALISI DEI CLUSTER
# =============================================================================
print("\n" + "=" * 70)
print("ANALISI DEI CLUSTER")
print("=" * 70)

# Distribuzione dei cluster
cluster_counts = pd.Series(cluster_result).value_counts().sort_index()
print(f"\nDistribuzione dei cluster:")
for i, count in enumerate(cluster_counts):
    print(f"  Cluster {i}: {count} campioni ({count/len(cluster_result)*100:.1f}%)")

# Se esiste la colonna 'Class/ASD', analizza la corrispondenza con i cluster
if 'Class/ASD' in dataset0.columns:
    print(f"\nCorrispondenza tra cluster e Class/ASD:")
    crosstab = pd.crosstab(dataset0['cluster'], dataset0['Class/ASD'], margins=True)
    print(crosstab)
    
    # Calcolo purezza dei cluster
    print(f"\nPurezza dei cluster (percentuale classe maggioritaria):")
    for cluster_id in range(best_k):
        cluster_data = dataset0[dataset0['cluster'] == cluster_id]
        if len(cluster_data) > 0:
            class_counts = cluster_data['Class/ASD'].value_counts()
            purity = class_counts.iloc[0] / len(cluster_data) * 100
            print(f"  Cluster {cluster_id}: {purity:.1f}% ({class_counts.index[0]})")

# Visualizzazione finale
plt.figure(figsize=(12, 5))

# Plot 1: WCSS completa con miglior K evidenziato
plt.subplot(1, 2, 1)
plt.plot(k_range, wcss, 'bx-', label='WCSS')
plt.axvline(x=best_k, color='r', linestyle='--', linewidth=2, label=f'Miglior K = {best_k}')
plt.plot(best_k, wcss[best_k-1], 'ro', markersize=15, markerfacecolor='red')
plt.title('WCSS - Modello Selezionato')
plt.xlabel('Numero di cluster (K)')
plt.ylabel('WCSS')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("RIEPILOGO FINALE")
print("=" * 70)
print(f"Numero ottimale di cluster: {best_k}")
print(f"Metodo di selezione: {selection_reason}")
print(f"WCSS finale: {kmeans_final.inertia_:.2f}")
print("=" * 70)

# =============================================================================
# FUNZIONE PER OTTENERE IL MIGLIOR MODELLO
# =============================================================================
def get_best_kmeans_model():
    """
    Restituisce il miglior modello KMeans identificato con KneeLocator.
    
    Returns:
        tuple: (kmeans_model, best_k, metrics_dict)
            - kmeans_model: il modello KMeans addestrato con il miglior K
            - best_k: numero di cluster ottimale
            - metrics_dict: dizionario con le metriche WCSS del modello
    """
    metrics = {
        'WCSS': kmeans_final.inertia_,
        'selection_method': selection_reason,
        'k_range_tested': list(k_range),
        'wcss_values': wcss
    }
    return kmeans_final, best_k, metrics

# Se lo script viene eseguito direttamente, mostra il riepilogo
if __name__ == "__main__":
    model, k, metrics = get_best_kmeans_model()
    print(f"\nModello disponibile tramite get_best_kmeans_model()")
    print(f"K ottimale: {k}")
    print(f"WCSS: {metrics['WCSS']:.2f}")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

# Configurazione seed per riproducibilità
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Caricamento dei dati (path relativo al file dello script)
from pathlib import Path
data_path = Path(__file__).parent / "Autism-Dataset.csv"
data = pd.read_csv(data_path)

# Seleziona le colonne desiderate per le domande e is_autistic
selected_columns = ["A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score", "A6_Score", "A7_Score", "A8_Score", "A9_Score", "A10_Score", "is_autistic"]
# Filtra il dataset con le colonne selezionate
filtered_data = data[selected_columns]

# Calcola le medie delle risposte per i gruppi "is_autistic" 0 e 1
mean_responses = filtered_data.groupby("is_autistic").mean()

# Crea un grafico a barre per le differenze nelle medie delle risposte
fig, ax = plt.subplots(figsize=(10, 6))
mean_responses.T.plot(kind="bar", ax=ax, color=["skyblue", "orange"])
ax.set_title("Differenze nelle risposte tra gruppo is_autistic 0 e 1")
ax.set_xlabel("Domande")
ax.set_ylabel("Media delle risposte")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

# Stampa i valori sulle colonne
for container in ax.containers:
    ax.bar_label(container, fmt='%2.2f', label_type='edge', color='black')

ax.legend(["Non Autistico (0)", "Autistico (1)"])
plt.show()
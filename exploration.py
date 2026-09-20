import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement du dataset brut
df = pd.read_csv("data/songs.csv")
df.head()

# Vue d'ensemble
print(df.shape)
df.info()
df.describe()

# Valeurs manquantes
print(df.isnull().sum())

# On supprime les lignes sans nom de chanson ou d'artiste
df = df.dropna(subset=["track_name", "artists"])
print("Après dropna :", df.shape)

# Suppression des colonnes inutiles (anciens index du fichier)
df = df.drop(columns=["Unnamed: 0.1", "Unnamed: 0"])
print("Après suppression des colonnes :", df.shape)

# Doublons
print("Doublons trouvés :", df.duplicated().sum())
df = df.drop_duplicates()
print("Après drop_duplicates :", df.shape)

# Analyse des caractéristiques musicales
df[[
    "danceability", "energy", "acousticness",
    "valence", "tempo", "loudness", "duration_ms"
]].describe()

# Valeurs anormales : tempo = 0 et durée < 30 secondes
print("Tempo = 0 :", (df["tempo"] == 0).sum())
print("Durée < 30 secondes :", (df["duration_ms"] < 30000).sum())

df = df[df["tempo"] > 0]
df = df[df["duration_ms"] >= 30000]
df = df.reset_index(drop=True)
print("Dataset final nettoyé :", df.shape)

# Sauvegarde du dataset nettoyé pour l'application Streamlit
df.to_csv("data/songs_clean.csv", index=False)
print(" Fichier data/songs_clean.csv créé.")

# Graphiques d'exploration
sns.histplot(df["energy"])
plt.title("Distribution de l'énergie des chansons")
plt.xlabel("Energy")
plt.ylabel("Nombre de chansons")
plt.show()

sns.scatterplot(data=df, x="energy", y="danceability")
plt.title("Énergie vs Danceability")
plt.xlabel("Energy")
plt.ylabel("Danceability")
plt.show()

features_corr = ["danceability", "energy", "tempo", "loudness", "valence", "acousticness"]
plt.figure(figsize=(8, 6))
sns.heatmap(df[features_corr].corr(), annot=True, cmap="coolwarm")
plt.title("Corrélation entre caractéristiques musicales")
plt.show()

# Système de recommandation
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

FEATURES = ["danceability", "energy", "tempo", "loudness", "valence", "acousticness"]

scaler = StandardScaler()
X = scaler.fit_transform(df[FEATURES])

# Modèle de recherche des voisins les plus proches (similarité cosinus)
model = NearestNeighbors(n_neighbors=20, metric="cosine")
model.fit(X)

# Fonction de recommandation (avec filtre anti-doublon)
def recommend_song(song_name, df, model, X, n=5):
    matches = df[df["track_name"] == song_name]
    if matches.empty:
        return f"Chanson '{song_name}' introuvable dans le dataset."

    index = matches.index[0]
    distances, indices = model.kneighbors([X[index]], n_neighbors=n + 10)

    recommendations = []
    for i, dist in zip(indices[0], distances[0]):
        candidate_name = df.iloc[i]["track_name"]
        if candidate_name == song_name:
            continue

        similarity_score = 1 - dist
        recommendations.append({
            "song": candidate_name,
            "artist": df.iloc[i]["artists"],
            "similarity": round(float(similarity_score), 3)
        })

        if len(recommendations) == n:
            break

    return recommendations

# Test
df["track_name"].sample(5)

recommend_song("I Bet On Flying High", df, model, X)
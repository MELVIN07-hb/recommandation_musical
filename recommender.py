"""
recommender.py
Contient toute la logique du système de recommandation de chansons :
- chargement des données déjà nettoyées
- construction du modèle de similarité (NearestNeighbors)
- fonction de recommandation
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Les 6 caractéristiques musicales utilisées pour comparer les chansons
FEATURES = ["danceability", "energy", "tempo", "loudness", "valence", "acousticness"]


def load_data(path="data/songs_clean.csv"):
    """Charge le dataset déjà nettoyé (voir notebook d'exploration)."""
    df = pd.read_csv(path)
    return df


def build_model(df, features=FEATURES, n_neighbors=20):
    """
    Normalise les caractéristiques et construit le modèle de recherche
    des voisins les plus proches (similarité cosinus).

    n_neighbors=20 : marge de sécurité pour pouvoir filtrer les doublons
    (une même chanson peut apparaître plusieurs fois avec des genres différents).
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])

    model = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    model.fit(X)

    return model, X


def recommend_song(song_name, df, model, X, n=5):
    """
    Retourne les n chansons les plus similaires à `song_name`.
    Retourne une liste vide si la chanson n'existe pas dans le dataset.
    """
    matches = df[df["track_name"] == song_name]
    if matches.empty:
        return []

    index = matches.index[0]

    # On demande plus de voisins que nécessaire (n+10) car certains seront
    # écartés s'ils correspondent à la chanson elle-même (doublon de genre)
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
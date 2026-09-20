"""
app.py
Interface Streamlit du système de recommandation de chansons.
"""

import streamlit as st
from recommender import load_data, build_model, recommend_song

st.set_page_config(page_title=" Recommandation de chansons", layout="centered")

st.title(" Système de recommandation de chansons")
st.write(
    "Choisis une chanson et découvre 5 titres similaires, "
    "basés sur leurs caractéristiques musicales (danceability, energy, "
    "tempo, loudness, valence, acousticness)."
)


# @st.cache_data : garde les données en mémoire après le premier chargement.
# Sans ça, Streamlit relit le CSV à chaque clic de l'utilisateur (très lent).
@st.cache_data
def get_data():
    return load_data()


# @st.cache_resource : comme cache_data, mais pour des objets non-tabulaires
# (ici le modèle NearestNeighbors). Sans ça, le modèle serait recalculé
# à chaque interaction avec la page.
@st.cache_resource
def get_model(df):
    return build_model(df)


df = get_data()
model, X = get_model(df)

# Liste triée des chansons disponibles (l'utilisateur peut taper pour filtrer)
song_list = sorted(df["track_name"].dropna().unique())
selected_song = st.selectbox("Choisis une chanson :", song_list)

if st.button(" Recommander"):
    results = recommend_song(selected_song, df, model, X)

    if not results:
        st.error("Aucune recommandation trouvée pour cette chanson.")
    else:
        st.subheader(f"Chansons similaires à « {selected_song} »")
        for r in results:
            st.write(f"**{r['song']}** — {r['artist']}  \nSimilarité : {r['similarity']}")
            st.divider()
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import umap
import plotly.express as px
import seaborn as sns
import kagglehub

randomState = 42

path = kagglehub.dataset_download("rodolfofigueroa/spotify-12m-songs")

music = pd.read_csv(path + "/tracks_features.csv")

X = music.drop(columns=["id", "name", "album", "album_id", "artists", "artist_ids", "track_number", "disc_number", "liveness", "explicit", "year", "release_date"])



scaler = StandardScaler()

pca = PCA(n_components=11, random_state=randomState)

kmean = KMeans(n_clusters=20, random_state=randomState)

pcaPipeline = Pipeline(steps=[("scaler",scaler),("pca",pca), ("kmeans",kmean)])

pcaPipeline.fit(X)

X_scaled = pcaPipeline.named_steps["scaler"].transform(X)
X_pca = pcaPipeline.named_steps["pca"].transform(X_scaled)

cluster_labels = pcaPipeline.named_steps["kmeans"].labels_

pca_df = pd.DataFrame({
    "PCA1": X_pca[:, 0],
    "PCA2": X_pca[:, 1],
    "Cluster": cluster_labels
})

fig_umap = px.scatter(
    pca_df,
    x="PCA1",
    y="PCA2",
    color="Cluster",
    color_continuous_scale="Viridis",
    opacity=0.7,
    title="PCA Visualization of K-Means Clusters"
)

fig_umap.update_traces(marker=dict(size=4))
fig_umap.update_layout(width=900, height=650)
fig_umap.show()


plot_df = X.copy()
plot_df["Cluster"] = cluster_labels

features = ["danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "valence", "tempo"]


for feature in features:
    fig = px.histogram(
        plot_df,
        x=feature,
        color="Cluster",
        marginal="violin",
        opacity=0.6,
        nbins=50,
        title=f"Distribution of {feature} by Cluster"
    )
    
    fig.update_layout(bargap=0.05, width=900, height=450)
    fig.show()

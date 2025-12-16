# =====================================
# 🎵 Amazon Music Clustering Dashboard
# =====================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="Amazon Music Clustering", layout="wide")

# ===============================
# Load Data
# ===============================

@st.cache_data
def load_data():
    df = pd.read_csv("C:\\Users\\USER\\OneDrive\\Desktop\\Amazon cluster\\amazon_music_final_clusters.csv")
    summary = pd.read_csv("C:\\Users\\USER\\OneDrive\\Desktop\\Amazon cluster\\amazon_music_final_summary.csv")

    # Ensure 'cluster' column exists in summary
    if "cluster" not in summary.columns:
        summary.reset_index(inplace=True)
        summary.rename(columns={"index": "cluster"}, inplace=True)

    return df, summary

df, summary = load_data()

# ===============================
# App Title
# ===============================

st.title("🎧 Amazon Music Clustering Dashboard")
st.markdown("Explore how songs are grouped into clusters using audio features like energy, danceability, tempo, and acousticness.")

# ===============================
# Sidebar Controls
# ===============================

st.sidebar.header("🔍 Cluster Selection")

clusters = sorted(df["cluster"].unique())
selected_cluster = st.sidebar.selectbox("Select a Cluster:", clusters)

st.sidebar.header("🎚️ Feature Filters")

features = [
    "danceability","energy","loudness","speechiness",
    "acousticness","instrumentalness","liveness",
    "valence","tempo","duration_min"
]

selected_features = st.sidebar.multiselect(
    "Select Features to View:",
    features,
    default=["danceability", "energy", "tempo"]
)

# ===============================
# Cluster Summary Section
# ===============================

st.subheader(f"📊 Cluster {selected_cluster} Summary")

col1, col2 = st.columns(2)

cluster_data = df[df["cluster"] == selected_cluster]

with col1:
    st.metric("Number of Songs", len(cluster_data))
    st.metric("Average Energy", f"{cluster_data['energy'].mean():.2f}")
    st.metric("Average Danceability", f"{cluster_data['danceability'].mean():.2f}")
    st.metric("Average Tempo (BPM)", f"{cluster_data['tempo'].mean():.1f}")

with col2:
    st.write("### Feature Means for Cluster")
    try:
        st.dataframe(
            summary[summary["cluster"] == selected_cluster]
            .set_index("cluster")
            .T
        )
    except:
        st.dataframe(summary)

# ===============================
# Heatmap Comparison
# ===============================

st.subheader("🔥 Feature Comparison Across Clusters")

# Determine correct index
if "cluster" in summary.columns:
    idx_col = "cluster"
elif "cluster_label" in summary.columns:
    idx_col = "cluster_label"
else:
    idx_col = summary.columns[0]

heatmap_data = summary.set_index(idx_col).select_dtypes(include=["float64", "int64"])

plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Cluster Feature Heatmap")
st.pyplot(plt)

# ===============================
# PCA Visualization
# ===============================

st.subheader("🎨 PCA Visualization of Clusters")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=df["cluster"],
    palette="tab10",
    s=10,
    alpha=0.7
)
plt.title("PCA 2D Projection of Songs by Cluster")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
st.pyplot(plt)

# ===============================
# Footer
# ===============================

st.markdown("---")
st.caption("Amazon Music Clustering Dashboard | Developed by T Sarath Sundar")

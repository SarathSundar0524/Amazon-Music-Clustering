# Amazon Music Clustering

## Project Overview
This project focuses on grouping Amazon Music tracks into meaningful clusters using unsupervised machine learning. Instead of relying on predefined genre labels, songs are clustered based purely on their audio characteristics, allowing hidden patterns and musical similarities to emerge.

---

## Project Overview

### 1. Data Understanding & Cleaning
- Loaded the Amazon Music audio dataset.
- Removed non-essential metadata such as song IDs, artist names, popularity metrics, and genres.
- Retained only numerical audio features relevant for clustering.
- Converted song duration from milliseconds to minutes for better interpretability.

---

### 2. Feature Selection
Selected audio features that describe the sound and mood of a track:
- Danceability  
- Energy  
- Loudness  
- Speechiness  
- Acousticness  
- Instrumentalness  
- Liveness  
- Valence  
- Tempo  
- Duration (minutes)

These features collectively represent rhythm, intensity, mood, and instrumentation.

---

### 3. Exploratory Data Analysis (EDA)
- Plotted feature distributions to understand data spread and skewness.
- Used correlation heatmaps to identify relationships between audio features.
- Analyzed summary statistics to validate feature ranges and quality.

---

### 4. Data Scaling
- Applied StandardScaler to normalize features.
- Ensured that distance-based algorithms like K-Means were not biased by feature scale.

---

### 5. Dimensionality Reduction
- Used PCA (Principal Component Analysis) to reduce dimensions.
- Visualized song distribution in 2D space before and after clustering.

---

### 6. Clustering & Model Selection
- Applied K-Means clustering on scaled audio features.
- Determined the optimal number of clusters using:
  - Elbow Method (SSE)
  - Silhouette Score
- Selected k = 3 based on best cluster separation and interpretability.
- Compared results with DBSCAN and Agglomerative clustering for validation.

---

### 7. Cluster Interpretation
Interpreted clusters using average feature values:

- Cluster 0 – Energetic & Danceable  
  High danceability and valence, suitable for upbeat or party tracks.

- Cluster 1 – Calm Acoustic  
  High acousticness and low energy, representing relaxed or chill music.

- Cluster 2 – Balanced Mainstream  
  Higher energy and tempo, typical of popular commercial tracks.

---

### 8. Visualization & Dashboard
- Created cluster-wise heatmaps and boxplots for feature comparison.
- Built an interactive Streamlit dashboard to explore:
  - Cluster statistics
  - Feature comparisons
  - PCA-based cluster visualization

---

### 9. Final Output
- Exported clustered dataset with labels.
- Generated cluster summary statistics.
- Ensured reproducibility and clean structure for deployment and review.

---

## Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- Streamlit  

---

## Domain
Music Analytics | Unsupervised Machine Learning

---

## Key Takeaways
- Demonstrated how unsupervised learning can uncover meaningful structure in audio data.
- Highlighted the importance of scaling and evaluation in clustering.
- Built a complete end-to-end ML pipeline from raw data to interactive visualization.


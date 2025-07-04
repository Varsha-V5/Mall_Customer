"""Importing the required libraries"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
pd.set_option("display.max_columns", None)

"""Importing the dataset and converting into dataframe"""
data = pd.read_csv("Mall_Customers.csv")
print("Printing the dataset:\n", data.head(5))
df = pd.DataFrame(data)
print("Converting into DataFrame:\n", df.head(5))

"""Sanity check of the dataset"""
print("Shape:\n", df.shape)
print("Dimension:\n", df.ndim)
print("Describe:\n", df.describe())
print("Info:\n", df.info())
print("Unique:\n", df.nunique())
print("Duplicates:\n", df.duplicated().sum())
print("Missing Values:\n", df.isnull().sum())

"""Data Preprocessing"""
"""Outlier Treatment"""
x = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
for col in x.columns:
    plt.figure(figsize=(10, 8))
    sns.boxplot(x=x[col], vert=False)
    plt.savefig(f"Box plot for outlier {col}")
    plt.show()
q1 = x["Annual Income (k$)"].quantile(0.25)
q3 = x["Annual Income (k$)"].quantile(0.75)
iqr = q3-q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
df = df[(df["Annual Income (k$)"] >= lower_bound) & (df["Annual Income (k$)"] <= upper_bound)]

"""Encoding the Gender column"""
label_encode = LabelEncoder()
data_en = label_encode.fit_transform(df["Gender"])
data_df = pd.DataFrame(data_en)
data_df.columns = ["Gender_Num"]
df_data = pd.concat([df, data_df], axis=1)
df_data.drop(columns=["CustomerID", "Gender"], inplace=True)
print(df_data.head(5))

"""Standardising the age, income, spending score columns"""
std_sc = StandardScaler()
scaled_df = std_sc.fit_transform(df_data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]])
sc_df = pd.DataFrame(scaled_df, columns=["Scaled_Age", "Scaled_Income", "Scaled_Score"])
df_sc = pd.concat([sc_df, df_data], axis=1)
df_sc.drop(columns=["Age", "Annual Income (k$)", "Spending Score (1-100)"], inplace=True)
print(df_sc.head(5))

"""Visualizing the dataset"""
plt.figure(figsize=(12, 8))
cor = df_sc.corr()
sns.heatmap(cor, annot=True)
plt.savefig("Heatmap")
plt.show()
gender_count = df["Gender"].value_counts()
plt.pie(gender_count, labels=gender_count.index)
plt.savefig("Pie Chart")
plt.show()
sns.histplot(x=df["Spending Score (1-100)"],hue=df["Gender"])
plt.savefig("Spending Score based on Gender")
plt.show()
sns.histplot(x=df["Annual Income (k$)"],hue=df["Gender"])
plt.savefig("Annual Income based on Gender")
plt.show()
sns.scatterplot(x=df["Spending Score (1-100)"],y=df["Annual Income (k$)"])
plt.savefig("Spending Score and Annual Income")
plt.show()

"""Find the optimal number of cluster using elbow method"""
inertia = []
k_range = range(1, 15)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_sc)
    inertia.append(kmeans.inertia_)
plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method For Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.savefig("Elbow Method for Optimal K")
plt.show()

"""Clustering"""
x = df_sc[["Scaled_Age", "Scaled_Income", "Scaled_Score", "Gender_Num"]]

"""K-means Clustering"""
kmeans = KMeans(n_clusters=4, init="k-means++", random_state=42)
kmeans.fit(x)
labels_k = kmeans.labels_
print(labels_k)
object_func = kmeans.inertia_
print("Objective Function Value:\n", object_func)
sil_score_k = silhouette_score(df_sc, labels_k)
print("Silhouette score:\n", sil_score_k)

"""DB-SCAN Clustering"""
dbscan = DBSCAN(eps=0.5, min_samples=2)
labels_db = dbscan.fit_predict(x)
print(labels_db)
sil_score_db = silhouette_score(df_sc, labels_db)
print("Silhouette score:\n", sil_score_db)

"""Agglomerative Clustering"""
agg_clustering = AgglomerativeClustering(n_clusters=4, linkage='ward')
labels_agg = agg_clustering.fit_predict(x)
print(labels_agg)
sil_score_agg = silhouette_score(df_sc, labels_agg)
print("Silhouette score:\n", sil_score_agg)

"""Visualizing the clustering by dimensionality reduction on dataset"""
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

plt.figure(figsize=(8, 6))

"""K-means Clustering"""
sns.scatterplot(x=x_pca[:, 0], y=x_pca[:, 1], hue=labels_k, palette='viridis', s=100, alpha=0.7, edgecolor='k')
plt.title('KMeans Clusters (PCA projection)', fontsize=16)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig("K-Means Cluster")
plt.show()
"""DBSCAN Clustering"""
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=labels_db, cmap="viridis", edgecolor="k")
plt.title("DBSCAN Clustering")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Cluster")
plt.savefig("DBSCAN Cluster")
plt.show()
"""Agglomerative Clustering"""
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=labels_agg, cmap='viridis', edgecolors='k')
plt.title("Agglomerative Clustering")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label='Cluster')
plt.savefig("Agglomerative Cluster")
plt.show()

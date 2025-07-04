Customer segmentation is a crucial task in marketing and business analytics, helping businesses understand customer behaviour and optimize their strategies. 
This project, "Customer Segmentation (Mall Customers)" aims to classify customers based on key demographic and spending behaviour attributes using various clustering techniques.

The dataset consists of columns such as 
1.	 Customer ID  –  The Identification Number of the Customer.
2.	 Age  –  Age of the Customer.
3.	 Annual income – Income of the Customer in k$.
4.	 Gender – Gender of the Customer.
5.	 Spending score – Scoring based on Spending from 1 to 100.

 Steps in Data Analytics include :
1.	Importing the  Dataset
2.	Sanity Check
3.	Data Preprocessing including outlier treatment, encoding and standardization. 
4.	Data Visualization

Machine Learning Algorithm like K-Means, DBSCAN, and Agglomerative Clustering were performed for segmentation to ensure accuracy and consistency.
Additionally, Principal Component Analysis (PCA) was used for dimensionality reduction.
Metric Like Silhouette Score is used for evaluation.
The study demonstrates how clustering techniques can be leveraged for targeted marketing and personalized customer engagement strategies.

This project follows a structured approach to customer segmentation using unsupervised machine learning techniques. 
The methodology consists of several key steps, from data acquisition to clustering and evaluation, ensuring a systematic and insightful analysis of mall customer behaviour.
     
1. Data Collection:
     The dataset used in this project, Mall_Customers.csv, contains demographic and spending-related information about customers. The data is imported using Pandas from kaggle, and an initial examination of the dataset is performed to understand its structure. Imported the required libraries.
   
2. Data Preprocessing and Cleaning:
     To ensure data quality and improve clustering results, the following preprocessing steps are applied:
•	Sanity Check: Analysing dataset properties such as shape, missing values, duplicates, and unique values.
•	Handling Missing Values: Checking for null values and addressing them if necessary.
•	Outlier Detection and Treatment: Using box plots to identify outliers, particularly in the Annual Income and Spending Score columns, and removing extreme values to avoid skewed clustering results.
•	Encoding Categorical Variables: Transforming categorical data (e.g., Gender) into numerical format using Label Encoding for machine learning compatibility.
•	Feature Scaling: Standardizing numerical features (Age, Annual Income, Spending Score) using Standard Scaler to ensure all attributes contribute equally to clustering.

3. Exploratory Data Analysis (EDA):
   EDA helps uncover patterns and relationships between variables using data visualization techniques:
•	Correlation Heatmaps: Identifying relationships between numerical features.
•	Histograms: Understanding data distribution and potential anomalies.
•	Scatter Plots: Observing trends between income and spending behaviour.

4. Clustering Techniques and Evaluation:
   Three different clustering algorithms are applied to segment customers:
•	K-Means Clustering: A centroid-based method to partition data into k clusters. The Elbow Method is used to determine the optimal number of clusters.
•	DBSCAN (Density-Based Spatial Clustering of Applications with Noise): A density-based clustering algorithm that identifies high-density regions while considering noise points.
•	Agglomerative Clustering: A hierarchical clustering method that groups customers based on similarity measures.
     Each technique is evaluated using Silhouette Score, which measures how well-separated the clusters are.
   
5. Dimensionality Reduction and Visualization:
     To visualize high-dimensional data effectively, Principal Component Analysis (PCA) is applied to reduce the dataset to two principal components. Scatter plots are then used to visualize clusters, helping interpret the segmentation results.

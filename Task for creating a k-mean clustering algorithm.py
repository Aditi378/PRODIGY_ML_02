import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Sample purchase history data
# Each row represents a customer and columns represent purchase amounts for different categories
data = {
    'Electronics': [250, 350, 450, 150, 550, 650, 850],
    'Clothing': [60, 120, 140, 180, 280, 320, 380],
    'Groceries': [25, 35, 45, 55, 65, 75, 85],
    'Home': [150, 250, 100, 320, 480, 520, 700]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Define the number of clusters
k = 4

# Create the KMeans model
kmeans = KMeans(n_clusters=k, random_state=42)

# Fit the model
kmeans.fit(scaled_data)

# Get cluster labels
df['Cluster'] = kmeans.labels_

# Print the DataFrame with cluster labels
print(df)

# Optional: Visualize the clusters
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=kmeans.labels_, cmap='viridis')
plt.xlabel('Electronics')
plt.ylabel('Clothing')
plt.title('Customer Clusters Based on Purchase History')
plt.show()

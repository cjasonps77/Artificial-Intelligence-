# %% Import Library
from sklearn.preprocessing import StandardScaler
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% Reading the File
data = pd.read_csv('Mall_Customers.csv')
data.head()

# %% 
data.rename(index=str, columns={'Annual Income (k$)': 'Income', 'Spending Score (1-100)': 'Score'}, inplace=True)
data.head()
# %% Clustering and Visualisation
from sklearn.cluster import KMeans
X = data.drop(['CustomerID', 'Gender'], axis=1)
km = KMeans(n_clusters=3).fit(X)
X['Labels'] = km.labels_
plt.figure(figsize=(12, 8))
sns.scatterplot(X['Income'], X['Score'], hue=X['Labels'], palette=sns.color_palette('hls', 3))
plt.title('KMeans with 3 Clusters')
plt.show()


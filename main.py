#import all the necessary libraries
import numpy as np
import pandas as pd

#import matplotlib library for visualization
import matplotlib.pyplot as plt

#import standardscaler and kmeans for clustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#import pca for dimensionality reduction
from sklearn.decomposition import PCA

#import folium for visualization of clusters
import folium
from folium.plugins import MarkerCluster
from matplotlib import cm

#import IFrame to display the map
from IPython.display import IFrame

#data cleaning and extracting relevant features
cook_book = pd.read_csv(r'C:\Users\hp\OneDrive\Documents\B Tech\SEM 7\ml intern\food_coded.csv')
cook_book.drop_duplicates()
columns_to_drop = ['Gender', 'GPA', 'breakfast',
    'calories_chicken', 'calories_day', 'calories_scone', 
    'coffee', 'comfort_food', 'comfort_food_reasons', 
    'comfort_food_reasons_coded', 'cuisine', 'diet_current', 
    'diet_current_coded', 'drink', 'eating_changes', 
    'eating_changes_coded', 'eating_changes_coded1',
    'father_education', 'father_profession', 'fav_cuisine',
    'fav_cuisine_coded', 'fav_food', 'food_childhood', 'fries',
    'grade_level', 'greek_food', 'healthy_feeling',
    'healthy_meal', 'ideal_diet', 'ideal_diet_coded',
    'indian_food', 'italian_food', 'life_rewarding',
    'marital_status', 'meals_dinner_friend',
    'mother_education', 'mother_profession',
    'nutritional_check', 'parents_cook', 'persian_food',
    'self_perception_weight' , 'soup', 'sports', 'thai_food',
    'tortilla_calories', 'turkey_calories', 'comfort_food_reasons_coded.1', 
    'vitamins', 'waffle_calories', 'weight']

existing_columns_to_drop = [col for col in columns_to_drop if col in cook_book.columns]
cook_book = cook_book.drop(existing_columns_to_drop, axis=1)
numeric_data = cook_book.select_dtypes(include=['number'])
numeric_data = numeric_data.dropna()

#plotting boxplot for clean data
fig, ax = plt.subplots(figsize = (12, 6))
colors = ['peachpuff', 'orange', 'tomato', 'pink', 'limegreen', 'lightblue', 'purple', 'skyblue', 'salmon', 'yellow']
colors = colors[:len(numeric_data.columns)]
medianprops = dict(color="black",linewidth=1.5)
bplot = ax.boxplot(numeric_data.values, patch_artist = True, labels = numeric_data.columns, medianprops = medianprops)
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
ax.set_title('Geolocational Data')
ax.set_ylabel('Numbers')
plt.xticks(rotation = 45)
plt.show()
cook_book.isna().sum() # Missing values
np.isinf(cook_book).sum()  # Infinite values

# Fix invalid values
# Replace infinite values with NaN
cook_book = cook_book.replace([np.inf, -np.inf], np.nan)
# Fill or drop missing values
cook_book = cook_book.fillna(cook_book.mean())

# Ensure all values are numeric
print(cook_book.applymap(np.isreal).all())
cook_book.isna().sum()
cook_book.fillna(cook_book.mean)
cook_book = cook_book.select_dtypes(include=['float64', 'int64'])

#kmeans clustering
scaler = StandardScaler()
cook_book_scaled = scaler.fit_transform(cook_book)
kmeans = KMeans(n_clusters = 3, random_state = 0, n_init = 'auto')
kmeans.fit(cook_book_scaled)
kmeans.labels_

# Plotting the clusters
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(cook_book_scaled)
plt.figure(figsize=(8,6))
for i in range(3):
    cluster_points = reduced_data[kmeans.labels_ == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i + 1}")
centroids = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='X', s=200, label='Centroids')
plt.title("KMeans Clustering Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid()
plt.show()
#elbow method
wcss = []  # List to store WCSS for each value of K

# Compute WCSS for different values of K
for k in range(1, 11):  # Try K values from 1 to 10
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
    kmeans.fit(cook_book_scaled)
    wcss.append(kmeans.inertia_)
# Plot the Elbow graph
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Inertia)')
plt.grid()
plt.show()

#api cleaning and extracting relevant features
data_api = pd.read_csv(r'C:\Users\hp\OneDrive\Documents\B Tech\SEM 7\ml intern\cleaned_apartment.csv')
X = data_api[['position.lat', 'position.lng']]
inertia = []
# Run K-Means for different values of K
K_values = range(1, 10)
for k in K_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
# Plot the Elbow Curve
plt.figure(figsize=(8, 6))
plt.plot(K_values, inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.show()
# Set the optimal K (e.g., based on the elbow point)
optimal_k = 4  # Replace this with the chosen K value
# Run K-Means with the optimal K
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data_api['Cluster'] = kmeans.fit_predict(X)
# Display the data with cluster assignments
print(data_api.head())


# Initialize the map centered at an average location of the dataset
center_lat = data_api['position.lat'].mean()
center_lon = data_api['position.lng'].mean()
map_clusters = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Generate a color scheme
n_clusters = data_api['Cluster'].nunique()
colors = cm.get_cmap('tab10', n_clusters)  # Use tab10 colormap

# Plot each cluster
for cluster in range(n_clusters):
    cluster_data = data_api[data_api['Cluster'] == cluster]
    cluster_color = f"rgb({int(colors(cluster)[0] * 255)}, {int(colors(cluster)[1] * 255)}, {int(colors(cluster)[2] * 255)})"
    for i, row in cluster_data.iterrows():
        folium.CircleMarker(
            location=(row['position.lat'], row['position.lng']),
            radius=5,
            color=cluster_color,
            fill=True,
            fill_color=cluster_color,
            fill_opacity=0.6,
        ).add_to(map_clusters)

# Add markers to show clustering points
marker_cluster = MarkerCluster().add_to(map_clusters)

# Add each data point into the marker cluster
for _, row in data_api.iterrows():
    folium.Marker(
        location=[row['position.lat'], row['position.lng']],
        popup=f"Cluster: {row['Cluster']} | Latitude: {row['position.lat']} | Longitude: {row['position.lng']}"
    ).add_to(marker_cluster)

# Save the map to an HTML file
map_show = map_clusters.save("clustered_map.html")

# Display the map
IFrame('clustered_map.html', width=800, height=600)

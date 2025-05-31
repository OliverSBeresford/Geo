import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from scipy.stats import linregress
from sklearn.neighbors import KernelDensity  # Ensure KernelDensity is imported
import os
import shutil

# Ensure output directories exist and are empty
for folder in ["images", "html"]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

# Central Davis Square
center_coords = (42.39645794794289, -71.12240324262818)

# Tree & Air Pollution Data
tree_air_data = [
    (0, 1.4, (42.39791366147681, -71.12363294369119)),
    (5, 0.5, (42.39672289769871, -71.12271428506915)),
    (15, 0.5, (42.39599500127744, -71.12110663248046)),
    (5, 1.8, (42.395277697027346, -71.11952768797352)),
    (0, 4.7, (42.39519642558778, -71.11945591776838)),
    (12, 5.9, (42.394101017634284, -71.12061381040657)),
    (0, np.nan, (42.393804473549004, -71.12088396278337)),
    (8, 4.1, (42.395101558846385, -71.12165235680547)),
    (6, 3.5, (42.39633475981494, -71.12240409988938)),
    (14, 2.9, (42.39657385428261, -71.12262012933378)),
    (0, 4.8, (42.39856064785183, -71.12403169330798)),
    (27, 5.4, (42.399677773557436, -71.122969538554)),
    (32, 0.7, (42.40078633324176, -71.12190135666364)),
    (0, 2.3, (42.39803250740352, -71.12257254357206)),
    (41, 1.9, (42.39919559248551, -71.12157904914245)),
    (44, 2.5, (42.4002742026886, -71.12050612157913)),
    (0, 3.1, (42.39599680275488, -71.12604125266843)),
    (32, 2.6, (42.39442186658949, -71.12424188732564)),
    (22, 3.9, (42.39340529052539, -71.12295129849545)),
]

# Car Traffic Data
car_traffic_data = [
    (83, (42.39663230124736, -71.1223139406676)),
    (52, (42.396982312390605, -71.12301857917917)),
    (38, (42.398764955872416, -71.12419298130004)),
    (19, (42.39531874526657, -71.12541133353453)),
    (3,  (42.39261137862512, -71.11972846103302)),
]

# Properly formatted Trash Data (count, (lat, lon))
trash_data = [
    (0, (42.39791366147681, -71.12363294369119)),
    (64, (42.39672289769871, -71.12271428506915)),
    (45, (42.39599500127744, -71.12110663248046)),
    (37, (42.395277697027346, -71.11952768797352)),
    (0, (42.39519642558778, -71.11945591776838)),
    (30, (42.394101017634284, -71.12061381040657)),
    (0, (42.393804473549004, -71.12088396278337)),
    (45, (42.395101558846385, -71.12165235680547)),
    (36, (42.39633475981494, -71.12240409988938)),
    (44, (42.39657385428261, -71.12262012933378)),
    (0, (42.39856064785183, -71.12403169330798)),
    (49, (42.399677773557436, -71.122969538554)),
    (15, (42.40078633324176, -71.12190135666364)),
    (0, (42.39803250740352, -71.12257254357206)),
    (20, (42.39919559248551, -71.12157904914245)),
    (13, (42.4002742026886, -71.12050612157913)),
    (0, (42.39599680275488, -71.12604125266843)),
    (15, (42.39442186658949, -71.12424188732564)),
    (26, (42.39340529052539, -71.12295129849545)),
]

# Convert Tree-Air to DataFrame
df = pd.DataFrame(tree_air_data, columns=["Tree Count", "Air Pollution", "Coordinates"])
df["Latitude"] = df["Coordinates"].apply(lambda x: x[0])
df["Longitude"] = df["Coordinates"].apply(lambda x: x[1])
df["Distance from Center (m)"] = df["Coordinates"].apply(lambda x: geodesic(center_coords, x).meters)
df["Zone"] = df["Distance from Center (m)"].apply(lambda d: "Core (<=200m)" if d <= 200 else "Periphery (>200m)")

# Remove NaNs
df_clean = df.dropna(subset=["Air Pollution"])

# Car traffic DataFrame
car_df = pd.DataFrame(car_traffic_data, columns=["Car Traffic Count", "Coordinates"])
car_df["Latitude"] = car_df["Coordinates"].apply(lambda x: x[0])
car_df["Longitude"] = car_df["Coordinates"].apply(lambda x: x[1])
car_df["Distance from Center (m)"] = car_df["Coordinates"].apply(lambda x: geodesic(center_coords, x).meters)

# Interpolate Air Pollution, Tree Count, and Distance from Center using Inverse Distance Weighting (IDW)
def interpolate_air_quality(coord, power=2):
    # Calculate distances to all air quality points
    distances = df_clean["Coordinates"].apply(lambda x: geodesic(coord, x).meters)
    # Avoid division by zero: if a car point coincides with a sample, return that sample
    if (distances == 0).any():
        idx = distances.idxmin()
        return df_clean.loc[idx, ["Air Pollution", "Tree Count", "Distance from Center (m)"]]
    # Compute weights (inverse distance^power)
    weights = 1 / (distances ** power)
    weights_sum = weights.sum()
    # Weighted average for each variable
    air_pollution = (df_clean["Air Pollution"] * weights).sum() / weights_sum
    tree_count = (df_clean["Tree Count"] * weights).sum() / weights_sum
    dist_center = (df_clean["Distance from Center (m)"] * weights).sum() / weights_sum
    return pd.Series([air_pollution, tree_count, dist_center], index=["Air Pollution", "Tree Count (nearest)", "Air Dist from Center"])

car_df[["Air Pollution", "Tree Count (nearest)", "Air Dist from Center"]] = car_df["Coordinates"].apply(interpolate_air_quality)

# ---------- Plotting ----------
def plot_with_r2(x, y, x_label, y_label, title, color):
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=x, y=y)
    sns.regplot(x=x, y=y, scatter=False, color=color)

    slope, intercept, r, p, std_err = linregress(x, y)
    r_squared = r**2
    plt.text(0.05, 0.95, f"$R^2 = {r_squared:.2f}$", transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"images/{title.replace(' ', '_').lower()}.png", dpi=300)
    plt.show()

# Plotting correlations
plot_with_r2(df_clean["Tree Count"], df_clean["Air Pollution"],
             "Tree Count", "Air Pollution (PM2.5)", "Tree Count vs Air Pollution", "red")

plot_with_r2(df_clean["Distance from Center (m)"], df_clean["Air Pollution"],
             "Distance from Center (m)", "Air Pollution (PM2.5)", "Distance from Center vs Air Pollution", "green")

plot_with_r2(df_clean["Distance from Center (m)"], df_clean["Tree Count"],
             "Distance from Center (m)", "Tree Count", "Distance from Center vs Tree Count", "blue")

plot_with_r2(car_df["Car Traffic Count"], car_df["Air Pollution"],
             "Car Traffic Count (cars)", "Air Pollution (PM2.5)", "Car Traffic vs Air Pollution", "purple")

plot_with_r2(car_df["Car Traffic Count"], car_df["Tree Count (nearest)"],
             "Car Traffic Count (cars)", "Tree Count", "Car Traffic vs Tree Count", "orange")

plot_with_r2(car_df["Car Traffic Count"], car_df["Distance from Center (m)"],
             "Car Traffic Count (cars)", "Distance from Center (m)", "Car Traffic vs Distance from Center", "gray")

# Boxplot
plt.figure(figsize=(6, 4))
sns.boxplot(data=df_clean, x="Zone", y="Air Pollution")
title = "Air Pollution by Zone"
plt.title(title)
plt.ylabel("Air Pollution (PM2.5)")
plt.tight_layout()
plt.savefig(f"images/{title.replace(' ', '_').lower()}.png", dpi=300)
plt.show()

# Fixing the issue by explicitly handling different value columns from the same dataset

def expand_data_fixed(data, value_index):
    return pd.DataFrame({
        "Value": [x[value_index] for x in data],
        "Latitude": [x[2][0] for x in data],
        "Longitude": [x[2][1] for x in data],
    })

# Separate dataframes for each heatmap
tree_df = expand_data_fixed(tree_air_data, 0).rename(columns={"Value": "Tree Count"})
air_df = expand_data_fixed(tree_air_data, 1).rename(columns={"Value": "Air Pollution"})
car_df = pd.DataFrame({
    "Car Traffic": [x[0] for x in car_traffic_data],
    "Latitude": [x[1][0] for x in car_traffic_data],
    "Longitude": [x[1][1] for x in car_traffic_data],
})
trash_df = pd.DataFrame({
    "Trash Count": [x[0] for x in trash_data],
    "Latitude": [x[1][0] for x in trash_data],
    "Longitude": [x[1][1] for x in trash_data],
})

# Drop missing air pollution values
air_df = air_df.dropna(subset=["Air Pollution"])

# Reuse the heatmap plotting function
def plot_heatmap(df, value_col, title):
    plt.figure(figsize=(8, 6))
    x = df["Longitude"].values
    y = df["Latitude"].values
    z = df[value_col].values

    xi, yi = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
    xy_sample = np.vstack([x, y]).T

    kde = KernelDensity(bandwidth=0.0008)
    kde.fit(xy_sample, sample_weight=z)
    zi = np.exp(kde.score_samples(np.vstack([xi.ravel(), yi.ravel()]).T)).reshape(xi.shape)

    plt.contourf(xi, yi, zi, cmap='viridis')
    plt.colorbar(label=value_col)
    plt.scatter(x, y, c='red', s=20, edgecolor='k', label='Data Points')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"images/{title.replace(' ', '_').lower()}.png", dpi=300)
    plt.show()

plot_heatmap(car_df, "Car Traffic", "Heatmap of Car Traffic")
plot_heatmap(tree_df, "Tree Count", "Heatmap of Tree Count")
plot_heatmap(trash_df, "Trash Count", "Heatmap of Trash Count")
plot_heatmap(air_df, "Air Pollution", "Heatmap of Air Pollution")

import folium
from folium.plugins import HeatMap

# Centered around Davis Square
map_center = [42.396632, -71.122314]

def plot_folium_heatmap(df, value_col, title):
    m = folium.Map(location=map_center, zoom_start=16, tiles="OpenStreetMap")
    
    # Create list of [lat, lon, weight]
    heat_data = [[row["Latitude"], row["Longitude"], row[value_col]] for index, row in df.iterrows()]
    # Increase radius and add blur for more interpolation
    HeatMap(heat_data, radius=50, blur=35, max_zoom=13).add_to(m)
    
    title_filename = "html/" + title.lower().replace(" ", "_") + ".html"
    m.save(title_filename)
    print(f"Saved {title} as {title_filename}")

# Generate each heatmap
plot_folium_heatmap(car_df, "Car Traffic", "Heatmap of Car Traffic")
plot_folium_heatmap(tree_df, "Tree Count", "Heatmap of Tree Count")
plot_folium_heatmap(trash_df, "Trash Count", "Heatmap of Trash Count")
plot_folium_heatmap(air_df, "Air Pollution", "Heatmap of Air Pollution")


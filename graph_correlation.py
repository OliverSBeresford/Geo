import pandas as pd
import numpy as np
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import seaborn as sns

# Coordinates of the center of Davis Square
center_coords = (42.3966, -71.1223)

# Tree and air quality data
tree_air_data = pd.DataFrame({
    "Tree Count": [0, 5, 15, 5, 0, 12, 0, 8, 6, 14, 0, 27, 32, 0, 41, 44, 0, 32, 22],
    "Air Quality": [1.4, 0.5, 0.5, 1.8, 4.7, 5.9, np.nan, 4.1, 3.5, 2.9, 4.8, 5.4, 0.7,
                    2.3, 1.9, 2.5, 3.1, 2.6, 3.9],
    "Latitude": [42.39791366147681, 42.39672289769871, 42.39599500127744, 42.395277697027346,
                 42.39519642558778, 42.394101017634284, 42.393804473549004, 42.395101558846385,
                 42.39633475981494, 42.39657385428261, 42.39856064785183, 42.399677773557436,
                 42.40078633324176, 42.39803250740352, 42.39919559248551, 42.4002742026886,
                 42.39599680275488, 42.39442186658949, 42.39340529052539],
    "Longitude": [-71.12363294369119, -71.12271428506915, -71.12110663248046, -71.11952768797352,
                  -71.11945591776838, -71.12061381040657, -71.12088396278337, -71.12165235680547,
                  -71.12240409988938, -71.12262012933378, -71.12403169330798, -71.122969538554,
                  -71.12190135666364, -71.12257254357206, -71.12157904914245, -71.12050612157913,
                  -71.12604125266843, -71.12424188732564, -71.12295129849545]
})

# Calculate distance from center
tree_air_data["Distance from Center (m)"] = tree_air_data.apply(
    lambda row: geodesic(center_coords, (row["Latitude"], row["Longitude"])).meters, axis=1)

# Remove rows with missing air quality
clean_data = tree_air_data.dropna(subset=["Air Quality"])

# Compute correlation coefficients
corr_tree_air = clean_data["Tree Count"].corr(clean_data["Air Quality"])
corr_dist_air = clean_data["Distance from Center (m)"].corr(clean_data["Air Quality"])
corr_dist_tree = clean_data["Distance from Center (m)"].corr(clean_data["Tree Count"])

print(corr_tree_air, corr_dist_air, corr_dist_tree)

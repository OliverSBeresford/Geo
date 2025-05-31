# Davis Square Environmental Data Analysis

This project analyzes and visualizes environmental data collected around Davis Square, including tree counts, air pollution (PM2.5), car traffic, and trash counts. The analysis explores spatial relationships and correlations between these variables using Python.

## Features
- **Spatial Data Interpolation:** Uses Inverse Distance Weighting (IDW) to interpolate air pollution and tree count values for car traffic points.
- **Correlation Analysis:** Plots and calculates $R^2$ values for relationships between tree count, air pollution, car traffic, and distance from the square's center.
- **Heatmaps:** Generates both static (matplotlib) and interactive (Folium) heatmaps for all variables.
- **Zone Analysis:** Compares air pollution levels between the core and periphery of Davis Square.

## Outputs
- Correlation plots (PNG)
- Boxplots by zone (PNG)
- Interactive HTML heatmaps for air pollution, car traffic, tree count, and trash count

## Requirements
- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- geopy
- scipy
- scikit-learn
- folium

Install dependencies with:
```sh
pip install -r requirements.txt
```

## Usage
1. Place your data in the script or adapt the data structures as needed.
2. Run the main script:
   ```sh
   python graph_davis_hyp2.py
   ```
3. View the generated PNG plots and HTML heatmaps in the project directory.

## Files
- `graph_davis_hyp2.py`: Main analysis and visualization script.
- `*.png`: Correlation and boxplot images.
- `heatmap_of_*.html`: Interactive heatmaps for each variable.
- `requirements.txt`: Python dependencies for the project.
- `LICENSE`: MIT License for this project.

## Notes
- All coordinates are centered around Davis Square, Somerville, MA.
- The project is designed for demonstration and educational purposes with sample data.

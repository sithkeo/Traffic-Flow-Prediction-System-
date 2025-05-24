# tools/plot_scats_graph.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

"""
Visualises SCATS sites and their neighbour links using latitude/longitude.
"""

import pandas as pd
import matplotlib.pyplot as plt
from utils.scats_graph import SCATSGraph

import argparse

# Handle command-line argument for radius
parser = argparse.ArgumentParser()
parser.add_argument("--radius", type=float, default=1.0, help="Neighbour radius in km")
args = parser.parse_args()

# Load the parsed SCATS CSV file

df = pd.read_csv("output/Scats_Data_October_2006_parsed.csv")
# Filter out invalid coordinates (0 or missing)
df = df[(df["Latitude"] != 0) & (df["Longitude"] != 0)].dropna(subset=["Latitude", "Longitude"])

# Create the SCATS spatial graph with a radius threshold (km)
graph = SCATSGraph(df, radius_km=args.radius)

# Set up plot
plt.figure(figsize=(10, 10))

# Colour-code nodes by neighbour count
neighbour_counts = [len(graph.graph[n]) for n in graph.nodes["SCATS"]]
plt.scatter(
    graph.nodes["Longitude"],
    graph.nodes["Latitude"],
    c=neighbour_counts,
    cmap="viridis",
    s=50,
    edgecolors='k',
    linewidths=0.2
)
plt.colorbar(label="Neighbour Count")

# Draw neighbour links if radius is reasonable
if args.radius <= 5.0:
    for node_id, neighbours in graph.graph.items():
        lat1 = graph.nodes.loc[graph.nodes["SCATS"] == node_id, "Latitude"].values[0]
        lon1 = graph.nodes.loc[graph.nodes["SCATS"] == node_id, "Longitude"].values[0]
        for nid, _ in neighbours:
            lat2 = graph.nodes.loc[graph.nodes["SCATS"] == nid, "Latitude"].values[0]
            lon2 = graph.nodes.loc[graph.nodes["SCATS"] == nid, "Longitude"].values[0]
            plt.plot([lon1, lon2], [lat1, lat2], color='lightgray', linewidth=0.5)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"SCATS Site Neighbours ({args.radius} km radius)")
plt.grid(True)
plt.tight_layout()
plt.show()

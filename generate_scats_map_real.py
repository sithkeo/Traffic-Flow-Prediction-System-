# generate_scats_map_real.py

"""
Generate road network graph using SCATS sensor locations and predicted volumes.
Weights are computed based on predicted traffic, not historical averages.
Retains original routing, snapping, and visualisation logic.
"""

import os
import pandas as pd
import osmnx as ox
import networkx as nx
from shapely.geometry import Point
from geopandas import GeoDataFrame
import folium
import matplotlib.pyplot as plt


def load_scats_sites(csv_path):
    """Load SCATS site locations from CSV and exclude invalid coordinates."""
    df = pd.read_csv(csv_path)
    sites = df.drop_duplicates(subset="SCATS")[["SCATS", "Location", "Longitude", "Latitude"]].copy()
    sites = sites.dropna(subset=["Longitude", "Latitude"])
    sites = sites[
        sites["Longitude"].between(140.5, 150.5) &
        sites["Latitude"].between(-39.5, -33.5)
    ]
    print(f"Loaded {len(sites)} SCATS sites within Victoria bounds")
    return sites


def build_road_graph(sites, buffer_km=2):
    """Build road graph using convex hull of all SCATS points."""
    from shapely.geometry import MultiPoint
    import geopandas as gpd

    points = [Point(xy) for xy in zip(sites["Longitude"], sites["Latitude"])]
    polygon = gpd.GeoSeries(points).union_all().convex_hull.buffer(buffer_km / 111)
    G = ox.graph_from_polygon(polygon, network_type='drive')
    return nx.DiGraph(G)  # Convert to DiGraph for shortest path compatibility


def snap_sites_to_graph(G, sites):
    """Snap SCATS sites to nearest OSM nodes using an offset for visual alignment."""
        # Apply positional adjustment *before* snapping for better road alignment.
        # Adjust here if SCATS coordinates appear consistently offset from road network.
        # To test raw positions directly, set this to: Point(row["Longitude"], row["Latitude"])
    gdf = GeoDataFrame(
        sites,
        geometry=sites.apply(lambda row: Point(row["Longitude"] + 0.0007, row["Latitude"] + 0.00108), axis=1),
        crs="EPSG:4326"
    )
    gdf["nearest_node"] = gdf["geometry"].apply(lambda pt: ox.distance.nearest_nodes(G, pt.x, pt.y))
    return gdf


def load_predicted_volumes(predicted_csv):
    """Load predicted traffic volumes for each SCATS site from model output."""
    df = pd.read_csv(predicted_csv)
    return dict(zip(df.SCATS.astype(str), df.PredictedVolume))


def compute_travel_time_weights(G, scats_volume_by_node):
    """
    Assign travel time to each edge in the road graph G based on predicted SCATS site traffic volumes.

    For each edge (u -> v):
    - Retrieves the distance (in metres) and converts to km
    - Looks up the predicted traffic volume for the origin node u
    - Applies the provided quadratic formula to estimate speed:
          flow = -1.4648375 * speed^2 + 93.75 * speed
    - Rearranged to compute speed from flow using the quadratic formula
    - Caps the calculated speed at 60 km/h if flow is < 351 veh/h (free-flow threshold)
    - Caps minimum speed at 5 km/h to avoid unrealistic low values
    - Computes travel time in seconds as:
        travel_time = (distance_km / speed) * 3600
      (no flat penalty is applied per edge)

    Integrates our trained ML-predicted SCATS volumes directly into the routing graph,
    giving better travel time estimates based on modelled traffic conditions.
    """
    for u, v, data in G.edges(data=True):
        dist_m = data.get("length", 0)       # Get edge length in metres
        dist_km = dist_m / 1000              # Convert to kilometres

        # 1. Get predicted traffic volume from the origin SCATS node
        volume_raw = scats_volume_by_node.get(u, 0)  # Previously used `v` — this is now fixed
        volume = min(volume_raw, 1500)  # Clamp to stated max volume for road capacity

        # Debug: Log when volume is clamped for transparency
        if volume_raw != volume:
            print(f"[INFO] Volume at node {u} clamped: raw={volume_raw:.1f} → capped={volume}")

        # 2. Invert quadratic flow-speed equation
        # flow = -1.4648375 * speed^2 + 93.75 * speed
        a, b = -1.4648375, 93.75
        discriminant = b**2 - 4 * a * (-volume)

        if discriminant < 0:
            # If no real solution (should be incredibly rare?), fallback to capped free-flow speed
            speed = 60
        else:
            root1 = (b + discriminant**0.5) / (2 * a)
            root2 = (b - discriminant**0.5) / (2 * a)

            # Select root based on whether flow is under or over capacity threshold (351)
            if volume <= 351:
                speed = min(max(root1, root2), 60)  # Cap under-capacity speed at 60 km/h
            else:
                speed = min(root1, root2)  # Use congested (lower) root

        # 3. Final safety check
        speed = max(speed, 5)  # Prevent unrealistic low speed

        # 4. Compute travel time (in seconds)
        travel_time = (dist_km / speed) * 3600  # distance / speed = time [hr], convert to sec

        # 5. Store weight on graph
        data["travel_time"] = travel_time

    print("Assigned travel_time to all edges using updated SCATS volume estimates.")


def example_routing(G, snapped_sites):
    """Prompt user for route endpoints and return the best of top 5 shortest paths."""
    from networkx.algorithms.simple_paths import shortest_simple_paths
    from itertools import islice

    print("Available SCATS sites:")
    print(snapped_sites[['SCATS', 'Location']].drop_duplicates().to_string(index=False))

    start_id = input("Enter the SCATS ID for the start point: ").strip()
    end_id = input("Enter the SCATS ID for the end point: ").strip()

    try:
        start_node = snapped_sites.loc[snapped_sites['SCATS'].astype(str) == start_id, 'nearest_node'].values[0]
        end_node = snapped_sites.loc[snapped_sites['SCATS'].astype(str) == end_id, 'nearest_node'].values[0]
    except IndexError:
        print("Invalid SCATS ID entered. Aborting routing.")
        return []

    try:
        k_paths = list(islice(shortest_simple_paths(G, start_node, end_node, weight='travel_time'), 5))
        travel_times = []
        for idx, path in enumerate(k_paths):
            total = sum(G[u][v]['travel_time'] for u, v in zip(path[:-1], path[1:]))
            scats_nodes = snapped_sites[snapped_sites['nearest_node'].isin(path)]["SCATS"].nunique()
            delay = scats_nodes * 30
            travel_times.append((path, total + delay))
            print(f"Route {idx + 1}: {len(path)} nodes, estimated time = {(total + delay) / 60:.2f} min")

        best_path, best_time = min(travel_times, key=lambda x: x[1])
        print(f"Selected best route from {start_id} to {end_id}: {best_time / 60:.2f} min")

        print_route_summary(best_path, G, snapped_sites)
        return best_path
    except nx.NetworkXNoPath:
        print("No path found between selected SCATS sites.")
        return []


def save_route_to_map(G, route, output_path="scats_route_map.html", snapped_sites=None, show_route=True):
    """Visualise the route and SCATS nodes on a map using Folium."""
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    if show_route and route:
        route_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]
        folium.PolyLine(route_coords, color="red", weight=5, opacity=0.8).add_to(m)
        folium.Marker(route_coords[0], tooltip="Start").add_to(m)
        folium.Marker(route_coords[-1], tooltip="End").add_to(m)

    if snapped_sites is not None:
        for _, row in snapped_sites.iterrows():
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=3,
                color="yellow",
                fill=True,
                fill_opacity=0.5,
                tooltip=f"Original SCATS {row['SCATS']}"
            ).add_to(m)
            
            # Snapped coordinates (blue)
            # Adjust snapped position display if needed (for visual alignment only)
            # To apply a visual offset post-snapping, modify the following line:
            # location=(G.nodes[row["nearest_node"]]['y'] + offset_lat, G.nodes[row["nearest_node"]]['x'] + offset_lon)
            folium.CircleMarker(
                location=(G.nodes[row["nearest_node"]]['y'], G.nodes[row["nearest_node"]]['x']),
                radius=4,
                color="blue",
                fill=True,
                fill_opacity=0.7,
                tooltip=f"SCATS {row['SCATS']} - {row['Location']}"
            ).add_to(m)

    m.fit_bounds(m.get_bounds())
    m.save(output_path)
    print(f"Route map saved to: {output_path}")
    return m


def print_route_summary(route, G, snapped_sites, save_path="segment_times.png"):
    print("\n[DEBUG] Route Summary")
    print(f"Route node count: {len(route)}")

    # Identify unique SCATS sites in route based on nearest_node
    scats_in_path_df = snapped_sites[snapped_sites['nearest_node'].isin(route)]
    scats_in_path = scats_in_path_df["SCATS"].unique()
    print(f"Unique SCATS nodes crossed: {len(scats_in_path)} → {[int(x) for x in scats_in_path]}")

    # Build a reverse lookup: node ID -> SCATS ID
    node_to_scats = dict(zip(snapped_sites['nearest_node'], snapped_sites['SCATS'].astype(str)))

    # Sum travel time for route
    segment_times = [(u, v, G[u][v]['travel_time']) for u, v in zip(route[:-1], route[1:])]
    total_time = sum(t for _, _, t in segment_times)
    delay = len(scats_in_path) * 30

    print(f"Estimated travel time (without delays): {total_time / 60:.2f} min")
    print(f"Estimated travel time (with 30s per SCATS): {(total_time + delay) / 60:.2f} min")

    # Visual breakdown - horizontal bar chart (reverse to show source at top)
    def label_for(u, v):
        left = f"SCATS {node_to_scats[u]}" if u in node_to_scats else str(u)
        right = f"SCATS {node_to_scats[v]}" if v in node_to_scats else str(v)
        return f"{left} -> {right}"

    labels = [label_for(u, v) for u, v, _ in segment_times][::-1]
    durations = [t / 60 for _, _, t in segment_times][::-1]  # convert to minutes and reverse to match

    max_height = 20  # Clamp maximum figure height to prevent overly tall images
    fig_height = min(0.4 * len(labels) + 2, max_height)

    # Assign colours: highlight bars where either u or v is a SCATS site
    scats_nodes = set(snapped_sites['nearest_node'])
    reversed_segments = list(zip(route[:-1], route[1:]))[::-1]
    colors = ["#4CAF50" if u in scats_nodes or v in scats_nodes else "#2196F3" for u, v in reversed_segments]

    plt.figure(figsize=(10, fig_height))
    bars = plt.barh(range(len(durations)), durations, color=colors)
    plt.yticks(range(len(labels)), labels, fontsize=7)
    plt.xlabel("Travel Time (min)")
    plt.title("Segment-wise Travel Time Along Route")
    plt.tight_layout()

    # Overlay travel time on each bar
    for i, (bar, dur) in enumerate(zip(bars, durations)):
        plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, f"{dur:.1f} min",
                 va='center', fontsize=6, color='black')

    # Overlay SCATS site names clearly
    if not scats_in_path_df.empty:
        lines = [f"{int(row.SCATS)}: {row.Location}" for _, row in scats_in_path_df.drop_duplicates('SCATS').iterrows()]
        full_label = "\n".join(lines)
        plt.gcf().text(0.01, 0.02, f"SCATS Sites Used in Route:\n{full_label}", ha='left', fontsize=7)

    # Save to file as well
    plt.savefig(save_path)
    print(f"[INFO] Segment-wise travel time chart saved to: {os.path.abspath(save_path)}")
    plt.show()


if __name__ == "__main__":
    csv_path = "output/Scats_Data_October_2006_parsed.csv"
    predicted_csv = "output/predicted/gru_site_predictions.csv"  # Replace with desired model output

    sites = load_scats_sites(csv_path)
    G = build_road_graph(sites)
    snapped_sites = snap_sites_to_graph(G, sites)

    predicted_volume_map = load_predicted_volumes(predicted_csv)
    scats_volume_by_node = {
        row["nearest_node"]: predicted_volume_map.get(row["SCATS"], 0)
        for _, row in snapped_sites.iterrows()
    }

    compute_travel_time_weights(G, scats_volume_by_node)
    route = example_routing(G, snapped_sites)
    # if route:
    #     print_route_summary(route, G, snapped_sites)
    save_route_to_map(G, route, snapped_sites=snapped_sites, show_route=True) # False to hide route

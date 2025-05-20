import os
import pandas as pd
import osmnx as ox
import networkx as nx
from shapely.geometry import Point
from geopandas import GeoDataFrame
import folium

def load_scats_sites(csv_path):
    """Load SCATS site locations from CSV and exclude sites far outside Victoria."""
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
    """Download road network using convex hull of all SCATS points."""
    from shapely.geometry import MultiPoint
    import geopandas as gpd

    points = [Point(xy) for xy in zip(sites["Longitude"], sites["Latitude"])]
    polygon = gpd.GeoSeries(points).union_all().convex_hull.buffer(buffer_km / 111)
    G = ox.graph_from_polygon(polygon, network_type='drive')
    G = nx.DiGraph(G)  # Convert MultiDiGraph to DiGraph for compatibility with shortest_simple_paths  # Convert MultiDiGraph to DiGraph for compatibility with shortest_simple_paths
    return G

def snap_sites_to_graph(G, sites):
    """Snap SCATS sites to the nearest road network nodes."""
    gdf = GeoDataFrame(
        sites,
        geometry=sites.apply(
        # Apply positional adjustment *before* snapping for better road alignment.
        # Adjust here if SCATS coordinates appear consistently offset from road network.
        # To test raw positions directly, set this to: Point(row["Longitude"], row["Latitude"])
        lambda row: Point(row["Longitude"] + 0.0007, row["Latitude"] + 0.00108), axis=1),
        crs="EPSG:4326"
    )
    gdf["nearest_node"] = gdf["geometry"].apply(lambda pt: ox.distance.nearest_nodes(G, pt.x, pt.y))
    return gdf

def example_routing(G, snapped_sites):
    """Prompt user to select two SCATS IDs and return a single route."""
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
        route = nx.shortest_path(G, start_node, end_node, weight='travel_time')
        print(f"Route found between SCATS {start_id} and SCATS {end_id}, {len(route)} nodes long.")
        return route
    except nx.NetworkXNoPath:
        print(f"No route found between SCATS {start_id} and SCATS {end_id}.")
        return []

def save_route_to_map(G, route, output_path="scats_route_map.html", snapped_sites=None, show_route=True):
    """Visualise the computed route with Folium."""
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    if show_route and route:
        route_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]
        folium.PolyLine(route_coords, color="red", weight=5, opacity=0.8).add_to(m)
        folium.Marker(route_coords[0], tooltip="Start").add_to(m)
        folium.Marker(route_coords[-1], tooltip="End").add_to(m)

    if snapped_sites is not None:
        for _, row in snapped_sites.iterrows():
            # Original coordinates (yellow)
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
                tooltip=f"Snapped SCATS {row['SCATS']} - {row['Location']}"
            ).add_to(m)

    m.fit_bounds(route_coords)
    m.save(output_path)
    print(f"\nRoute map saved to: {output_path}")

def compute_travel_time_weights(G, scats_volume_by_node):
    """Assign travel time to each edge in G based on SCATS flow data and distance."""
    for u, v, data in G.edges(data=True):
        dist_m = data.get("length", 0)
        dist_km = dist_m / 1000

        # Estimate flow using SCATS data for node v (destination)
        volume = scats_volume_by_node.get(v, 0)

        # Solve inverted speed formula
        a, b = -1.4648375, 93.75
        discriminant = b**2 - 4 * a * (-volume)
        if discriminant < 0:
            speed = 60  # fallback to max speed if no real solution
        else:
            root1 = (b + discriminant**0.5) / (2 * a)
            root2 = (b - discriminant**0.5) / (2 * a)
            speed = min(root1, root2) if volume > 351 else max(root1, root2)

        speed = max(speed, 5)  # prevent zero or negative speed
        travel_time = (dist_km / speed) * 3600 + 30  # in seconds with 30s delay
        data["travel_time"] = travel_time

    print("Assigned travel_time to all edges using SCATS volume estimates.")


if __name__ == "__main__":
    csv_path = "output/Scats_Data_October_2006_parsed.csv"
    sites = load_scats_sites(csv_path)
    G = build_road_graph(sites)

    # Load SCATS volume data and compute average volume per site for travel time estimation
    volume_df = pd.read_csv(csv_path)
    scats_avg_volume = volume_df.groupby("SCATS")["Volume"].mean().to_dict()

    snapped_sites = snap_sites_to_graph(G, sites)
    scats_volume_by_node = {
        row["nearest_node"]: scats_avg_volume.get(row["SCATS"], 0)
        for _, row in snapped_sites.iterrows()
    }
    compute_travel_time_weights(G, scats_volume_by_node)
    snapped_sites = snap_sites_to_graph(G, sites)
    route = example_routing(G, snapped_sites)
    save_route_to_map(G, route, snapped_sites=snapped_sites, show_route=True)  # Set to False to hide route

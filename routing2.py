# routing.py (formerly generate_scats_map_real.py)

import os
import sys
import pandas as pd
import osmnx as ox
import networkx as nx
from shapely.geometry import Point
from geopandas import GeoDataFrame
import folium
import matplotlib.pyplot as plt
from search_algorithms import astar, bfs, dfs, gbfs, dijkstra, landmark_astar, reconstruct_path

# Define and ensure route output folder exists
ROUTE_OUTPUT_DIR = "output/routes"
os.makedirs(ROUTE_OUTPUT_DIR, exist_ok=True)


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
    import geopandas as gpd

    # Build convex hull around SCATS sites and buffer it
    points = [Point(xy) for xy in zip(sites["Longitude"], sites["Latitude"])]
    polygon = gpd.GeoSeries(points).union_all().convex_hull.buffer(buffer_km / 111)

    # Filter OSM ways to include only valid driving roads
    custom_filter = (
        '["highway"~"motorway|trunk|primary|secondary|tertiary|residential|unclassified"]'
    )
    G_multi = ox.graph_from_polygon(polygon, custom_filter=custom_filter)

    # Convert MultiDiGraph to DiGraph (removes parallel edges)
    # OSMnx returns MultiDiGraph by default, which complicates routing and makes calculations slower.
    # For routing car traffic only, DiGraph after filtering works better and avoids the complexity of dealing with edge keys.
    G = nx.DiGraph(G_multi)

    # Restore CRS to enable spatial snapping
    G.graph["crs"] = G_multi.graph.get("crs", "EPSG:4326")

    # Prune to largest strongly connected component
    G = G.subgraph(max(nx.strongly_connected_components(G), key=len)).copy()
    print(f"[DEBUG] Pruned to largest strongly connected component: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return G


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
        dist_km = data.get("length", 0) / 1000
        volume_raw = scats_volume_by_node.get(u, 0)
        volume = min(volume_raw, 1500)
        if volume_raw != volume:
            print(f"[INFO] Volume at node {u} clamped: raw={volume_raw:.1f} → capped={volume}")
        a, b = -1.4648375, 93.75
        d = b**2 - 4*a*(-volume)
        if d < 0:
            speed = 60
        else:
            root1 = (b + d**0.5) / (2 * a)
            root2 = (b - d**0.5) / (2 * a)
            speed = min(root1, root2) if volume > 351 else min(max(root1, root2), 60)
        speed = max(speed, 5)
        travel_time = (dist_km / speed) * 3600
        data["travel_time"] = travel_time
    print("Assigned travel_time to all edges using updated SCATS volume estimates.")

def convert_nx_graph_to_search_inputs(G):
    """Convert NetworkX graph to edge/coord format expected by search algorithms."""
    edges = {}
    coords = {}
    for node in G.nodes:
        coords[node] = (G.nodes[node]['x'], G.nodes[node]['y'])
        edges[node] = {
            nbr: min(data["travel_time"] for data in G[node][nbr].values())
            if isinstance(G[node][nbr], dict) and isinstance(next(iter(G[node][nbr].values())), dict)
            else G[node][nbr]["travel_time"] for nbr in G[node]
        }
    return edges, coords


def run_custom_route(G, snapped_sites, start_id, end_id, algo):
    """Run a single routing algorithm between start and end SCATS IDs."""
    start = snapped_sites.loc[snapped_sites['SCATS'].astype(str) == start_id, 'nearest_node'].values[0]
    end = snapped_sites.loc[snapped_sites['SCATS'].astype(str) == end_id, 'nearest_node'].values[0]
    edges, coords = convert_nx_graph_to_search_inputs(G)
    destinations = {end}
    algo_map = {
        "astar": (astar, True),
        "gbfs": (gbfs, True),
        "landmark_astar": (landmark_astar, True),
        "bfs": (bfs, False),
        "dfs": (dfs, False),
        "dijkstra": (dijkstra, True)
    }
    algorithm_func, needs_coords = algo_map[algo]
    if needs_coords:
        final_node, _ = algorithm_func(start, destinations, edges, coords)
    else:
        final_node, _ = algorithm_func(start, destinations, edges)
    return [int(n) for n in reconstruct_path(final_node)] if final_node else []


def run_all_algorithms(G, snapped_sites, start_id, end_id):
    """Run all supported routing algorithms and return route and time for each."""
    algos = ["astar", "bfs", "dfs", "gbfs", "dijkstra", "landmark_astar"]
    results = []
    for algo in algos:
        try:
            route = run_custom_route(G, snapped_sites, start_id, end_id, algo)
            if not route:
                print(f"[WARN] {algo} failed to find a route.")
                continue
            scats_crossed = snapped_sites[snapped_sites['nearest_node'].isin(route)]['SCATS'].nunique()
            total_time = sum(G[u][v]['travel_time'] for u, v in zip(route[:-1], route[1:])) + scats_crossed * 30
            results.append({"algo": algo, "route": route, "time_min": total_time / 60})
        except Exception as e:
            print(f"[ERROR] {algo} failed: {e}")
    return sorted(results, key=lambda r: r["time_min"])


def print_route_summary(route, G, snapped_sites, save_path="segment_times.png"):
    print("[DEBUG] Route Summary")
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
    plt.title(f"{os.path.splitext(os.path.basename(save_path))[0].replace('segment_times_', '').upper()} Route Segment Times")
    plt.tight_layout()

    # Overlay travel time on each bar
    for i, (bar, dur) in enumerate(zip(bars, durations)):
        plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, f"{dur:.1f} min",
                 va='center', fontsize=6, color='black')

    # Overlay SCATS site names clearly
    if not scats_in_path_df.empty:
        lines = [f"{int(row.SCATS)}: {row.Location}" for _, row in scats_in_path_df.drop_duplicates('SCATS').iterrows()]
        full_label = "".join(lines)
        plt.gcf().text(0.01, 0.02, f"SCATS Sites Used in Route:{full_label}", ha='left', fontsize=7)

    # Save to file as well
    plt.savefig(os.path.join(ROUTE_OUTPUT_DIR, os.path.basename(save_path)))
    print(f"[INFO] Segment-wise travel time chart saved to: {os.path.abspath(save_path)}")
    
    plt.close()


def save_multi_route_map(G, results, snapped_sites, start_id, end_id):
    """Visualise all algorithm routes on a single Folium map with distinct colours."""
    m = folium.Map(location=[-37.81, 144.96], zoom_start=13, tiles="OpenStreetMap", control_scale=True)
    colours = ["red", "blue", "green", "orange", "purple", "black"]
    for i, result in enumerate(results):
        coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in result['route']]
        folium.PolyLine(coords, color=colours[i % len(colours)], weight=5 if i > 0 else 8, opacity=0.8,
                        tooltip=f"{result['algo'].upper()} ({result['time_min']:.2f} min)").add_to(m)
    folium.Marker(coords[0], tooltip="Start").add_to(m)
    folium.Marker(coords[-1], tooltip="End").add_to(m)
    output_path = os.path.join(ROUTE_OUTPUT_DIR, f"multi_route_map_{start_id}_to_{end_id}.html")
    m.fit_bounds(m.get_bounds())
    m.save(output_path)
    print(f"[INFO] Multi-route map saved: {output_path}")


if __name__ == "__main__":
    # CLI entry point for routing pipeline. Supports default mode and fallback to example routing.
    mode = sys.argv[1] if len(sys.argv) > 1 else "default"
    start_id = sys.argv[2] if len(sys.argv) > 2 else "970"
    end_id = sys.argv[3] if len(sys.argv) > 3 else "4821"

    csv_path = "output/Scats_Data_October_2006_parsed.csv"
    predicted_csv = "output/predicted/gru_site_predictions.csv"

    sites = load_scats_sites(csv_path)
    G = build_road_graph(sites)
    snapped_sites = snap_sites_to_graph(G, sites)
    predicted_volume_map = load_predicted_volumes(predicted_csv)
    scats_volume_by_node = {
        row["nearest_node"]: predicted_volume_map.get(row["SCATS"], 0)
        for _, row in snapped_sites.iterrows()
    }
    compute_travel_time_weights(G, scats_volume_by_node)

    if mode == "example":
        from routing import example_routing
        route = example_routing(G, snapped_sites)
    else:
        results = run_all_algorithms(G, snapped_sites, start_id, end_id)
        for r in results:
            print(f"{r['algo'].upper()}: {r['time_min']:.2f} min")
            output_chart_path = os.path.join(ROUTE_OUTPUT_DIR, f"segment_times_{r['algo']}_{start_id}_to_{end_id}.png")
            print_route_summary(r['route'], G, snapped_sites, save_path=output_chart_path)

        save_multi_route_map(G, results, snapped_sites, start_id, end_id)

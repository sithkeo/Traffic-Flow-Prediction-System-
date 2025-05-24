import os
import pandas as pd
import re
import networkx as nx
from geopy.distance import geodesic
import folium

def parse_location(loc):
    """Extract Road_A, Direction, Road_B from location string."""
    match = re.match(r"(.+?)\s+([NSEW]{1,2})\s+of\s+(.+)", loc)
    if match:
        return match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
    return None, None, None

def normalize_intersection_id(road_a, road_b):
    """Create a consistent ID for an intersection regardless of road order."""
    roads = sorted([road_a, road_b])
    return f"{roads[0]} & {roads[1]}"

def build_graph_with_intersections(df):
    """Builds a graph using virtual intersections to link SCATS sites."""
    df[["Road_A", "Direction", "Road_B"]] = df["Location"].apply(lambda loc: pd.Series(parse_location(loc)))
    unique_nodes = df.drop_duplicates(subset="SCATS")[["SCATS", "Road_A", "Longitude", "Latitude"]]

    G = nx.DiGraph()

    # Add SCATS nodes
    for _, row in unique_nodes.iterrows():
        G.add_node(row["SCATS"], node_type="scats", road=row["Road_A"], pos=(row["Latitude"], row["Longitude"]))

    # Add virtual intersection nodes and connect SCATS to them
    for _, row in df.iterrows():
        scats_id = row["SCATS"]
        road_a = row["Road_A"]
        road_b = row["Road_B"]
        direction = row["Direction"]
        if pd.isna(road_a) or pd.isna(road_b):
            continue

        intersection_id = normalize_intersection_id(road_a, road_b)

        # Add virtual intersection node (no position data, just logical)
        if not G.has_node(intersection_id):
            G.add_node(intersection_id, node_type="intersection", roads=(road_a, road_b))

        # Add edge from SCATS site to virtual intersection
        G.add_edge(scats_id, intersection_id, direction=direction, distance=0)

    return G

def plot_graph_with_intersections(G, output_html="scats_intersection_graph.html"):
    """Plot the graph with SCATS and intersection nodes."""
    scats_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "scats"]
    if not scats_nodes:
        print("No SCATS nodes to plot.")
        return

    center_lat = sum(G.nodes[n]["pos"][0] for n in scats_nodes) / len(scats_nodes)
    center_lon = sum(G.nodes[n]["pos"][1] for n in scats_nodes) / len(scats_nodes)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    for node, data in G.nodes(data=True):
        if data["node_type"] == "scats":
            folium.CircleMarker(
                location=[data["pos"][0], data["pos"][1]],
                radius=4,
                color="blue",
                fill=True,
                fill_opacity=0.7,
                popup=f"SCATS: {node}, Road: {data['road']}"
            ).add_to(m)
        elif data["node_type"] == "intersection":
            folium.Marker(
                location=[center_lat, center_lon],  # virtual node, approximate
                icon=folium.DivIcon(html=f"<div style='font-size:10px;'>{node}</div>"),
                popup=f"Intersection: {node}"
            ).add_to(m)

    for u, v, data in G.edges(data=True):
        if G.nodes[u]["node_type"] == "scats":
            u_pos = G.nodes[u]["pos"]
            v_pos = [center_lat, center_lon]  # approximation for virtual node
            folium.PolyLine(
                locations=[u_pos, v_pos],
                color="green",
                weight=2,
                tooltip=f"{u} to {v} ({data['direction']})"
            ).add_to(m)

    m.save(output_html)
    print(f"\n✅ Map successfully saved to: {output_html}")

# Example usage
if __name__ == "__main__":
    input_csv = "output/Scats_Data_October_2006_parsed_1.csv"
    df = pd.read_csv(input_csv)
    G = build_graph_with_intersections(df)
    plot_graph_with_intersections(G, output_html="scats_intersection_graph.html")

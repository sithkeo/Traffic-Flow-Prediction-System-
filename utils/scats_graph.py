# scats_graph.py

"""
Builds a spatial graph of SCATS sites using latitude and longitude.
Useful for finding neighbouring sites and estimating directional travel times.
"""

import pandas as pd
from geopy.distance import geodesic

class SCATSGraph:
    def __init__(self, dataframe, radius_km=1.0):
        """
        Initialise the SCATS spatial graph.

        Args:
            dataframe (pd.DataFrame): DataFrame containing SCATS, Latitude, Longitude.
            radius_km (float): Distance in kilometres to define a 'neighbour'.
        """
        self.radius_km = radius_km
        self.nodes = dataframe[['SCATS', 'Latitude', 'Longitude']].drop_duplicates()
        self.graph = self._build_graph()

    def _build_graph(self):
        """Constructs a dictionary-based spatial graph."""
        graph = {}
        for i, row_i in self.nodes.iterrows():
            scats_id_i = row_i['SCATS']
            coord_i = (row_i['Latitude'], row_i['Longitude'])
            neighbours = []
            for j, row_j in self.nodes.iterrows():
                if i == j:
                    continue
                coord_j = (row_j['Latitude'], row_j['Longitude'])
                dist = geodesic(coord_i, coord_j).km
                if dist <= self.radius_km:
                    neighbours.append((row_j['SCATS'], round(dist, 4)))
            graph[scats_id_i] = neighbours
        return graph

    def get_neighbours(self, scats_id):
        """
        Get list of neighbouring SCATS IDs within the radius.

        Args:
            scats_id (int or str): The SCATS ID to query.

        Returns:
            List of (SCATS ID, distance_km) tuples.
        """
        return self.graph.get(scats_id, [])

    def print_graph_summary(self):
        """Prints how many neighbours each node has."""
        for node, edges in self.graph.items():
            print(f"SCATS {node} has {len(edges)} neighbours within {self.radius_km} km")

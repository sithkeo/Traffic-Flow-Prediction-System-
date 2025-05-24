import math
import heapq
from collections import deque
from itertools import count

# Node class representing a search node in the graph
# state: node ID, parent: where it came from, cost: g(n), heuristic: h(n)
class Node:
    def __init__(self, state, parent=None, cost=0, heuristic=0):
        self.state = state
        self.parent = parent
        self.cost = cost  # g(n): path cost so far
        self.heuristic = heuristic  # h(n): estimate to goal (used in GBFS and A*)

    def total(self):
        return self.cost + self.heuristic  # f(n) = g(n) + h(n) (used in A*)

    def __lt__(self, other):
        # Ensures Node can be compared in a priority queue
        return (self.total(), self.state) < (other.total(), other.state)

# Straight-line distance heuristic (Euclidean) between two nodes
def heuristic(node_a, node_b, coords):
    ax, ay = coords[node_a]
    bx, by = coords[node_b]
    return math.hypot(ax - bx, ay - by)

# Builds the solution path by backtracking from the goal node to origin
def reconstruct_path(node):
    path = []
    while node:
        path.append(str(node.state))  # convert to string for printing
        node = node.parent
    return path[::-1]  # reverse to get path from start → goal

# Breadth-First Search (uninformed)
def bfs(origin, destinations, edges, debug=False):
    # Uses a queue (FIFO) for level-order expansion
    frontier = deque([Node(origin)])
    frontier_states = set([origin])  # to avoid re-adding nodes
    explored = set()
    nodes_created = 1

    while frontier:
        node = frontier.popleft()
        frontier_states.remove(node.state)

        if node.state in explored:
            continue
        explored.add(node.state)

        # print expanded node if debug is enabled
        if debug:
            print(f"[EXPAND] Node {node.state}")

        if node.state in destinations:
            return node, nodes_created

        for neighbor in sorted(edges.get(node.state, {})):
            if neighbor not in explored and neighbor not in frontier_states:
                frontier.append(Node(neighbor, node))
                frontier_states.add(neighbor)
                nodes_created += 1

    return None, nodes_created

# Depth-First Search (uninformed)
def dfs(origin, destinations, edges, debug=False):
    # Fix: Ensure destinations is a set for O(1) lookup performance
    if not isinstance(destinations, set):
        destinations = set(destinations)

    frontier = [Node(origin)]
    frontier_states = set([origin])
    explored = set()
    nodes_created = 1

    while frontier:
        node = frontier.pop()

        # Fix: Use discard to avoid KeyError if state is unexpectedly missing
        frontier_states.discard(node.state)

        # Fix: Skip node if it's already explored (prevents redundant re-expansion)
        if node.state in explored:
            continue

        if debug:
            print(f"[EXPAND] Node {node.state}")

        if node.state in destinations:
            return node, nodes_created

        explored.add(node.state)

        # Fix: Reverse order to preserve intended DFS stack behaviour with sorted neighbours
        for neighbor in sorted(edges.get(node.state, {}), reverse=True):
            if neighbor not in explored and neighbor not in frontier_states:
                frontier.append(Node(neighbor, node))
                frontier_states.add(neighbor)
                nodes_created += 1

    return None, nodes_created


# Greedy Best-First Search (informed but not optimal)
# Expands the node that appears closest to the goal (based on h(n) only)
def gbfs(origin, destinations, edges, coords, debug=False):
    if debug:
        print(f"  [ADD] Node {neighbor} (h={h:.2f})")
    goal = min(destinations, key=lambda d: heuristic(origin, d, coords))
    start = Node(origin, heuristic=heuristic(origin, goal, coords))

    frontier = []
    counter = count()  # tie-breaker to ensure stable expansion order
    heapq.heappush(frontier, (start.heuristic, start.state, next(counter), start))
    frontier_states = set([origin])
    explored = set()
    nodes_created = 1

    while frontier:
        _, _, _, node = heapq.heappop(frontier)

        if node.state in explored:
            continue
        frontier_states.remove(node.state)
        explored.add(node.state)

        # print expanded node if debug is enabled
        if debug:
            print(f"[EXPAND] Node {node.state} (h={node.heuristic:.2f})")

        if node.state in destinations:
            return node, nodes_created

        for neighbor in sorted(edges.get(node.state, {})):
            if neighbor not in explored and neighbor not in frontier_states:
                h = heuristic(neighbor, goal, coords)
                if debug:
                    print(f"  [ADD] Node {neighbor} (h={h:.2f})")
                new_node = Node(neighbor, node, 0, h)  # g(n)=0 because GBFS doesn't use path cost
                heapq.heappush(frontier, (h, neighbor, next(counter), new_node))
                frontier_states.add(neighbor)
                nodes_created += 1

    return None, nodes_created

# A* Search (informed and optimal with admissible heuristic)
# Expands nodes based on f(n) = g(n) + h(n)
def astar(origin, destinations, edges, coords, debug=False):
    # Use set for fast lookup of goal states
    if not isinstance(destinations, set):
        destinations = set(destinations)

    # Priority queue stores (f(n), tie-breaker, Node)
    frontier = []
    start = Node(origin, cost=0, heuristic=0)  # h(n) added per goal below
    heapq.heappush(frontier, (0, 0, start))  # total, tie-breaker, node
    explored = {}  # state -> best g(n) seen
    nodes_created = 1
    counter = 1  # tie-breaker for stable ordering

    while frontier:
        _, _, node = heapq.heappop(frontier)

        if debug:
            print(f"[EXPAND] Node {node.state} (g={node.cost}, h={node.heuristic:.2f}, f={node.total():.2f})")

        if node.state in destinations:
            return node, nodes_created

        # Only expand if new path is better
        if node.state in explored and node.cost >= explored[node.state]:
            continue
        explored[node.state] = node.cost

        for neighbor in sorted(edges.get(node.state, {})):
            new_cost = node.cost + edges[node.state][neighbor]

            # Fix: Use closest goal for *each* neighbour to ensure proper heuristic value
            h = min(heuristic(neighbor, g, coords) for g in destinations)
            new_node = Node(neighbor, node, new_cost, h)

            heapq.heappush(frontier, (new_node.total(), counter, new_node))
            counter += 1
            nodes_created += 1

    return None, nodes_created

#Custom 1
#Djikstra search (uninformed algorithm)
#Only uses actual path cost f = g
def dijkstra(origin, destinations, edges, coords, debug=False):
    start = Node(origin, cost=0)  # Starting node with cost 0
    frontier = []
    heapq.heappush(frontier, (start.cost, start.state, start))
    explored = {}  # Maps node state to the lowest cost found so far for that state
    nodes_created = 1

    while frontier:
        current_cost, current_state, node = heapq.heappop(frontier)

        if debug:
            print(f"[EXPAND] Node {node.state} (cost={node.cost})")

        if node.state in destinations:
            return node, nodes_created

        # Only expand if this path to the node is better than any previous one.
        if node.state in explored and explored[node.state] <= node.cost:
            continue
        explored[node.state] = node.cost

        # Expand successors
        for neighbor in sorted(edges.get(node.state, {})):
            new_cost = node.cost + edges[node.state][neighbor]
            new_node = Node(neighbor, parent=node, cost=new_cost)
            if debug:
                print(f"  [ADD] Node {neighbor} (new cost = {new_cost})")
            heapq.heappush(frontier, (new_cost, neighbor, new_node))
            nodes_created += 1

    return None, nodes_created

# Landmark-Based A* Search (informed, optimal, with landmark-enhanced heuristic)
# ------------------------------------------------------------
# Variation of A* that uses a static landmark to guide the heuristic.
# The heuristic is computed as the minimum of:
#   - Direct Euclidean distance to the goal (standard h(n))
#   - Path through a pre-selected landmark: h(n) = dist(n → landmark) + dist(landmark → goal)
# 
# This hybrid heuristic can encourage exploration through important intermediate points,
# especially in maps with bottlenecks or structured layouts.
# The selected landmark is the node farthest from the origin.
def landmark_astar(origin, destinations, edges, coords, debug=False):
    import heapq
    from itertools import count

    class Node:
        def __init__(self, state, parent=None, cost=0, heuristic=0):
            self.state = state
            self.parent = parent
            self.cost = cost
            self.heuristic = heuristic

        def total(self):
            return self.cost + self.heuristic

        def __lt__(self, other):
            return (self.total(), self.state) < (other.total(), other.state)

    def heuristic_with_landmark(node_id, goal_id, coords, landmark_id):
        # Standard Euclidean to goal + bonus from landmark
        def euclidean(a, b):
            ax, ay = coords[a]
            bx, by = coords[b]
            return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

        h_goal = euclidean(node_id, goal_id)
        h_landmark = euclidean(node_id, landmark_id) + euclidean(landmark_id, goal_id)
        return min(h_goal, h_landmark)  # You can adjust this logic to favor landmark path

    goal = min(destinations, key=lambda d: heuristic(origin, d, coords))
    # Select a landmark: mid node or the furthest from origin
    landmark = max(coords, key=lambda n: heuristic(origin, n, coords) if n != origin else -1)

    start = Node(origin, cost=0, heuristic=heuristic_with_landmark(origin, goal, coords, landmark))
    frontier = []
    heapq.heappush(frontier, (start.total(), start.state, start))
    explored = {}
    nodes_created = 1

    while frontier:
        _, _, node = heapq.heappop(frontier)

        if debug:
            print(f"[EXPAND] Node {node.state} (g={node.cost}, h={node.heuristic:.2f}, f={node.total():.2f})")

        if node.state in destinations:
            return node, nodes_created

        if node.state in explored and explored[node.state] <= node.cost:
            continue
        explored[node.state] = node.cost

        for neighbor in sorted(edges.get(node.state, {})):
            new_cost = node.cost + edges[node.state][neighbor]
            h = heuristic_with_landmark(neighbor, goal, coords, landmark)
            new_node = Node(neighbor, node, new_cost, h)
            heapq.heappush(frontier, (new_node.total(), neighbor, new_node))
            nodes_created += 1

    return None, nodes_created
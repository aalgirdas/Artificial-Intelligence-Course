
from search import *
from notebook import *

country = "Romania"  # "Lithuania" "Romania"  "Russia" "India" "Ukraine"
switch_country_map(country)

romania_problem = GraphProblem(maps.romania_map_start, maps.romania_map_goal, maps.romania_map)

search_method = "breadth_first"  # "random" "breadth_first" "best_first" "uniform" "astar"

if search_method == "random":
    solution = random_search(romania_problem, 1000).solution()
if search_method == "breadth_first":
    solution = breadth_first_graph_search(romania_problem).solution()
if search_method == "best_first":
    solution = best_first_graph_search(romania_problem, lambda n: romania_problem.h(n)).solution()
if search_method == "uniform":
    solution = uniform_cost_search(romania_problem).solution()
if search_method == "astar":
    solution = astar_search(romania_problem).solution()

solution.insert(0, maps.romania_map_start)

print(f'solution size is {len(solution)} and solution :   {solution}')

# node colors, node positions and node label positions
node_colors = {node: 'white' for node in maps.romania_map.locations.keys()}
for item in solution:
    if item in node_colors:
        node_colors[item] = "green"

node_positions = maps.romania_map.locations
node_label_pos = { k:[v[0],v[1]-10]  for k,v in maps.romania_map.locations.items() }
edge_weights = {(k, k2) : v2 for k, v in maps.romania_map.graph_dict.items() for k2, v2 in v.items()}

romania_graph_data = {  'graph_dict' : maps.romania_map.graph_dict,
                        'node_colors': node_colors,
                        'node_positions': node_positions,
                        'node_label_positions': node_label_pos,
                         'edge_weights': edge_weights
                     }
show_map(romania_graph_data)






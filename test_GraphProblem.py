
from search import *



romania_problem = GraphProblem("Arad", "Bucharest", romania_map)
solution = breadth_first_graph_search(romania_problem).solution()
#solution = best_first_graph_search(romania_problem, lambda n: romania_problem.h(n)).solution()

print(solution)


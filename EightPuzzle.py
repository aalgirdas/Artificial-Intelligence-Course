
from search import *


puzzle = EightPuzzle((2, 4, 3, 1, 5, 6, 7, 8, 0))

solution = breadth_first_graph_search(puzzle).solution()

#solution = best_first_graph_search(puzzle, lambda n: puzzle.h(n) ).solution()

print(f'{solution}')
'''

Knuth conjectured that starting with the number 4 or 3, a sequence of square root, floor, and factorial operations can reach any desired positive integer.

floor   sqrt sqrt sqrt sqrt  sqrt ((4!)!)  -> 5
floor   sqrt  ((3!)!)  -> 26


Let's try
'''






from search import *


puzzle = Knuth_conject_4(3,('End' , 26))  # (3,26) is OK

solution = breadth_first_graph_search(puzzle).solution()

print(solution)




from csp import *


# 1. Test min_conflicts

solution = min_conflicts(usa_csp)
print(solution)


eight_queens = NQueensCSP(4)
solution = min_conflicts(eight_queens)
print(solution)


import numpy as np
import matplotlib.pyplot as plt

def plot_solution(solution):
    n = len(solution)
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_xticks(np.arange(0.5, n, 1))
    ax.set_yticks(np.arange(0.5, n, 1))
    ax.set_xticklabels(np.arange(1, n + 1))
    ax.set_yticklabels(np.arange(1, n + 1))
    ax.grid(which='both', color='black', linestyle='-', linewidth=1)
    for row in range(n):
        for col in range(n):
            if solution[row] == col:
                ax.text(col, row, "Q", ha='center', va='center', color='white', fontsize=12)
    ax.invert_yaxis()
    ax.set_title(f"{n}-Queens Solution")
    plt.show()


plot_solution(solution)




# 2. Test backtracking_search

eight_queens = NQueensCSP(12)
solution = backtracking_search(eight_queens)  # doesn't work

print(solution)

solution = backtracking_search(usa_csp, select_unassigned_variable=mrv, order_domain_values=lcv, inference=mac)

print(solution)
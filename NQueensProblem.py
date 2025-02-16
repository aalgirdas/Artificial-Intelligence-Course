
from search import *

number_of_queens = 4

nq_problem = NQueensProblem(number_of_queens)  # >30 min for 30 queens
#solution = recursive_best_first_search(nq_problem).solution()
#solution = breadth_first_graph_search(nq_problem).solution()
solution = best_first_graph_search(nq_problem, lambda n: nq_problem.h(n) ).solution()

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
                ax.text(col, row, "Q", ha='center', va='center', color='red', fontsize=24)
    ax.invert_yaxis()
    plt.show()


plot_solution(solution)






from PIL import Image
import matplotlib.pyplot as plt

def plot_NQueens(solution):
    n = len(solution)
    board = np.array([2 * int((i + j) % 2) for j in range(n) for i in range(n)]).reshape((n, n))
    im = Image.open('images/queen_s.png')
    height = im.size[1]
    im = np.array(im).astype(np.float) / 255
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.set_title('{} Queens'.format(n))
    plt.imshow(board, cmap='binary', interpolation='nearest')
    # NQueensCSP gives a solution as a dictionary
    if isinstance(solution, dict):
        for (k, v) in solution.items():
            newax = fig.add_axes([0.064 + (k * 0.112), 0.062 + ((7 - v) * 0.112), 0.1, 0.1], zorder=1)
            newax.imshow(im)
            newax.axis('off')
    # NQueensProblem gives a solution as a list
    elif isinstance(solution, list):
        for (k, v) in enumerate(solution):
            newax = fig.add_axes([0.064 + (k * 0.112), 0.062 + ((7 - v) * 0.112), 0.1, 0.1], zorder=1)
            newax.imshow(im)
            newax.axis('off')
    fig.tight_layout()
    plt.show()

if number_of_queens == 8:
    plot_NQueens(solution)

'''
while True:
    user_input = input("Do you want graphical picture of NQueensProblem  (y/n)? ")
    if user_input.lower() == 'y':
        plot_NQueens(solution)
        break
    else:
        exit()

'''

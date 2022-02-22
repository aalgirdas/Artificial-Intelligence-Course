
from search import *





nq_problem = NQueensProblem(18)
#solution = recursive_best_first_search(nq_problem).solution()
#solution = breadth_first_graph_search(nq_problem).solution()
solution = best_first_graph_search(nq_problem, lambda n: nq_problem.h(n) ).solution()

print(solution)
exit()


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

plot_NQueens(solution)





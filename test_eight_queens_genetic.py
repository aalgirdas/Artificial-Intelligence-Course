
'''
From:  https://github.com/aimacode/aima-python/blob/master/search.ipynb
'''

from search import *


def fitness(q):
    non_attacking = 0
    for row1 in range(len(q)):
        for row2 in range(row1+1, len(q)):
            col1 = int(q[row1])
            col2 = int(q[row2])
            row_diff = row1 - row2
            col_diff = col1 - col2

            if col1 != col2 and row_diff != col_diff and row_diff != -col_diff:
                non_attacking += 1

    return non_attacking

population = init_population(50, range(8), 8)

solution = genetic_algorithm(population, fitness, f_thres=28, gene_pool=range(8))  # Note that the best score achievable is 28.
print(solution)
print(fitness(solution))

exit(0)


population = init_population(50000, range(8), 8)

max_value = None
max_idx = None

for idx, i in enumerate(population):
    num = fitness(i)
    if (max_value is None or num > max_value):
        max_value = num
        max_idx = idx

print('Maximum value:', max_value, "At index: ", max_idx, "Solution: ", population[max_idx])






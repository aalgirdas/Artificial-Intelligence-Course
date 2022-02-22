
from search import *

target = 'Genetic Algorithm'

# The ASCII values of uppercase characters ranges from 65 to 91
u_case = [chr(x) for x in range(65, 91)]
# The ASCII values of lowercase characters ranges from 97 to 123
l_case = [chr(x) for x in range(97, 123)]

gene_pool = []
gene_pool.extend(u_case) # adds the uppercase list to the gene pool
gene_pool.extend(l_case) # adds the lowercase list to the gene pool
gene_pool.append(' ')    # adds the space character to the gene pool

max_population = 100

mutation_rate = 0.07 # 7%

def fitness_fn(sample):
    # initialize fitness to 0
    fitness = 0
    for i in range(len(sample)):
        # increment fitness by 1 for every matching character
        if sample[i] == target[i]:
            fitness += 1
    return fitness


population = init_population(max_population, gene_pool, len(target))

parents = select(2, population, fitness_fn)

# The recombine function takes two parents as arguments, so we need to unpack the previous variable
child = recombine(*parents)

child = mutate(child, gene_pool, mutation_rate)

population = [mutate(recombine(*select(2, population, fitness_fn)), gene_pool, mutation_rate) for i in range(len(population))]

current_best = max(population, key=fitness_fn)

print(current_best)

current_best_string = ''.join(current_best)
print(current_best_string)

ngen = 1200 # maximum number of generations
# we set the threshold fitness equal to the length of the target phrase
# i.e the algorithm only terminates whne it has got all the characters correct
# or it has completed 'ngen' number of generations
f_thres = len(target)





population = init_population(max_population, gene_pool, len(target))
solution = genetic_algorithm(population, fitness_fn, gene_pool, f_thres, ngen, mutation_rate)

print(solution)















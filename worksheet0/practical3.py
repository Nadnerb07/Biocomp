import random
import matplotlib.pyplot as plt
import copy

fitness_plot = []
best_fitness = []
N = 50
P = 50
MUTATION_RATE = 0.04 # 0.05
MUTATION_STEP = 0.85 # 0.80
MIN = 0.0
MAX = 1.0
offspring = []


class Individual:
    def __init__(self):
        self.gene = [0] * N
        self.fitness = 0


def fitness_function(ind):
    fitness = 0

    for i in range(N):
        fitness += ind.gene[i]

    return fitness


population = []

# Create random population of genes
for i in range(0, P):
    temp_gene = []
    for j in range(0, N):
        temp_gene.append(random.uniform(MIN, MAX))
    new_ind = Individual()
    new_ind.gene = temp_gene.copy()
    population.append(new_ind)

for ind in population:
    ind.fitness = fitness_function(ind)

for z in range(0, 50):
    # SELECTION (Select Parents - Tournament Selection)
    for i in range(0, P):
        parent_1 = random.randint(0, P - 1)
        off_1 = population[parent_1]
        parent_2 = random.randint(0, P - 1)
        off_2 = population[parent_2]
        if off_1.fitness > off_2.fitness:
            offspring.append(off_1)
        else:
            offspring.append(off_2)
    temp_individual = Individual()
    for i in range(0, P, 2):
        for j in range(0, N):
            temp_individual.gene[j] = offspring[i].gene[j]
        crossover_point = random.randint(0, N - 1)
        for j in range(crossover_point, N):
            offspring[i].gene[j] = offspring[i + 1].gene[j]
            offspring[i + 1].gene[j] = temp_individual.gene[j]

    for i in range(0, P):
        new_individual = Individual()
        new_individual.gene = []
        for j in range(0, N):
            gene = offspring[i].gene[j]
            mutation_probability = random.uniform(MIN, MAX)

            if mutation_probability < MUTATION_RATE:
                alter = random.uniform(0, MUTATION_STEP)
                if random.choice([0, 1]) == 1:
                    offspring[i].gene[j] = offspring[i].gene[j] + alter
                    if offspring[i].gene[j] > MAX:
                        offspring[i].gene[j] = MAX
                    else:
                        offspring[i].gene[j] = offspring[i].gene[j] - alter
                        if offspring[i].gene[j] < MIN:
                            offspring[i].gene[j] = MIN

            new_individual.gene.append(gene)
        offspring[i] = new_individual

    offspring_total_fitness = 0
    for individual in offspring:
        individual.fitness = fitness_function(individual)
        # print(individual.gene, individual.fitness)
        offspring_total_fitness += individual.fitness

    best = 0
    bestIndividual = None
    for bestInd in population:
        if bestInd.fitness > best:
            best = bestInd.fitness
            bestIndividual = bestInd
    print('best', best)

    # best_index = offspring.index(best_fitness)
    best_index = population.index(bestIndividual)

    best_fitness.append(max(individual.fitness for individual in population))
    print(best_index)
    # Mean Fitness Plot
    fitness_plot.append(offspring_total_fitness / P)

    # Best Solution
    population = copy.deepcopy(offspring)

    worst = 1000
    worstIndividual = None
    for worstInd in population:
        if worstInd.fitness < worst:
            worst = worstInd.fitness
            worstIndividual = worstInd

    worst_index = population.index(worstIndividual)
    population[worst_index] = bestIndividual
    print('worst index', worst_index)

    offspring.clear()

plt.plot(fitness_plot)
plt.plot(best_fitness)

plt.ylabel('Fitness')
plt.show()

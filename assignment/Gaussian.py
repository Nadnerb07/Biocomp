import random
import matplotlib.pyplot as plt
import copy

N = 20  # Gene Size
P = 400  # Population size
mutations = [0.07]  # Mutation Rate
steps = [1]  # List to hold mutation steps
MIN = -100  # Lower Bound
MAX = 100  # Upper bound
probability = 0.1  # Crossover Probability

"""
class that create a individuals attributes
"""
class Individual:
    def __init__(self):
        self.gene = [0] * N  # initialise gene size
        self.fitness = 0  # initialise fitness


def generate_genes():
    population = []
    # Create random population of genes
    for i in range(0, P):
        temp_gene = []  # List to hold a temp gene
        for j in range(0, N):
            temp_gene.append(random.uniform(MIN, MAX))  # appending a random value from between the bounds
        new_ind = Individual()  # New instance of an individual
        new_ind.gene = temp_gene.copy()
        new_ind.fitness = rosenbrock(new_ind)  # new individual is assigned a fitness value from rosenbrock
        population.append(new_ind)  # new individual gets appended to the population

    return population


def tournament_selection(population):
    offspring = []

    for i in range(0, P):
        parent_1 = random.randint(0, P - 1)  # Generate 1st random integer in population
        off_1 = population[parent_1]  # Get random individual in population
        parent_2 = random.randint(0, P - 1)  # Generate 1st random integer in population
        off_2 = population[parent_2]  # Get random individual in population
        if off_1.fitness > off_2.fitness:  # Compete to get best
            offspring.append(off_2)
        else:
            offspring.append(off_1)

    return offspring  # Winner is returned for crossover


def arithmetic_combination(parent_1, parent_2, cross_prob):
    child_1 = (cross_prob * parent_1) + (1 - cross_prob) * parent_2  # create child 1 from parents
    child_2 = (cross_prob * parent_2) + (1 - cross_prob) * parent_1  # create child 2 from parents

    return child_1, child_2


def simple_arithmetic_combination(offspring, cross_prob):
    for i in range(0, len(offspring), 2):
        parent_1 = offspring[i].gene  # Get offspring gene for parent 1
        parent_2 = offspring[i + 1].gene  # Get offspring gene for parent 1

        cross_point = random.randint(0, N - 1)  # Generate crossover point

        for j in range(cross_point, N):  # Get range of gene to change
            produce_child = arithmetic_combination(parent_1[j], parent_2[j], cross_prob)  # get children
            parent_1[j] = produce_child[0]
            parent_2[j] = produce_child[1]
    return offspring


def mutation(offspring):
    for i in range(0, P):
        new_individual = Individual()  # New instance of individual
        new_individual.gene = []  # Clear genes
        for j in range(0, N):
            gene = offspring[i].gene[j]
            mutation_probability = random.random()  # Get mute prob

            if mutation_probability < MUTATION_RATE:  # condition to check if mutation should occur
                alter = random.uniform(0.0, step)  # 0 1.0
                if random.choice([0, 1]) == 1:
                    offspring[i].gene[j] = offspring[i].gene[j] + alter  # alter gene
                    if offspring[i].gene[j] > MAX:
                        offspring[i].gene[j] = MAX  # alter gene value
                else:
                    offspring[i].gene[j] = offspring[i].gene[j] - alter  # alter gene value
                    if offspring[i].gene[j] < MIN:
                        offspring[i].gene[j] = MIN  # alter gene value

            new_individual.gene.append(gene)
        new_individual.fitness = rosenbrock(new_individual)  # new
        offspring[i] = new_individual
    return offspring


def mutation_gauss(offspring):
    for i in range(0, P):
        new_individual = Individual()  # New instance of individual
        new_individual.gene = []  # Clear genes
        for j in range(0, N):
            gene = offspring[i].gene[j]
            mutation_probability = random.random()  # Get mute prob

            if mutation_probability < MUTATION_RATE:  # condition to check if mutation should occur
                alter = random.gauss(0.0, MUTATION_STEP)  # mu 0.0, sigma mutation step for gaussian distribution
                if random.choice([0, 1]) == 1:
                    offspring[i].gene[j] = offspring[i].gene[j] + alter  # alter gene
                    if offspring[i].gene[j] > MAX:
                        offspring[i].gene[j] = MAX  # alter gene value
                else:
                    offspring[i].gene[j] = offspring[i].gene[j] - alter  # alter gene value
                    if offspring[i].gene[j] < MIN:
                        offspring[i].gene[j] = MIN  # alter gene value

            new_individual.gene.append(gene)
        new_individual.fitness = rosenbrock(new_individual)  # new
        offspring[i] = new_individual
    return offspring


def rosenbrock(individual) -> float:
    # Rosenbrock minimisation
    fitness = 0
    for i in range(1, N - 1):
        fitness += 100 * pow(individual.gene[i + 1] - individual.gene[i] ** 2, 2) + pow(1 - individual.gene[i], 2)
    return fitness


def elitism(population, offspring):
    # Sorts the population in order of fitness
    population.sort(key=lambda individual: individual.fitness, reverse=True)
    # The best individual for minimisation is at the end of list
    bestIndividual = population[-1]
    # Take the offspring and overwrite population with offspring
    new_population = copy.deepcopy(offspring)

    # Sort the population in order of fitness again
    new_population.sort(key=lambda individual: individual.fitness, reverse=True)

    # Take the worst individual in the new population and overwrite it with the best from the old pop
    new_population[0] = bestIndividual

    return new_population


def myGA(population, mutation):
    # Set the amount of generations
    generations = 500
    best_fitness = []
    mean_fitness_plot = []
    # iterate through generations
    for i in range(generations):
        # Get offspring from tournament selection
        offspring = tournament_selection(population)
        # Get crossed over offspring
        offspring_crossover = simple_arithmetic_combination(offspring, probability)
        # Get mutated offspring
        offspring_mutated = mutation(offspring_crossover)
        # Apply elitism
        population = elitism(population, offspring_mutated)

        fitness = []
        for individual in population:
            fitness.append(individual.fitness)
        min_fitness = min(fitness)  # get the best fitness in population for generation

        # Allows to see via command line what the specific fitness are for each step
        if i == generations - 1:
            print("Mutation Rate:", str(MUTATION_RATE), f" | Step Size:{MUTATION_STEP}", " | Fitness:", min_fitness)

        #mean_fitness = (sum(fitness) / P)  # optional get mean, not used

        best_fitness.append(min_fitness)
        # mean_fitness_plot.append(mean_fitness) # not in use

    return best_fitness


test_gauss = [] # list to hold gauss mutation results
for i in range(5): # Amount of runs
    for rate in mutations: # loop through mutation rates
        for step in steps: # loop through mutation steps
            MUTATION_RATE = rate
            MUTATION_STEP = step
            best_fitness_data = myGA(generate_genes(), mutation) # Call GA for uniform mutation operator
            test_gauss = myGA(generate_genes(), mutation_gauss) # Call GA for Gauss mutation operator

plt.plot(best_fitness_data, label=str("Uniform")) # plot uniform mutation results
plt.plot(test_gauss, label='Gauss') # plot gauss mutation results
plt.title("Uniform vs Gauss") # plot title
plt.ylabel('Fitness') # y label
plt.xlabel('Generations')# x label
plt.legend()
plt.show()

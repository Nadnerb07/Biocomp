import random
import matplotlib.pyplot as plt
import copy
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import pi

N = 20 # gene size
P = 200 # population size
mutations = [0.1] # List of mutation rates
MIN = -32.0 # Lower Bound
MAX = 32.0  # Upper bound
steps = [1] # List of mutation steps

"""
class that create a individuals attributes
"""


class Individual:
    def __init__(self):
        self.gene = [0] * N  # initialise gene size
        self.fitness = 0  # initialise fitness


def single_point_crossover(offspring):
    for i in range(0, P, 2):
        off1 = copy.deepcopy(offspring[i]) # Copy two individuals
        off2 = copy.deepcopy(offspring[i + 1])
        temp = copy.deepcopy(offspring[i])
        crossover_point = random.randint(1, N) # generate crossover point in gene
        for j in range(crossover_point, N):
            off1.gene[j] = off2.gene[j]
            off2.gene[j] = temp.gene[j]
        off1.fitness = ackley(off1) # designate fitness
        off2.fitness = ackley(off2)
        offspring[i] = copy.deepcopy(off1)
        offspring[i + 1] = copy.deepcopy(off2)
    return offspring


def generate_genes():
    population = []
    # Create random population of genes
    for i in range(0, P):
        temp_gene = []  # List to hold a temp gene
        for j in range(0, N):
            temp_gene.append(random.uniform(MIN, MAX))  # appending a random value from between the bounds
        new_ind = Individual()  # New instance of an individual
        new_ind.gene = temp_gene.copy()
        new_ind.fitness = ackley(new_ind)  # new individual is assigned a fitness value from rosenbrock
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


def RWS(population):
    # total fitness of initial pop
    total = 0
    for individual in population:
        total += abs(individual.fitness) # abs adapts RWS for negative values

    offspring = []

    for i in range(0, P):
        selection_point = random.uniform(0.0, total) # Generating crossover point
        count_total = 0
        j = 0
        while count_total <= selection_point:
            count_total += abs(population[j].fitness) # keep running total
            j += 1
            if (j == P):
                break
        offspring.append(copy.deepcopy(population[j - 1])) # Add the individual who got selected by the wheel

    return offspring


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
        new_individual.fitness = ackley(new_individual)  # new
        offspring[i] = new_individual
    return offspring


# Ackely
def ackley(individual) -> float:
    # Ackley minimisation function
    fitness = 0
    a = 0
    b = 0
    # Execute loop part of equation
    for i in range(1, N):
        a += (individual.gene[i] ** 2)

        b += (cos(2 * pi * individual.gene[i]))
    # Calculate the first half for easier understanding
    part1 = -20 * exp(-0.2 * sqrt((1 / N) * a))
    # Calculate the 2nd half for easier understanding
    part2 = exp((1 / N) * b)
    # sum of the 2
    fitness = part1 - part2
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


def myGA(population, crossover):
    # Set the amount of generations
    generations = 300
    best_fitness = []
    mean_fitness_plot = []
    for i in range(generations):
        # Get offspring from tournament selection
        offspring = tournament_selection(population)
        # Get crossed over offspring
        offspring_crossover = crossover(offspring)
        # offspring_crossover = single_point_crossover(offspring, cross_prob)
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

        # mean_fitness = (sum(fitness) / P)  # optional get mean, not used

        best_fitness.append(min_fitness)
        # mean_fitness_plot.append(mean_fitness) # not in use

    return best_fitness


test_fit = []
for i in range(1):
    for rate in mutations:  # iterate through mutations
        for step in steps:  # iterate through mutation steps
            MUTATION_RATE = rate
            MUTATION_STEP = step
            best_fitness_data = myGA(generate_genes(), simple_arithmetic_combination) # GA for crossover variation
            test_fit = myGA(generate_genes(), single_point_crossover) # GA for crossover variation

    plt.plot(best_fitness_data, label='Simple Arithmetic')
    plt.plot(test_fit, label='Single Point')
    plt.title("Single vs Simple Arithmetic")
    plt.ylabel('Fitness')
    plt.xlabel('Generations')
    plt.legend()
    plt.show()

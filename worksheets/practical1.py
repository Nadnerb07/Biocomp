import random

N = 10
P = 50
offspring = []
MUTRATE = 0.2


class Individual:
    def __init__(self):
        self.gene = [0] * N
        self.fitness = 0


population = []

# 50
for i in range(0, P):
    tempgene = []
    # 10
    for j in range(0, N):
        tempgene.append(random.randint(0, 1))
    # print(tempgene)
    newind = Individual()
    # copy temp into the new individual gene
    newind.gene = tempgene.copy()
    # then add to population
    population.append(newind)


def test(ind):
    fitness = 0
    for i in range(N):
        if ind.gene[i] == 1:
            fitness += 1
    return fitness


for ind in population:
    ind.fitness = test(ind)

temp = 0
for x in population:
    #print(x.gene, x.fitness)
    temp = temp + x.fitness
print('Initial Population: ' + str(temp))

temp = 0
for i in range(0, P):
    parent1 = random.randint(0, P - 1)
    off1 = population[parent1]
    parent2 = random.randint(0, P - 1)
    off2 = population[parent2]
    if off1.fitness > off2.fitness:
        offspring.append(off1)
    else:
        offspring.append(off2)

ind_temp = Individual()
for i in range(0, P, 2):
    for j in range(0, N):
        ind_temp.gene[j] = offspring[i].gene[j]
    crosspoint = random.randint(0, N - 1)
    for j in range(crosspoint, N):
        offspring[i].gene[j] = offspring[i + 1].gene[j]
        offspring[i + 1].gene[j] = ind_temp.gene[j]

#def crossover(ind1, ind2):
    #crosspoint = random.randint(1, N - 1)
    #ind1.gene[crosspoint], ind2.gene[crosspoint] = ind2.gene[crosspoint:], ind1.gene[crosspoint:]
    #return ind1, ind2


for i in range(0, P):
    newind = Individual()
    newind.gene = []
    for j in range(0, N):
        gene = offspring[i].gene[j]
        mutprob = random.randint(0, 100)
        if mutprob < (100 * MUTRATE):
            if gene == 1:
                gene = 0
            else:
                gene = 1
        newind.gene.append(gene)

for x in offspring:
    temp = temp + x.fitness
print('Offspring Population: ' + str(temp))


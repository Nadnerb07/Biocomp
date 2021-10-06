import random
import matplotlib.pyplot as plt
import csv

N = 10
LOOPS = 10


class Solution:
    def __init__(self, var, util):
        self.var = [0] * N
        self.util = 0


individual = Solution()

# populating the array with 10 random numbers
for j in range(N):
    individual.var[j] = random.randint(0, 100)

print(individual.var)

individual.util = 0


def test_function(ind):
    utility = 0
    for k in range(N):
        utility = utility + ind.var[i]
        return utility


newind = Solution()
for x in range(LOOPS):
    for i in range(N):
        newind.var[i] = individual.var[i]

    change_point = random.randint(0, N - 1)
    newind.var[change_point] = random.randint(0, 100)

    newind.util = test_function(newind)

    if individual.util <= newind.util:
        individual.var[change_point] = newind.var[change_point]
        individual.util = newind.util
    print(individual.util)

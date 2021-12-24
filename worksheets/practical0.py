import random
import matplotlib.pyplot as plt
import csv

N = 10
LOOPS = 10


class Solution:
    def __init__(self):
        self.var = [0] * N
        self.util = 0

    def getUtility(self):
        return sum(self.var)


individual = Solution()

# populating the array with 10 random numbers
for j in range(N):
    individual.var[j] = random.randint(0, 100)
print(str(individual.var) + "Individual")

individual.util = 100


def test_function(ind):
    utility = 0
    for k in range(N):
        utility = utility + ind.var[i]
        return utility


newind = Solution()
for x in range(LOOPS):
    for i in range(N):
        newind.var[i] = individual.var[i]
    print(str(newind.var) + '\n')
    change_point = random.randint(0, N - 1)
    print("Change Point: " + str(change_point))
    newind.var[change_point] = random.randint(0, 100)

    newind.util = test_function(newind)
    print("Utility: " + str(newind.util))
    if individual.util <= newind.util:
        individual.var[change_point] = newind.var[change_point]
        individual.util = newind.util

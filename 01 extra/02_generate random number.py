# creating 200 numbers in a list between 0 and 7
import random
randomnum = []
for i in range(0, 200):
    n = random.randint(0, 7)
    randomnum.append(n)
print(randomnum)

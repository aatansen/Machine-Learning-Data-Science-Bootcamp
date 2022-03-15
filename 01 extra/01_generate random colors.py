import random
randomcolors = []
colours=['blue','brown','green','red','yellow','orange','black','white']
# print(colours[0])
for i in range(0, 200):
    n = random.randint(0, 7)
    randomcolors.append(colours[n])
print(randomcolors)

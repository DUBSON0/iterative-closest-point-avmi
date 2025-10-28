import numpy as np
import matplotlib.pyplot as plt

# Load the data
with open('pointcloud4line.csv', 'r') as file:
    data = file.readlines()

data = [line.strip().split(';') for line in data]
data = [[float(x) for x in line if x != ''] for line in data]

print(len(data[0]))

length = len(data[0])

points=[]

while len(data[0]) > 0:
    # The .pop method removes and returns an item at the given index from a list.
    # Example: list.pop(0) removes and returns the first item.
    points.append([data[0].pop(0), data[0].pop(0), data[0].pop(0)])

print(points[2])
print(len(points))

points = np.array(points) 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:,0], points[:,1], points[:,2])
plt.show()

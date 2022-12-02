# Brandon Gatewood
# CS 445
# Program 3: Fuzzy C Mean

# This is a simple program that implements the fuzzy c means algorithm. The program will run the algorithm r times and record the
# sum square error for each run. Then it will choose the solution with the lowest sum square error to graph onto a 2-d
# plot.

import numpy as np
import csv
import matplotlib.pyplot as plt

# Load data
data = []
f = open('cluster_data.txt', 'r')
read = csv.reader(f)

for row in read:
    r = row[0].split()
    data.append(r)

data = np.array(list(np.float_(data)))

n = data.shape


# FCM class contains the fuzzy c means object, it will initially assign coefficients randomly to each data point for
# being in the clusters. The algorithm is contained in the fit function, which starts off with computing the centroid
# for each cluster, and updating the coefficients/membership grades for being in the cluster. This will loop until the
# max iterations exceeds.
class FCM:
    # Experiment with different k values
    #c = 3
    # c = 5
    c = 10
    max_iter = 300
    n, m = data.shape
    fuzzy = 1.3
    # Assign coefficients randomly to each data point
    membership_grades = np.random.rand(n, c)

    # Runs the k-means algorithm until converges or max iterations exceeded
    def fit(self):
        iteration = 0

        while iteration < self.max_iter:
            # Updates the centroids from 1 - c
            centroids = []
            for i in range(self.c):
                weight_sum = np.power(self.membership_grades[:, i], self.fuzzy).sum()
                cj = []
                for j in range(self.m):
                    numerator = (data[:, j] * np.power(self.membership_grades[:, i], self.fuzzy)).sum()
                    c_val = numerator / weight_sum
                    cj.append(c_val)

                centroids.append(cj)

            # For each data point, compute its coefficients/membership grades for being in the clusters
            denominator = np.zeros(self.n)

            for i in range(self.c):
                distance = (data[:, :] - centroids[i]) ** 2
                distance = np.sum(distance, axis=1)
                distance = np.sqrt(distance)
                denominator = denominator + np.power(1 / distance, 1 / (self.fuzzy - 1))

            for i in range(self.c):
                distance = (data[:, :] - centroids[i]) ** 2
                distance = np.sum(distance, axis=1)
                distance = np.sqrt(distance)
                self.membership_grades[:, i] = np.divide(np.power(1 / distance, 1 / (self.fuzzy - 1)), denominator)

            iteration += 1

        # Calculate sum square error

        for i in range(self.c):
            distance = (data[:, :] - centroids[i]) ** 2
            distance = np.sum(distance, axis=1)
            distance = np.power(self.membership_grades[:, i], self.fuzzy) * distance
            distance = np.sqrt(distance)

        sse = np.sqrt(np.sum(distance))
        return centroids, sse


# Run the algorithm r times and select the solution that gives the lowest sum of squares error
r = 10
centroid_array = []
sse_array = []

for i in range(r):
    fcm = FCM()
    i_centroid, i_sse = fcm.fit()
    centroid_array.append(i_centroid)
    sse_array.append(i_sse)

# print data and centroids
i = np.argmin(sse_array)
print(sse_array[i])
plt.scatter(data[:, 0], data[:, 1], c="red")
for c in i_centroid:
    plt.plot(c[0], c[1], '+', markersize=10)

plt.show()

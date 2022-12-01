# Brandon Gatewood
# CS 445
# Program 3: Fuzzy C Mean

# Load data
data = []
f = open('cluster_data.txt', 'r')
read = csv.reader(f)

for row in read:
    r = row[0].split()
    data.append(r)

data = np.array(list(np.float_(data)))

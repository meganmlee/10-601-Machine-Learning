import sys
import csv
import numpy as np

# Open input file and get last column values
infilename = sys.argv[1]
labels = []
with open(infilename, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)
        labels = [int(row[-1]) for row in reader]

total = len(labels)
ones = np.sum(labels)
zeros = total - ones

# Calculate entropy
entropy = -((ones/total) * np.log2(ones/total)) - ((zeros/total) * np.log2(zeros/total))

# Calculate error rate
error = zeros/total if ones >= zeros else ones/total

# Write to output file
outfilename = sys.argv[2]
with open(outfilename, 'w') as outfile:
        outfile.write("entropy:" + str(entropy) + "\n")
        outfile.write("error:" + str(error) + "\n")
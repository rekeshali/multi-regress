filen = open('results.out','r')
wall = filen.read()
filen.close()

wall = wall.split('\n')
i = 0
err = []
for line in wall[:-1]:
    if i == 0:
        i += 1
        continue
    else:
        line = line.split()
        err.append([line[0], line[3], line[4]])

import numpy as np
err2 = []
for line in err:
    err2.append(float(line[2]))

i = np.argsort(err2)



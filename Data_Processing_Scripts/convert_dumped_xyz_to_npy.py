import numpy as np
import pandas as pd

###
# this file converts the dumped XYZ into the arrays of num_frames x 40 x 3
###

replica = 0
curr = 0

filename = "rand-orient-rep" + str(replica) + "-" + str(curr) + ".prexyz"
print("current replica: ", replica, " and curr: ", curr)

with open(filename) as f:
    content = f.readlines()
content = [x.strip() for x in content]

traj = []
for i in range(len(content)):
    frame = content[i]
    frame = frame.split("} ")
    new_list = [s.replace("{", "").replace("}", "").split(" ") for s in frame]
    arr = np.array(new_list)
    arr = arr.astype(np.float)
    traj.append(arr)

traj = np.array(traj)
outfile = "../processed_orient/rand-orient-rep" + str(replica) + "-" + str(curr) + ""
np.save(outfile , traj)

exit()
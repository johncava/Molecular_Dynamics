import numpy as np

# This is the straightforward approach as outlined in the answers to
# "How do I calculate a dihedral angle given Cartesian coordinates?"
def dihedral2(p):
    b = p[:-1] - p[1:]
    b[0] *= -1
    v = np.array( [ v - (v.dot(b[1])/b[1].dot(b[1])) * b[1] for v in [b[0], b[2]] ] )
    # Normalize vectors
    v /= np.sqrt(np.einsum('...i,...i', v, v)).reshape(-1,1)
    b1 = b[1] / np.linalg.norm(b[1])
    x = np.dot(v[0], v[1])
    m = np.cross(v[0], b1)
    y = np.dot(m, v[1])
    return np.degrees(np.arctan2( y, x ))


def getPhiVals(frame):

    phi_indices = np.array([
    [3, 5, 6, 7], #4
    [3, 5, 6, 7], #8
    [7, 9, 10, 11],  #12
    [11, 13, 14, 15], # 16
    [15, 17, 18, 19], # 20
    [19, 21, 22, 23], # 24
    [23, 25, 26, 27], # 28
    [27, 29, 30, 31], # 32
    [31, 33, 34, 35], # 36
    [35, 39, 40, 37] # 40 ugh 
    ])

    phi_vals = []
    for i in range(len(phi_indices)):
        sel = phi_indices[i] - 1
        chosen_frames = np.stack((frame[sel[0]], frame[sel[1]], frame[sel[2]], frame[sel[3]]))
        phi = dihedral2(chosen_frames)
        phi_vals.append(phi)
        #phi_vals.append(phi)
        #phi_vals.append(phi)
        #phi_vals.append(phi)

    return np.array(phi_vals)


def getPsiVals(frame):

    ## N CA C N
    psi_indices = np.array([
        [1, 2, 3, 5],
        [5, 6, 7, 9],
        [9, 10, 11, 13],
        [13, 14, 15, 17],
        [17, 18, 19, 21],
        [21, 22, 23, 25],
        [25, 26, 27, 29],
        [29, 30, 31, 33],
        [33, 34, 35, 39],
        [33, 34, 35, 39]
    ])

    psi_vals = []
    for i in range(len(psi_indices)):
        sel = psi_indices[i] - 1
        chosen_frames = np.stack((frame[sel[0]], frame[sel[1]], frame[sel[2]], frame[sel[3]]))
        psi = dihedral2(chosen_frames)
        psi_vals.append(psi)
        #psi_vals.append(psi)
        #psi_vals.append(psi)
        #psi_vals.append(psi)

    return np.array(psi_vals)


## returns a list of the length of the number of atoms corresponding to the distances
def getDistFrame(frame):
    distList = np.linalg.norm(np.diff(frame, axis=0), axis=1)
    last = distList[-1]
    distance_for_current_frame = np.append(distList, last)
    return distance_for_current_frame


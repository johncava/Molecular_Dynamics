import os, sys
import torch, argparse
import numpy as np
import glob
import time


def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"


def get_dataset(raw_data, saved_x_dataset, saved_dx_dataset, num_atoms, downsample_num, data_whitened):
    
    if data_whitened == True:
        print("Preparing Whitened Data...")
        num_trajectories = 200
        
        files = glob.glob(raw_data)
        dataset = []
        for file_ in files:
            print("Loading File", file_)
            X_positions = np.load(file_)
            downsampled_X = X_positions[::downsample_num]
            dataset.append(downsampled_X)

        dataset = np.array(dataset)
        dataset = dataset.reshape(num_trajectories, -1, num_atoms*3)
    else:
        print("Preparing unwhitened Data...")
        num_trajectories = 100
        num_atoms = 40

        dataset = []
        for i in range(num_trajectories):
            NUM = str(i).zfill(2)
            fName = "../../../10_deca_alanine/" + NUM + "/backbone.npy"
            print("Loading file", fName)
            rawCoord = np.load(fName)
            downsampled_X = rawCoord[::downsample_num]
            dataset.append(downsampled_X)

        dataset = np.array(dataset) ## shape: (100, 20000, 40, 3)
        dataset = dataset.reshape(num_trajectories, -1, num_atoms*3)


    ### Preprocessing to create the training step

    x_dataset = [] ## vector of position and momentum
    dx_dataset = [] ## vector of change in position and momentum
    ## Getting the velocity of each trajectory
    
    for traj_idx in range(num_trajectories):
        traj = dataset[traj_idx]
    #     traj = dataset[0]
        num_datapoints = traj.shape[0] - 1

        momenta = []
        for i in range(traj.shape[0] - 1):
            dx = traj[i+1] - traj[i]
            momenta.append(dx)

        momenta = np.array(momenta) ## convert to an np array

        ## stack the coordinates so that we have all the positions and all the momentums
        coords = np.stack((traj[:-1], momenta), 1) ## shape: (19999, 2, 120)
        coords = coords.reshape(num_datapoints, 2 * 120) ## shape: (19999, 240)

        ## Now we need to calculate the change in coords (positions and momentums)
        delta_coords = []
        for frame in range(coords.shape[0] - 1):
            dx = coords[frame+1] - coords[frame]
            delta_coords.append(dx)
        delta_coords = np.array(delta_coords)

        ## appending to the dataset
        x_dataset.append(coords[:-1])
        dx_dataset.append(delta_coords)


    x_dataset = np.array(x_dataset) ## shape of (100, 19998, 240)
    dx_dataset = np.array(dx_dataset) ## shape of (100, 19998, 240)

#     x_dataset = x_dataset.reshape(-1, 240) ## shape of (1999800, 240)
#     dx_dataset = dx_dataset.reshape(-1, 240) ## shape of (1999800, 240)

    # x_dataset[0] + dx_dataset[0] == x_dataset[1] ## This is just a sanity check that we can predict the next step

    ## just for quick load don't need this
    np.save(saved_x_dataset, x_dataset)
    np.save(saved_dx_dataset, dx_dataset)
    del dataset
    return x_dataset, dx_dataset


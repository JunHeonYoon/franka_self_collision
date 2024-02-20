from srmt.planning_scene import PlanningScene
import numpy as np
from math import sqrt, pi
import time
import os
import datetime as dt
import sys
import pickle
import argparse

# np.printoptions(precision=3, suppress=True, linewidth=100, threshold=10000)
np.set_printoptions(threshold=sys.maxsize)
title_font = {
    'fontsize': 16,
    'fontweight': 'bold'
}


def main(args):
    # Parameters
    np.random.seed(args.seed)
    joint_limit = np.array([[-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973],  # min 
                            [ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973]]) # max

    
    # Create Planning Scene
    pc = PlanningScene(arm_names=["panda"], arm_dofs=[7], base_link="world")

    dataset = {}

    normal_q_set = []
    nerf_q_set= []
    coll_set  = []
    min_dist_set = []

    t0 = time.time()
    for iter in range(args.num_q):
        q = np.random.uniform(low=joint_limit[0], high=joint_limit[1], size=7)
        pc.display(q)
        min_dist = pc.min_distance(q)
            
        normal_q_set.append( np.array([(q[q_idx] - joint_limit[0, q_idx]) / (joint_limit[1, q_idx] - joint_limit[0, q_idx]) for q_idx in range(7)]) )
        nerf_q_set.append( np.concatenate([q, np.cos(q), np.sin(q)], axis=0) )

        if min_dist == -1: # collide
            coll_set.append(1)
            min_dist_set.append(0.0)
        else:
            coll_set.append(0)
            min_dist_set.append(min_dist)

        if (iter / args.num_q) % 10 == 0 :
            t1 = time.time()
            print("{0:.2f}% of dataset accomplished! (Time: {1:.02f})".format((iter/args.num_q)*100, t1-t0))

    date = dt.datetime.now()
    data_dir = "data/{:04d}_{:02d}_{:02d}_{:02d}_{:02d}_{:02d}".format(date.year, date.month, date.day, date.hour, date.minute,date.second)
    os.mkdir(data_dir)


    with open(data_dir + "/box_grid.pickle", "wb") as f:
        dataset["normalize_q"] = np.array(normal_q_set)
        dataset["nerf_q"] = np.array(nerf_q_set)
        dataset["coll"] = np.array(coll_set)
        dataset["min_dist"] = np.array(min_dist_set)
        pickle.dump(dataset,f)
    with open(data_dir + "/param_setting.txt", "w", encoding='UTF-8') as f:
        params = {"num_q": args.num_q}
        for param, value in params.items():
            f.write(f'{param} : {value}\n')

    import shutil
    folder_path = "data/"
    num_save = 3
    order_list = sorted(os.listdir(folder_path), reverse=True)[1:]
    remove_folder_list = order_list[num_save:]
    for rm_folder in remove_folder_list:
        shutil.rmtree(folder_path+rm_folder)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_q", type=int, default=10000000)

    args = parser.parse_args()
    main(args)


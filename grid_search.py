#!/usr/bin/python3
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os

# scenes = ["smoke"]
# n_init_gaussians_list = [5_000, 20_000, 50_000, 100_000]
# opacity_lrs = [0.05, 0.02]  # 0.02 mit abstand
# position_lrs = [0.00032, 0.00016, 0.00008] # 0.00032 mit wenig gaussians, mit mehr gaussians fast kein unterschied
# feature_lrs = [0.0025, 0.0012] # 0.0025, aber weniger wichtig
# scaling_lrs = [0.010, 0.005, 0.0025, 0.00125] # 0.005-0.0025, groesser, und es wird mit der zeit schlechter, bei kleineren konvergiert es langsamer, aber zuverlaessiger
# rotation_lrs = [0.004, 0.002, 0.001, 0.0005, 0.00025] # 0.00025, groesser, und es konvergieren nicht mehr alle

# scenes = ["burning_ficus", "coloured_wdas", "explosion_1", "explosion_2", "explosion_3", "wdas_cloud_1", "wdas_cloud_2", "wdas_cloud_3", ]
# scenes = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship", ]
# scenes = ["burning_ficus", "coloured_wdas", "explosion_1", "explosion_2", "explosion_3", "wdas_cloud_1", "wdas_cloud_2", "wdas_cloud_3", "chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship", ]
scenes = ["lego", "explosion_3", ]
n_init_gaussians_list = [36_000, ]
opacity_lrs = [0.01, 0.005, 0.0025, 0.00125]  # 0.02 mit abstand
position_lrs = [0.00032, ] # 0.00032 mit wenig gaussians, mit mehr gaussians fast kein unterschied
feature_lrs = [0.0025, ] # 0.0025, aber weniger wichtig
scaling_lrs = [0.005, ] # 0.005-0.0025, groesser, und es wird mit der zeit schlechter, bei kleineren konvergiert es langsamer, aber zuverlaessiger
rotation_lrs = [0.000125, ] # 0.00025, groesser, und es konvergieren nicht mehr alle
formulation = "vol_marcher"


# iteration_args = " --test_iterations 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 --save_iterations 12000 --iterations 12000"
iteration_args = " --test_iterations 30000 --save_iterations 30000 --iterations 30000"

for scene in scenes:
    output_dir = f"./output/grid_search/{scene}"
    common_args = f" --densify_from_iter 100000 --renderer=vol_marcher -s /home/madam/Documents/work/tuw/gaussian_rendering/datasets/nerf_synthetic/{scene}/"
    for n_init_gaussians in n_init_gaussians_list:
        for opacity_lr in opacity_lrs:
            for position_lr in position_lrs:
                for feature_lr in feature_lrs:
                    for scaling_lr in scaling_lrs:
                        for rotation_lr in rotation_lrs:
                            output_path = f"{output_dir}_{formulation}_n{n_init_gaussians}_o{opacity_lr}_p{position_lr}_f{feature_lr}_s{scaling_lr}_r{rotation_lr}"
                            grid_args_1 = f"--n_init_gaussians_for_synthetic {n_init_gaussians} --opacity_lr {opacity_lr} --position_lr_init {position_lr} --position_lr_final {position_lr * 0.01}"
                            grid_args_2 = f"--feature_lr {feature_lr} --scaling_lr {scaling_lr} --rotation_lr {rotation_lr}"
                            command = f"python3 train.py {common_args} {iteration_args} {grid_args_1} {grid_args_2} -m {output_path}"
                            print(f"=========================================\n{command}")
                            os.system(command)

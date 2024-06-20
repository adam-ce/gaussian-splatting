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
from argparse import ArgumentParser

mipnerf360_outdoor_scenes = []
mipnerf360_indoor_scenes = []
tanks_and_temples_scenes = []
deep_blending_scenes = []
vienna_scenes = []
nerf_synthetic_scenes = []

# n_gaussians_list = [4000, 12000, 36000, 108000, 324000, 972000]
n_gaussians_list = [4000, 12000, 36000, 108000, ]
# algorithms = [("sorted_splatter", 0.01, 0), ("sorted_splatter", 0.01, 0), ("inria_splatter", 0.01, 0), ("vol_marcher", 0.001, 3)]
algorithms = [("vol_marcher", 0.001, 3), ]
# algorithms = [("inria_splatter", 0.01, 0), ]

# mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
# mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
# tanks_and_temples_scenes = ["truck", "train"]
# deep_blending_scenes = ["drjohnson", "playroom"]
# vienna_scenes = ["colourlab3", "hohe_veitsch"]
#vienna_scenes = ["insti_roof22"]
# nerf_synthetic_scenes = ["burning_ficus", "coloured_wdas", "explosion_1", "explosion_2", "explosion_3", "wdas_cloud_1", "wdas_cloud_2", "wdas_cloud_3", "chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship", ]
nerf_synthetic_scenes = ["ficus", "hotdog", ] # gataki
# nerf_synthetic_scenes = ["burning_ficus", "coloured_wdas", "explosion_1", "explosion_2", "explosion_3", "wdas_cloud_1", "wdas_cloud_2", "wdas_cloud_3", "chair", "drums", "materials", "mic"] # gs1-10
# nerf_synthetic_scenes = ["materials", "mic"] # king


parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval")
parser.add_argument("--ns_scenes", nargs="+", default=[], help="List of scenes for nerfsynth")
args, _ = parser.parse_known_args()

if args.ns_scenes:
    nerf_synthetic_scenes = args.ns_scenes

all_scenes = []
all_scenes.extend(mipnerf360_outdoor_scenes)
all_scenes.extend(mipnerf360_indoor_scenes)
all_scenes.extend(tanks_and_temples_scenes)
all_scenes.extend(deep_blending_scenes)
all_scenes.extend(vienna_scenes)
all_scenes.extend(nerf_synthetic_scenes)

white_bg_scenes = []
white_bg_scenes.extend(nerf_synthetic_scenes)

if not args.skip_training or not args.skip_rendering:
    if (len(mipnerf360_outdoor_scenes) + len(mipnerf360_indoor_scenes) > 0):
        parser.add_argument('--mipnerf360', "-m360", required=False, type=str, default="../360_v2")
    if (len(tanks_and_temples_scenes) > 0):
        parser.add_argument("--tanksandtemples", "-tat", required=False, type=str, default="../tandt_db/tandt")
    if (len(deep_blending_scenes) > 0):
        parser.add_argument("--deepblending", "-db", required=False, type=str, default="../tandt_db/db")
    if (len(vienna_scenes) > 0):
        parser.add_argument("--tuwien", "-tuw", required=False, type=str, default="../")
    if (len(nerf_synthetic_scenes) > 0):
        parser.add_argument("--nerfsynth", "-ns", required=False, type=str, default="/home/madam/Documents/work/tuw/gaussian_rendering/datasets/nerf_synthetic/")
    args = parser.parse_args()

if not args.skip_training:
    common_args = " --quiet --eval --test_iterations -1  --save_iterations 5000 10000 15000 20000 30000 --iterations 30000 --densify_from_iter 10000000 --position_lr_init 0.00032 --feature_lr 0.0025 --scaling_lr 0.005 --rotation_lr 0.000125"
    for n_gaussians in n_gaussians_list:
        for algorithm, opacity_learning_rate, formulation in algorithms:
                config_args = f" --renderer={algorithm} --opacity_lr {opacity_learning_rate} --formulation={formulation} --n_init_gaussians_for_synthetic {n_gaussians}"
                for scene in mipnerf360_outdoor_scenes:
                    source = args.mipnerf360 + "/" + scene
                    os.system(f"python3 train.py -s {source} -i images_4 -m {args.output_path}/{algorithm}_{n_gaussians}_{scene} {config_args} {common_args}")
                for scene in mipnerf360_indoor_scenes:
                    source = args.mipnerf360 + "/" + scene
                    os.system(f"python3 train.py -s {source} -i images_2 -m {args.output_path}/{algorithm}_{n_gaussians}_{scene} {config_args} {common_args}")
                for scene in tanks_and_temples_scenes:
                    source = args.tanksandtemples + "/" + scene
                    os.system(f"python3 train.py -s {source} -m {args.output_path}/{algorithm}_{n_gaussians}_{scene} {config_args} {common_args}")
                for scene in deep_blending_scenes:
                    source = args.deepblending + "/" + scene
                    os.system(f"python3 train.py -s {source} -m {args.output_path}/{algorithm}_{n_gaussians}_{scene} {config_args} {common_args}")
                for scene in vienna_scenes:
                    source = args.tuwien + "/" + scene
                    os.system(f"python3 train.py -s {source} -m {args.output_path}/{algorithm}_{n_gaussians}_{scene} {config_args} {common_args}")
                for scene in nerf_synthetic_scenes:
                    if n_gaussians < 108000 and (scene == "hotdog" or scene == "ficus"):
                        continue
                    if n_gaussians < 12000 and (scene == "mic"):
                        continue
                    if n_gaussians < 36000 and (scene == "materials"):
                        continue
                    if n_gaussians < 300000 and (scene == "wdas_cloud_1" or scene == "wdas_cloud_2" or scene == "wdas_cloud_3"):
                        continue
                    source = args.nerfsynth + "/" + scene
                    os.system(f"python3 train.py -s {source} -m {args.output_path}/{algorithm}_{n_gaussians}_{scene} {config_args} {common_args}")

if not args.skip_rendering:
    all_sources = []
    for scene in mipnerf360_outdoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in mipnerf360_indoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in tanks_and_temples_scenes:
        all_sources.append(args.tanksandtemples + "/" + scene)
    for scene in deep_blending_scenes:
        all_sources.append(args.deepblending + "/" + scene)
    for scene in vienna_scenes:
        all_sources.append(args.tuwien + "/" + scene)
    for scene in nerf_synthetic_scenes:
        all_sources.append(args.nerfsynth + "/" + scene)

    common_args = " --quiet --eval"
    for scene, source in zip(all_scenes, all_sources):
        for n_gaussians in n_gaussians_list:
            for algorithm, _, _ in algorithms:
                    config_args = f" --renderer={algorithm}"
                    for iter in [5000, 10000, 15000, 20000, 30000]:
                        if scene not in white_bg_scenes:
                            os.system(f"python3 render.py --iteration {iter} -s {source} -m {args.output_path}/{algorithm}_{n_gaussians}_{scene} {config_args} {common_args}")
                        else:
                            os.system(f"python3 render.py --background white --iteration {iter} -s {source} -m {args.output_path}/{algorithm}_{n_gaussians}_{scene} {config_args} {common_args}")

if not args.skip_metrics:
    scenes_string = ""
    for scene in all_scenes:
        for n_gaussians in n_gaussians_list:
            for algorithm, _, _ in algorithms:
                scenes_string += f"\"{args.output_path}/{algorithm}_{n_gaussians}_{scene}\" "

    os.system("python3 metrics.py -m " + scenes_string)

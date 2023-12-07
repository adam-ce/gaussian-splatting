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

mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = ["drjohnson", "playroom"]
vienna_scenes = ["colourlab3", "hohe_veitsch"]
#vienna_scenes = ["insti_roof22"]
#nerf_synthetic_scenes = ["mic", "ship"]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval")
args, _ = parser.parse_known_args()

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
        parser.add_argument("--nerfsynth", "-ns", required=False, type=str, default="../nerf_synthetic")
    args = parser.parse_args()

if not args.skip_training:
    common_args = " --quiet --eval --test_iterations -1 "
    for scene in mipnerf360_outdoor_scenes:
        source = args.mipnerf360 + "/" + scene
        os.system("python3 train.py -s " + source + " -i images_4 -m " + args.output_path + "/" + scene + common_args)
    for scene in mipnerf360_indoor_scenes:
        source = args.mipnerf360 + "/" + scene
        os.system("python3 train.py -s " + source + " -i images_2 -m " + args.output_path + "/" + scene + common_args)
    for scene in tanks_and_temples_scenes:
        source = args.tanksandtemples + "/" + scene
        os.system("python3 train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)
    for scene in deep_blending_scenes:
        source = args.deepblending + "/" + scene
        os.system("python3 train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)
    for scene in vienna_scenes:
        source = args.tuwien + "/" + scene
        os.system("python3 train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)
    for scene in nerf_synthetic_scenes:
        source = args.nerfsynth + "/" + scene
        os.system("python3 train.py --white_background -s " + source + " -m " + args.output_path + "/" + scene + common_args)

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

    common_args = " --quiet --eval --skip_train"
    for scene, source in zip(all_scenes, all_sources):
        if scene not in white_bg_scenes:
            os.system("python3 render.py --iteration 7000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)
            os.system("python3 render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)
        else:
            os.system("python3 render.py --white_background --iteration 7000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)
            os.system("python3 render.py --white_background --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)

if not args.skip_metrics:
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene + "\" "

    os.system("python3 metrics.py -m " + scenes_string)

#!/usr/bin/python3

import os
from argparse import ArgumentParser

mipnerf360_outdoor_scenes = ["bicycle", "garden", "stump"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = ["drjohnson", "playroom"]

all_scenes = []
all_scenes.extend(mipnerf360_outdoor_scenes)
all_scenes.extend(mipnerf360_indoor_scenes)
all_scenes.extend(tanks_and_temples_scenes)
all_scenes.extend(deep_blending_scenes)

scenes_string = ""
for scene in all_scenes:
    scenes_string += f"\"./eval/{scene}\" "

os.system("python3 metrics.py -m " + scenes_string)

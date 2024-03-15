import os
from tensorboard.backend.event_processing import event_accumulator
import numpy as np

scenes = ["burning_ficus", "coloured_wdas", "explosion_1", "explosion_2", "explosion_3", "wdas_cloud_1", "wdas_cloud_2", "wdas_cloud_3", ]
grid_searches = ["grid_search_wdas_cloud_1"]

tb_size_guidance={ # see below regarding this argument
            event_accumulator.COMPRESSED_HISTOGRAMS: 1,
            event_accumulator.IMAGES: 1,
            event_accumulator.AUDIO: 1,
            event_accumulator.SCALARS: 0, # means all
            event_accumulator.HISTOGRAMS: 1,
        }

with open('grid_search_eval_out.txt', 'a') as output_file:
    for scene in scenes:
        grid_search_directory = f"grid_search_{scene}"
        grid_runs = os.listdir(grid_search_directory)
        for run in grid_runs:
            print(f"processing {grid_search_directory}/{run}")
            params = run.split('_')
            run_path = f"{grid_search_directory}/{run}"
            tb_file_paths = [f"{run_path}/{p}" for p in os.listdir(f"{run_path}") if p.startswith("events.out.tfevents")]
            if len(tb_file_paths) == 0:
                print(f"WARNING: run {run_path}/ does not contain a tensorboard file!")
                continue

            if len(tb_file_paths) > 1:
                print(f"WARNING: run {run_path}/ contains multiple tensorboard file (taking first one)!")
            
            tb_file_path = tb_file_paths[0]
            ea = event_accumulator.EventAccumulator(tb_file_path, tb_size_guidance)
            ea.Reload()
            scalars = ea.Scalars('train/loss_viewpoint - psnr')
            PSNRs = [s.value for s in scalars]
            output_file.write(f"{scene}, {params}, {PSNRs}\n")
            output_file.flush()

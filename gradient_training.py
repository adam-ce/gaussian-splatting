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
import typing
import torch
from utils.loss_utils import l1_loss, ssim
import sys
from scene import Scene
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, GradientLearningParams
from gradient_dataset import GradientDb
from gradient_model import make_unet
from torchvision.io import read_image
from torchvision.transforms import RandomCrop
from torch import optim
from torch import Tensor

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def random_cutouts(a: Tensor, b: Tensor, n_cutouts: int, size: int) -> typing.Tuple[Tensor, Tensor]:
    assert a.shape == b.shape
    a = a.reshape(1, 1, *a.shape)
    b = b.reshape(*a.shape)
    ab = torch.cat((a, b), dim=0)
    cropper = RandomCrop(size)
    crops = [cropper(ab) for _ in range(n_cutouts)]
    crops = torch.concat(crops, dim=1)
    return crops[0, ...], crops[1, ...]

def training(dataset: ModelParams, opt: GradientLearningParams):
    tb_writer = prepare_output_and_logger(dataset)
    scene = Scene(dataset, None)
    gradient_db =  GradientDb(dataset.source_path + "/gradient_db/")
    gradient_db.load()

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    train_cameras = scene.getTrainCameras().copy()
    model = make_unet(3, 3).cuda()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    progress_bar = tqdm(range(0, opt.n_epochs * len(gradient_db.entries)), desc="Training progress")
    smoothed_loss = 0
    for epoch in range(0, opt.n_epochs):
        iter_start.record()

        for sample_no, training_sample in enumerate(gradient_db.entries):
            viewpoint_cam = train_cameras[training_sample.cam_id]
            gt_image = viewpoint_cam.original_image.cuda()
            rendering = read_image(training_sample.image_path).cuda().to(torch.float32) / 255.0
            rendering.requires_grad = True

            Ll1 = l1_loss(rendering, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(rendering, gt_image))
            loss.backward()

            gradient_target = rendering.grad * 1000
            gradient_target, rendering = random_cutouts(gradient_target, rendering, 50, 128)
            optimizer.zero_grad()
            gradient_prediction = model(rendering)
            gradient_loss = l1_loss(gradient_target, gradient_prediction)
            gradient_loss.backward()
            optimizer.step()
            scheduler.step(gradient_loss)

            with torch.no_grad():
                # Progress bar
                if smoothed_loss == 0:
                    smoothed_loss = gradient_loss.item()
                smoothed_loss = 0.4 * gradient_loss.item() + 0.6 * smoothed_loss
                if sample_no % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{smoothed_loss:.{4}f}", "grad abs mean": f"{gradient_target.abs().mean():.{4}f}"})
                    progress_bar.update(10)

        iter_end.record()
        if epoch == opt.n_epochs - 1:
            progress_bar.close()

            # # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            # if (iteration in saving_iterations):
            #     print("\n[ITER {}] Saving Gaussians".format(iteration))
            #     scene.save(iteration)

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = GradientLearningParams(parser)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args))

    # All done
    print("\nTraining complete.")

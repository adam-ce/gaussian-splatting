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
from utils.loss_utils import l1_loss, l2_loss, ssim
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
import mlflow
import mlflow.pytorch

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
    scene = Scene(dataset, None)
    gradient_db =  GradientDb(dataset.source_path + "/gradient_db/")
    gradient_db.load()

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    train_cameras = scene.getTrainCameras().copy()
    model = make_unet(3, 3).cuda()
    mlflow.pytorch.log_model(model, "prediction_model")
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
            rendering_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(rendering, gt_image))
            rendering_loss.backward()

            target = rendering.detach()
            target, rendering = random_cutouts(target, rendering, 50, 128)
            optimizer.zero_grad()
            prediction = model(rendering)
            loss = l1_loss(target, prediction)
            loss = (1.0 - opt.lambda_dssim) * loss + opt.lambda_dssim * (1.0 - ssim(prediction, target))
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            with torch.no_grad():
                # Progress bar
                if smoothed_loss == 0:
                    smoothed_loss = loss.item()
                smoothed_loss = 0.4 * loss.item() + 0.6 * smoothed_loss
                mlflow.log_metric("loss", loss.item(), epoch * len(gradient_db.entries) + sample_no)

                if sample_no % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{smoothed_loss:.{4}f}"})
                    progress_bar.update(10)

                if sample_no % 500 == 0:
                    log_target =     target[0:10].transpose(1, 3).transpose( 1, 2).transpose(0, 1).reshape(target.shape[3], -1, 3)
                    log_prediction = prediction[0:10].transpose(1, 3).transpose( 1, 2).transpose(0, 1).reshape(target.shape[3], -1, 3)
                    log_image = torch.cat((log_prediction, log_target), dim=0)
                    mlflow.log_image(log_image.cpu().numpy(), f"prediction_e{epoch:02}_s{sample_no:05}.png")

        iter_end.record()
        if epoch == opt.n_epochs - 1:
            progress_bar.close()

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

    # run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args))

    # All done
    print("\nTraining complete.")

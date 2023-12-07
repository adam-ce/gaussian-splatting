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

import torch
import sys
from scene import GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.sh_utils import RGB2SH
from utils.general_utils import inverse_sigmoid
from gaussian_renderer import render
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from scene.cameras import MiniCam
import numpy as np
import torchvision

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


parser = ArgumentParser(description="Training script parameters")
lp = ModelParams(parser)
op = OptimizationParams(parser)
pp = PipelineParams(parser)
args = parser.parse_args(sys.argv[1:])


# MAKE THE GAUSSIANS
gaussians = GaussianModel(sh_degree=0)
xyz = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.0, 0.1]]).float().cuda()
rgb = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]).float().cuda().unsqueeze(1)
scales = torch.tensor([[0.5, 0.05, 0.5], [1.2, 1.2, 1.2]]).float().cuda()
rots = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]).float().cuda()
opacity = torch.tensor([[0.5], [0.0]]).float().cuda()

scales = torch.log(scales)
# opacity = inverse_sigmoid(opacity)
features_dc = RGB2SH(rgb)


gaussians._xyz = xyz
gaussians._features_dc = features_dc
gaussians._features_rest = torch.zeros((features_dc.shape[0], 0, 3)).float().cuda()
gaussians._scaling = scales
gaussians._rotation = rots
gaussians._opacity = opacity


# MAKE THE CAMERA
R = np.eye(3)
T = np.array([0.0, 0.0, 3.0])
FOV = 0.7
znear = 0.01
zfar = 100.0
world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FOV,
                                        fovY=FOV).transpose(0, 1).cuda()
full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

viewpoint_cam = MiniCam(800, 800, FOV, FOV, znear, zfar, world_view_transform, full_proj_transform)

image = render(viewpoint_cam, gaussians, pp.extract(args),
               bg_color=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"))['render']
torchvision.utils.save_image(image, "./output/simple_test_scene/render.png")
gaussians.save_ply("./output/simple_test_scene/point_cloud/iteration_30000/point_cloud.ply")

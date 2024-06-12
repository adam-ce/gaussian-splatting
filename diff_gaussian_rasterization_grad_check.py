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
import time
from scene import GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.sh_utils import RGB2SH
from utils.general_utils import inverse_sigmoid, inverse_softplus
from gaussian_renderer import render, network_gui
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from scene.cameras import MiniCam
from torch.autograd.gradcheck import gradcheck
import numpy as np
import torchvision

parser = ArgumentParser(description="Training script parameters")
lp = ModelParams(parser)
op = OptimizationParams(parser, lp)
pp = PipelineParams(parser)
args = parser.parse_args(sys.argv[1:])


# MAKE THE GAUSSIANS
r = 0.6
a = 0.9
s = 0.4
xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 1.0]]).float().cuda()
rgb = torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0], [0.0, r, 0.0], [0.0, 0.0, r]]).float().cuda().unsqueeze(1)
scales = torch.tensor([[s*0.8, s*1.1, s*1.2], [s*0.9, s*1.2, s*0.7], [s*0.9, s*1.2, s*1.3], [s*1.4, s*0.8, s*0.7]]).float().cuda()
rots = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]).float().cuda()
opacity = torch.tensor([[a], [a], [a], [a]]).float().cuda()

scales = torch.log(scales)
features_dc = RGB2SH(rgb)
features_rest = torch.zeros((features_dc.shape[0], 0, 3)).float().cuda()


xyz.requires_grad = True
features_dc.requires_grad = True
features_rest.requires_grad = True
scales.requires_grad = True
rots.requires_grad = True
opacity.requires_grad = True


# MAKE THE CAMERA
R = np.eye(3)
T = np.array([0.0, 0.0, 30.0])
FOV = 0.1
znear = 0.1
zfar = 100.0
world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()

projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FOV,
                                        fovY=FOV).transpose(0, 1).cuda()
full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

viewpoint_cam = MiniCam(6, 6, FOV, FOV, znear, zfar, world_view_transform, full_proj_transform)

ppe = pp.extract(args)
# ppe.renderer = "vol_marcher"
background = torch.tensor([0.9, 0.7, 0.4], dtype=torch.float32, device="cuda")
bg_color=background

def render_wrapper(xyz, features_dc, features_rest, scales, rots, opacity):
    gaussians = GaussianModel(lp)
    gaussians._xyz = xyz
    gaussians._features_dc = features_dc
    gaussians._features_rest = features_rest
    gaussians._scaling = scales
    gaussians._rotation = rots
    gaussians._opacity = opacity
    return render(viewpoint_cam, gaussians, ppe, bg_color=background)['render']

image = render_wrapper(xyz, features_dc, features_rest, scales, rots, opacity)
torchvision.utils.save_image(image, "./output/diff_gaussian_rasterization_grad_check_render.png")

# defaults: eps=1e-06, atol=1e-05, rtol=0.001, but we have float
gradcheck(render_wrapper, (xyz, features_dc, features_rest, scales, rots, opacity), eps=1e-04, atol=1e-01, rtol=0.1)

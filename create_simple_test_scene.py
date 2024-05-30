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
from utils.general_utils import inverse_sigmoid
from gaussian_renderer import render, network_gui
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
op = OptimizationParams(parser, lp)
pp = PipelineParams(parser)
args = parser.parse_args(sys.argv[1:])


# MAKE THE GAUSSIANS
gaussians = GaussianModel(lp)
# xyz = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).float().cuda()
# rgb = torch.tensor([[0.9, 0.1, 0.1], [0.0, 1.0, 0.0]]).float().cuda().unsqueeze(1)
# scales = torch.tensor([[0.15, 0.45, 0.25], [0.5, 0.5, 0.5]]).float().cuda()
# rots = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]).float().cuda()
# opacity = torch.tensor([[0.8], [0.8]]).float().cuda()

# xyz = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).float().cuda()
# rgb = torch.tensor([[0.9, 0.1, 0.1], [0.0, 1.0, 0.0]]).float().cuda().unsqueeze(1)
# scales = torch.tensor([[0.15, 0.45, 0.5], [0.5, 0.2, 0.5]]).float().cuda()
# rots = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]).float().cuda()
# opacity = torch.tensor([[0.8], [0.8]]).float().cuda()

# xyz = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]).float().cuda()
# rgb = torch.tensor([[0.2, 0.5, 0.1], [1.0, 0.0, 0.0]]).float().cuda().unsqueeze(1)
# scales = torch.tensor([[0.5, 0.5, .15], [0.5, 0.5, 0.5]]).float().cuda()
# rots = torch.tensor([[ 0, 0.3826834, 0, 0.9238795 ], [ 0, -0.3826834, 0, 0.9238795 ]]).float().cuda()
# opacity = torch.tensor([[1.0], [1.0]]).float().cuda()

xyz = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 1.0]]).float().cuda()
# rot gelb:
# rgb = torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [1.0, 0.8, 0.0], [0.0, 0.0, 1.0]]).float().cuda().unsqueeze(1)
# rot blau: 
rgb = torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.8, 0.0, 0.0], [0.0, 0.0, 1.0]]).float().cuda().unsqueeze(1)
scales = torch.tensor([[0.15, 0.15, .15], [1.3, 0.2, 0.2], [0.2, 1.3, 0.2], [0.15, 0.15, 0.15]]).float().cuda()
rots = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]).float().cuda()
opacity = torch.tensor([[0.0], [0.5], [0.5], [0.0]]).float().cuda()

scales = torch.log(scales)
opacity = inverse_sigmoid(opacity)
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
FOV = 0.1
znear = 1.0
zfar = 100.0
# world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
# world_view_transform = torch.tensor([[-0.7071068,  0.0000000, -0.7071068, -0.0000],
#         [ 0.7071068, -0.0000000, -0.7071068,  0.0000],
#         [ 0.0000000, -1.0000000, -0.0000000, 0.0000],
#         [ 0.0000000,  0.0000000,  40.0000000,  1.0000]], device='cuda:0')
world_view_transform = torch.tensor([[-6.1184e-02, -1.1879e-02,  9.9806e-01,  0.0000e+00],
        [ 1.8715e-03,  9.9993e-01,  1.2016e-02,  0.0000e+00],
        [-9.9813e-01,  2.6030e-03, -6.1156e-02, -0.0000e+00],
        [ 1.0700e+00, -5.7830e-02,  5.9388e+00,  1.0000e+00]], device='cuda:0')


projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FOV,
                                        fovY=FOV).transpose(0, 1).cuda()
full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

viewpoint_cam = MiniCam(800, 800, FOV, FOV, znear, zfar, world_view_transform, full_proj_transform)

ppe = pp.extract(args)
# ppe.renderer = "vol_marcher"
background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
image = render(viewpoint_cam, gaussians, ppe,
               bg_color=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda"))['render']
torchvision.utils.save_image(image, "./output/simple_test_scene/render.png")
gaussians.save_ply("./output/simple_test_scene/point_cloud/iteration_30000/point_cloud.ply")

network_gui.init(network_gui.host, network_gui.port)
keep_alive = True
custom_cam_printed = None
while keep_alive or True:
    if network_gui.conn == None:
        network_gui.try_connect()
    while network_gui.conn != None:
        try:
            net_image_bytes = None
            custom_cam, do_training, _, _, keep_alive, scaling_modifer = network_gui.receive()
            if custom_cam != None:
                net_image = render(custom_cam, gaussians, ppe, background, scaling_modifer)["render"]
                net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
            network_gui.send(net_image_bytes, "/home/madam/Documents/work/tuw/work_of_others/gaussian_splatting/gaussian-splatting/eval/vol_marcher_108000_burning_ficus/")
            if (custom_cam_printed is None) or (torch.any(custom_cam_printed != custom_cam.world_view_transform)):
                custom_cam_printed = custom_cam.world_view_transform
                print(custom_cam.world_view_transform)
        except Exception as e:
            network_gui.conn = None
    time.sleep(1)
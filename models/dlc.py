import torch
from torchvision import models
import pytorch_lightning as pl
from torch import nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from dlcutils.softargmax2d import SoftArgmax2D
from dlcutils.losses import *
from dlcutils.image_helpers import *  # position_and_offset_to_absolute_coordinates, heatmap_tensor_to_image_list, save_overlay_image_and_heatmaps

from models.abstractnetworks import AbstractKeypointNetwork


class ResnetDLC(AbstractKeypointNetwork):
    """
    Basic ResNet50-DeepLabCut.

    Already provides some additional losses, however the offset prediction step seems to be different from the original
    paper.

    """

    def __init__(self, stride=8, replace_stride=None, scoremap_size=(64, 80), eval_root=os.getcwd()):

        # go with 1 replaced layer by default -> this will result in the default stride of 8
        if replace_stride is None:
            replace_stride = [False, False, True]

        def batchnorm2instancenorm(m):
            if isinstance(m, nn.BatchNorm2d):
                m = nn.InstanceNorm2d(m.num_features, affine=False, track_running_stats=False)

        super().__init__()

        self.eval_root = eval_root  # this is where we put all debug output, later our checkpoints etc.

        self.num_joints = 2  # (Nose, Leftear, Rightear, Tailbase)
        self.numResNetFeatures = 2048  # Number of ResNet features that come out at the last ResNet layer. Depends on Network architecture.
        self.deconvolutionstride = 2

        self.stride = stride  # heatmap stride, origimg gets scaled down by this factor to get heatmap dimensions
        self.scoremap_size = scoremap_size  # heatmap size

        # each element in the tuple "replace_stride_with_dilation" indicates if we should replace
        # the 2x2 stride with a dilated convolution with stride 1
        # self.base_model = models.resnet50(pretrained=True, replace_stride_with_dilation=[False, False, True])
        self.base_model = models.resnet50(pretrained=True, replace_stride_with_dilation=replace_stride)

        # replace batchnorm with instance norm for better train -> val -> test coherence
        # BM and IN shoud be equivalend with batch_size=1
        self.base_model.apply(batchnorm2instancenorm)

        # Remove the average pooling and fully connected network in the ResNet architecture
        self.base_layers = list(self.base_model.children())
        self.new_model = nn.Sequential(*self.base_layers[:8])

        # Add the transposed convolution. Kernel size and stride are taken from DeeperCut.
        # Tensorflow implementation uses padding='SAME'.
        # This refers to an input and output padding of 1 in PyTorch.
        self.transpconv = nn.ConvTranspose2d(in_channels=self.numResNetFeatures, out_channels=self.num_joints,
                                             kernel_size=3,
                                             stride=self.deconvolutionstride, padding=1, output_padding=1)
        self.offset_conv = nn.ConvTranspose2d(in_channels=self.numResNetFeatures, out_channels=self.num_joints,
                                              kernel_size=3,
                                              stride=1, padding=0, output_padding=0)
        self.offset_fc = torch.nn.Sequential(
            torch.nn.Linear(10240, 80),
            torch.nn.Linear(80, 4),
            # torch.nn.Tanh()
        )

        self.val_pts_list = []
        self.test_results = []

        self.soft_argmax_layer = SoftArgmax2D(softmax_temp=0.000001)

    def forward(self, x):
        res_out = self.new_model(x)
        out = self.transpconv(res_out)
        offset = self.offset_fc(out.flatten())

        pred_soft_argmax_tensor = self.soft_argmax_layer(out)

        return out, pred_soft_argmax_tensor, offset

    def training_step(self, batch, batch_idx):

        img, gt_heatmaps, gt_key_points_in_heatmap, gt_key_points_in_image = batch

        pred_heatmaps, pred_key_points_in_heatmap, pred_offset = self.forward(img)

        loss = DLC_loss(
            gt_key_points_in_image,
            gt_heatmaps,
            pred_key_points_in_heatmap,
            pred_heatmaps,
            pred_offset
        )

        self.log('train_loss', loss)

        if batch_idx % 100 == 0:
            pred_key_points_in_image = position_and_offset_to_absolute_coordinates(
                pred_key_points_in_heatmap, pred_offset, self.stride
            )

            gt_heatmap_image_list = heatmap_tensor_to_image_list(gt_heatmaps)
            pred_heatmap_image_list = heatmap_tensor_to_image_list(pred_heatmaps)

            save_overlay_image_and_heatmaps(
                img,
                gt_heatmap_image_list,
                pred_heatmap_image_list,
                gt_key_points_in_image,
                pred_key_points_in_image,
                os.path.join(self.eval_root, "train_" + str(batch_idx).zfill(5) + ".png")
            )

        return loss

    def validation_step(self, batch, batch_idx):
        img, gt_heatmaps, gt_key_points_in_heatmap, gt_key_points_in_image = batch

        pred_heatmaps, pred_key_points_in_heatmap, pred_offset = self.forward(img)

        val_loss = DLC_loss(
            gt_key_points_in_image,
            gt_heatmaps,
            pred_key_points_in_heatmap,
            pred_heatmaps,
            pred_offset
        )

        if batch_idx % 5 == 0:
            pred_key_points_in_image = position_and_offset_to_absolute_coordinates(
                pred_key_points_in_heatmap, pred_offset, self.stride
            )

            gt_heatmap_image_list = heatmap_tensor_to_image_list(gt_heatmaps)
            pred_heatmap_image_list = heatmap_tensor_to_image_list(pred_heatmaps)

            save_overlay_image_and_heatmaps(
                img,
                gt_heatmap_image_list,
                pred_heatmap_image_list,
                gt_key_points_in_image,
                pred_key_points_in_image,
                os.path.join(self.eval_root, "val_" + str(batch_idx).zfill(5) + ".png")
            )

        self.log('val_loss', val_loss)

        return val_loss

    def test_step(self, batch, batch_idx):
        # we assume we get the data without any gt
        # also currently works with bs=1 only
        img, filename = batch
        pred_heatmaps, pred_key_points_in_heatmap, pred_offset = self.forward(img)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True, threshold=0.1)

        return {
            "optimizer": optimizer,
            "lr_schedeuler": lr_scheduler,
            "monitor": "val_loss"
        }

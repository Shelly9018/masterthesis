'''
Hourglass network inserted in the pre-activated Resnet
Use lr=0.01 for current version
(c) YANG, Wei
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from dlcutils.image_helpers import *
import cv2
import os
from dlcutils.losses import *
from torch.hub import load_state_dict_from_url
from dlcutils.softargmax2d import SoftArgmax2D
from models.abstractnetworks import AbstractKeypointNetwork
from dlcutils.pcgrad import PCGrad




__all__ = ['HourglassNet', 'hg', 'Bottleneck']


model_urls = {
    'hg1': 'https://github.com/anibali/pytorch-stacked-hourglass/releases/download/v0.0.0/bearpaw_hg1-ce125879.pth',
    'hg2': 'https://github.com/anibali/pytorch-stacked-hourglass/releases/download/v0.0.0/bearpaw_hg2-15e342d9.pth',
    'hg8': 'https://github.com/anibali/pytorch-stacked-hourglass/releases/download/v0.0.0/bearpaw_hg8-90e5d470.pth',
}


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes*block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n-1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n-1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n-1, low1)
        else:
            low2 = self.hg[n-1][3](low1)
        low3 = self.hg[n-1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class HourglassNet(AbstractKeypointNetwork):
    '''Hourglass model from Newell et al ECCV 2016'''
    def __init__(self, block, num_stacks=2, num_blocks=4, num_classes=16, eval_root=os.getcwd(), network_name="BasicSH"):
        super(HourglassNet, self).__init__()

        self.eval_root = eval_root
        self.stride = 4 # sh stride seems to be constant so we init it here

        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        ch = self.num_feats*block.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_feats, 4))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))
            if i < num_stacks-1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1, bias=True))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

        self.soft_argmax_layer = SoftArgmax2D(softmax_temp=0.000001)

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
                conv,
                bn,
                self.relu,
            )

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < self.num_stacks-1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_

        return out[0]

    def training_step(self, batch, batch_idx):

        img, gt_heatmaps, gt_key_points_in_heatmap, gt_key_points_in_image, filename = batch

        pred_heatmaps = self.forward(img)   #generate a prediction

        pred_points_in_heatmap = self.soft_argmax_layer(pred_heatmaps)     # different from the literature

        loss = intermediateSupervision_softArgmax_Heatmap_ceLoss_reductionMean_skeleton(
            pred_heatmaps,
            gt_key_points_in_heatmap,
            gt_heatmaps,
            {}
        )

        #loss = F.binary_cross_entropy_with_logits(pred_heatmaps, gt_heatmaps)

        self.log('train_loss', loss)       # 220-238 visualization

        pred_key_points_in_image = position_and_offset_to_absolute_coordinates(
            pred_points_in_heatmap, torch.zeros_like(pred_points_in_heatmap), self.stride
        )

        if batch_idx % 100 == 0:

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

        img, gt_heatmaps, gt_key_points_in_heatmap, gt_key_points_in_image, filename = batch

        pred_heatmaps = self.forward(img)

        # test time augmentation hat nicht wirklich geholfen, daher by default off
        tta = False

        if tta == True:
            #hm_fliplr = self.forward(img.fliplr()).fliplr()
            #hm_flipud = self.forward(img.flipud()).flipud()
            #hm_doubleflip = self.forward(img.flipud().fliplr()).fliplr().flipud()
            hm_fliplr = self.forward(img.flip(3)).flip(3)
            hm_flipud = self.forward(img.flip(2)).flip(2)
            hm_doubleflip = self.forward(img.flip(2, 3)).flip(3, 2)
            all_heatmaps = (pred_heatmaps + hm_fliplr + hm_flipud + hm_doubleflip) * 0.25

        else:

            all_heatmaps = pred_heatmaps

        pred_points_in_heatmap = self.soft_argmax_layer(all_heatmaps)

        val_loss = intermediateSupervision_softArgmax_Heatmap_ceLoss_reductionMean_skeleton(
            all_heatmaps,
            gt_key_points_in_heatmap,
            gt_heatmaps,
            {}
        )

        self.log('val_loss', val_loss)

        pred_key_points_in_image = position_and_offset_to_absolute_coordinates(
            pred_points_in_heatmap, torch.zeros_like(pred_points_in_heatmap), self.stride
        )

        # works with batch size 0 only as we currently only grab the first filename in the batch
        self.val_keypoint_list_gt.append([filename[0]] + [float(a) for a in gt_key_points_in_image.flatten()])
        self.val_keypoint_list_pred.append([filename[0]] + [float(a) for a in pred_key_points_in_image.flatten()])

        # ab und an was zum drüberschauen speichern
        if batch_idx % 5 == 0:

            gt_heatmap_image_list = heatmap_tensor_to_image_list(gt_heatmaps)
            pred_heatmap_image_list = heatmap_tensor_to_image_list(all_heatmaps)

            save_overlay_image_and_heatmaps(
                img,
                gt_heatmap_image_list,
                pred_heatmap_image_list,
                gt_key_points_in_image,
                pred_key_points_in_image,
                os.path.join(self.eval_root, "val_" + str(batch_idx).zfill(5) + ".png")
            )

        return val_loss

    def test_step(self, batch, batch_idx):

        return self.validation_step(batch, batch_idx)

        # img, gt_heatmaps, gt_key_points_in_heatmap, gt_key_points_in_image, filename = batch
        #
        # pred_heatmaps = self.forward(img)
        #
        # pred_points_in_heatmap = self.soft_argmax_layer(pred_heatmaps)
        #
        # test_loss = intermediateSupervision_softArgmax_Heatmap_ceLoss_reductionMean_skeleton(
        #     pred_heatmaps,
        #     gt_key_points_in_heatmap,
        #     gt_heatmaps,
        #     {}
        # )
        #
        # self.log('test_loss', test_loss)
        #
        # pred_key_points_in_image = position_and_offset_to_absolute_coordinates(
        #     pred_points_in_heatmap, torch.zeros_like(pred_points_in_heatmap), self.stride
        # )
        #
        # if batch_idx % 1 == 0:
        #
        #     gt_heatmap_image_list = heatmap_tensor_to_image_list(gt_heatmaps)
        #     pred_heatmap_image_list = heatmap_tensor_to_image_list(pred_heatmaps)
        #
        #     save_overlay_image_and_heatmaps(
        #         img,
        #         gt_heatmap_image_list,
        #         pred_heatmap_image_list,
        #         gt_key_points_in_image,
        #         pred_key_points_in_image,
        #         os.path.join(self.eval_root, "val_" + str(batch_idx).zfill(5) + ".png")
        #     )
        #
        # return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        pc_adam = PCGrad(optimizer)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True, threshold=0.1)

        return {
            "optimizer": optimizer,
            "lr_schedeuler": lr_scheduler,
            "monitor": "val_loss"
        }

    def on_validation_epoch_start(self) -> None:
        # reset validation coordinates so they don't get stacked all the time for every iteration
        self.val_keypoint_list_gt = []
        self.val_keypoint_list_pred = []

    def on_validation_epoch_end(self) -> None:
        # jede runde überschreiben, mehr hilft mehr :)
        self.export_as_dlc_csv(self.val_keypoint_list_gt, os.path.join(self.eval_root, self.network_name + "_val_kpts_gt.csv"))
        self.export_as_dlc_csv(self.val_keypoint_list_pred, os.path.join(self.eval_root, self.network_name + "_val_kpts_pred.csv"))
        # vorsichtshalber den CP jede epoche speichern solange ich mit lightning checkpointing nicht klarkomme
        self.save_state_dict(os.path.join(self.eval_root + "chkpt_" + str(self.current_epoch).zfill(4) + ".pth"))


def hg(**kwargs):
    model = HourglassNet(Bottleneck, num_stacks=kwargs['num_stacks'], num_blocks=kwargs['num_blocks'],
                         num_classes=kwargs['num_classes'])
    return model


def _hg(arch, pretrained, progress, **kwargs):
    model = hg(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def hg1(pretrained=False, progress=True, num_blocks=1, num_classes=16):
    return _hg('hg1', pretrained, progress, num_stacks=1, num_blocks=num_blocks,
               num_classes=num_classes)


def hg2(pretrained=False, progress=True, num_blocks=1, num_classes=16):
    return _hg('hg2', pretrained, progress, num_stacks=2, num_blocks=num_blocks,
               num_classes=num_classes)


def hg8(pretrained=False, progress=True, num_blocks=1, num_classes=16):
    return _hg('hg8', pretrained, progress, num_stacks=8, num_blocks=num_blocks,
               num_classes=num_classes)
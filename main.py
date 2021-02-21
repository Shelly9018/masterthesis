import sys
# this one was needed for vs code to get cwd right so imports were possible.
# there must be a way to specify this in the project cfg file however
# sys.path.append('/work/scratch/kopaczka/git/ma_khader')

import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import pandas as pd

import cv2
import numpy as np
import math
import glob
from imgaug.augmentables import Keypoint, KeypointsOnImage
# from DeepLabCutLightning.Dataloading.groundtruth_creator import GroundtruthCreator
from dlcutils.augmentation_pipeline import PipelineCreator

import argparse
# import wandb

import matplotlib.pyplot as plt  # just needed for debug output

from imgaug.augmentables import Keypoint, KeypointsOnImage

import imgaug.augmenters as iaa
import configparser


from models.dlc import ResnetDLC
from models.sh import HourglassNet, Bottleneck

from dlcutils.dataloaders import *
from dlcutils.general import WinLinuxPathReplacer

import platform


#  we do config parsing here, maybe there's a better method but this one works just fine

print ("=======================CWD:", os.getcwd())


# general initialization, init train/val/test peocedures
parser = argparse.ArgumentParser()
parser.add_argument("--ini_file")
args = parser.parse_args()

cfg = configparser.ConfigParser()
cfg.read(args.ini_file)

pr = WinLinuxPathReplacer()

ini_network_name = cfg.get("General", "network_name")

train_annotations = pr(cfg.get("Training", "annotation_file"))
train_aug_ini_file = pr(cfg.get("Training", "augmentation_file"))
train_image_root = pr(cfg.get("Training", "image_root"))
train_num_workers = int(cfg.get("Training", "num_workers")) if platform.system() is not "Windows" else 0
train_batch_size = int(cfg.get("Training", "batch_size"))
train_num_epochs = int(cfg.get("Training", "num_epochs"))

val_annotations = pr(cfg.get("Validation", "annotation_file"))
val_aug_ini_file = pr(cfg.get("Validation", "augmentation_file"))
val_image_root = pr(cfg.get("Validation", "image_root"))
eval_root = pr(cfg.get("Validation", "img_save_path"))
val_num_workers = int(cfg.get("Validation", "num_workers")) if platform.system() is not "Windows" else 0
val_batch_size = int(cfg.get("Validation", "batch_size"))

test_annotations = pr(cfg.get("Test", "annotation_file"))
test_aug_ini_file = pr(cfg.get("Test", "augmentation_file"))
test_image_root = pr(cfg.get("Test", "image_root"))
test_num_workers = int(cfg.get("Test", "num_workers")) if platform.system() is not "Windows" else 0
test_batch_size = int(cfg.get("Test", "batch_size"))


# wandb_logger = WandbLogger(name='DLC_01',project='pytorchlightning')

ini_model_type = cfg.get("Model", "type")
model_type_found = False

if ini_model_type == "BasicDLC":

    model_type_found = True

    # model-specific initialization
    ini_stride = int(cfg.get(ini_model_type, "stride"))
    ini_replace_stride = [False if x.strip() == "False" else True for x in
                          cfg.get(ini_model_type, "replace_stride_with_dilation").split(",")]

    ini_scoremap_size = (
        int(cfg.get(ini_model_type, "score_map_size_y")),
        int(cfg.get(ini_model_type, "score_map_size_x"))
    )  # care y first x second (row, col)

    train_dataset = OwnDataset(
        train_annotations,
        train_image_root,
        train_aug_ini_file,
        ini_stride,
        ini_scoremap_size
    )

    val_dataset = OwnDataset(
        val_annotations,
        val_image_root,
        val_aug_ini_file,
        ini_stride,
        ini_scoremap_size
    )

    model = ResnetDLC(
        stride=ini_stride,
        replace_stride=ini_replace_stride,
        scoremap_size=ini_scoremap_size,
        eval_root=eval_root
    )

if ini_model_type == "BasicSH":

    model_type_found = True

    # model-specific initialization
    ini_stride = int(cfg.get(ini_model_type, "stride"))

    ini_scoremap_size = (
        int(cfg.get(ini_model_type, "score_map_size_y")),
        int(cfg.get(ini_model_type, "score_map_size_x"))
    )  # care y first x second (row, col)

    train_dataset = OwnDataset(
        train_annotations,
        train_image_root,
        train_aug_ini_file,
        ini_stride,
        ini_scoremap_size
    )

    val_dataset = OwnDataset(
        val_annotations,
        val_image_root,
        val_aug_ini_file,
        ini_stride,
        ini_scoremap_size
    )

    model = HourglassNet(
        Bottleneck,
        num_stacks=2,
        num_blocks=2,
        num_classes=2,
        eval_root=eval_root,
        network_name=ini_network_name
    )

if not model_type_found:
    print("Terminating before start. Model type not found:", ini_model_type)
    sys.exit()

if cfg.get("General", "run_training") == "True":

    if cfg.get("Training", "model_load_file") != "None":
        checkpoint_file = pr(cfg.get("Training", "model_load_file"))
        if os.path.isfile(checkpoint_file):
            model.load_state_dict(torch.load(checkpoint_file))
            print("checkpoint loaded successfully:", checkpoint_file)

    trainer = pl.Trainer(
        max_epochs=train_num_epochs,
        gpus=1,
        default_root_dir=eval_root
    )  # , logger= wandb_logger)

    trainer.fit(
        model,
        DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=train_batch_size,
            num_workers=train_num_workers
        ),
        DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=val_batch_size,
            num_workers=val_num_workers
        )
    )

    if cfg.get("Training", "model_save_file") != "None":
        torch.save(model.state_dict(), pr(cfg.get("Training", "model_save_file")))

if cfg.get("General", "run_test") == "True":

    print("running test")

    model.load_state_dict(torch.load(pr(cfg.get("Test", "model_load_file"))))

    #video_dataset = ImageFolderDatasetWithoutGT(cfg.get("Test", "image_root"))
    test_dataset = OwnDataset(
        test_annotations,
        test_image_root,
        test_aug_ini_file,
        ini_stride,
        ini_scoremap_size
    )

    # trainer is initialized but all the training-relevant parameters are not relevant
    # as we run test only anyway without any weight updates
    trainer = pl.Trainer(
        max_epochs=train_num_epochs,
        gpus=1,
        default_root_dir=eval_root
    )  # , logger= wandb_logger)

    # trainer.test(
    #     model,
    #     ckpt_path=None,
    #     test_dataloaders=DataLoader(
    #         video_dataset,
    #         shuffle=False,
    #         batch_size=1,
    #         num_workers=1)
    # )
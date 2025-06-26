# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import os
from logging import getLogger
from math import ceil

import h5py
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from decord import VideoReader, cpu
from scipy.spatial.transform import Rotation
import json

_GLOBAL_SEED = 0
logger = getLogger()


def init_data(
    data_path,
    batch_size,
    frames_per_clip=16,
    fps=5,
    crop_size=224,
    rank=0,
    world_size=1,
    camera_views=0,
    stereo_view=False,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    collator=None,
    transform=None,
    camera_frame=False,
    tubelet_size=2,
):
    dataset = DROIDVideoDataset(
        data_path=data_path,
        frames_per_clip=frames_per_clip,
        transform=transform,
        fps=fps,
        camera_views=camera_views,
        frameskip=tubelet_size,
        camera_frame=camera_frame,
    )

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )

    logger.info("VideoDataset unsupervised data loader created")

    return data_loader, dist_sampler


def get_json(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding JSON in file: {filename}")
            except Exception as e:
                print(f"An unexpected error occurred while processing {filename}: {e}")


class DROIDVideoDataset(torch.utils.data.Dataset):
    """Video classification dataset."""

    def __init__(
        self,
        data_path,
        camera_views=["left_mp4_path", "right_mp4_path"],
        frameskip=2,
        frames_per_clip=16,
        fps=5,
        transform=None,
        camera_frame=False,
    ):
        self.data_path = data_path
        self.frames_per_clip = frames_per_clip
        self.frameskip = frameskip
        self.fps = fps
        self.transform = transform
        self.camera_frame = camera_frame
        if VideoReader is None:
            raise ImportError('Unable to import "decord" which is required to read videos.')

        # Camera views
        # ---
        # wrist camera view
        # left camera view
        # right camera view
        self.camera_views = camera_views
        self.h5_name = "trajectory.h5"

        samples = []
        with open(data_path, "r") as f:
            for line in f:
                samples.append(json.loads(line))
        self.samples = samples

    def __getitem__(self, index):
        loaded_sample = self.samples[index]

        # -- keep trying to load videos until you find a valid sample
        loaded_video = False
        while not loaded_video:
            try:
                buffer, actions, states, extrinsics, indices = self.loadvideo_decord(loaded_sample)
                loaded_video = True
            except Exception as e:
                logger.info(f"Encountered exception when loading video {loaded_sample=} {e=}")
                loaded_video = False
                index = np.random.randint(self.__len__())
                loaded_sample = self.samples[index]

        return buffer, actions, states, extrinsics, indices

    def poses_to_diffs(self, poses):
        xyz = poses[:, :3]  # shape [T, 3]
        thetas = poses[:, 3:6]  # euler angles, shape [T, 3]
        matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas]
        xyz_diff = xyz[1:] - xyz[:-1]
        angle_diff = [matrices[t + 1] @ matrices[t].T for t in range(len(matrices) - 1)]
        angle_diff = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff]
        angle_diff = np.stack([d for d in angle_diff], axis=0)
        closedness = poses[:, -1:]
        closedness_delta = closedness[1:] - closedness[:-1]
        return np.concatenate([xyz_diff, angle_diff, closedness_delta], axis=1)

    def loadvideo_decord(self, loaded_sample):
        # -- load metadata
        metadata = loaded_sample
        states = np.array(loaded_sample["robot_states"])
        vpath = "/mnt/weka/home/yi.gu/world-model/evaluation/bridge/output_video0622/" + metadata["video"]
        vr = VideoReader(vpath, num_threads=-1, ctx=cpu(0))
        # --
        vfps = vr.get_avg_fps()
        fpc = self.frames_per_clip
        fps = self.fps if self.fps is not None else vfps
        fstp = ceil(vfps / fps)
        nframes = int(fpc * fstp)
        vlen = len(vr)

        if vlen < nframes:
            raise Exception(f"Video is too short {vpath=}, {nframes=}, {vlen=}")

        # sample a random window of nframes
        ef = np.random.randint(nframes, vlen)
        sf = ef - nframes
        indices = np.arange(sf, sf + nframes, fstp).astype(np.int64)
        # --
        states = states[indices, :]
        actions = self.poses_to_diffs(states)
        # --
        vr.seek(0)  # go to start of video before sampling frames
        buffer = vr.get_batch(indices).asnumpy()
        if self.transform is not None:
            buffer = self.transform(buffer)

        return buffer, actions, states, 0, indices

    def __len__(self):
        return len(self.samples)

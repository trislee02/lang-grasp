import glob
import os
import re

import pickle
import torch

from utils.dataset_processing import grasp, image, mask
from .grasp_data import GraspDatasetBase


class GraspAnythingDataset(GraspDatasetBase):
    """
    Dataset wrapper for the Grasp-Anything dataset.
    """

    def __init__(self, file_path, ds_rotate=0, **kwargs):
        """
        :param file_path: Grasp-Anything Dataset directory.
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(GraspAnythingDataset, self).__init__(**kwargs)

        if kwargs["seen"]:
            file_path = os.path.join(file_path, 'seen')
        else:
            file_path = os.path.join(file_path, 'unseen')
        
        self.grasp_files = glob.glob(os.path.join(file_path, 'grasp_label', '*.pt'))
        self.prompt_files = glob.glob(os.path.join(file_path, 'grasp_instructions', '*.pkl'))
        self.rgb_files = glob.glob(os.path.join(file_path, 'image', '*.jpg'))

        self.grasp_files.sort()
        self.prompt_files.sort()
        self.rgb_files.sort()

        self.length = len(self.grasp_files)

        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            self.grasp_files = self.grasp_files[int(self.length * ds_rotate):] + self.grasp_files[
                                                                                 :int(self.length * ds_rotate)]
            

    def _get_crop_attrs(self, idx):
        gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx])
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 416 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 416 - self.output_size))
        return center, left, top

    def get_gtbb(self, idx, rot=0, zoom=1.0):       
        # Jacquard try
        gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx], scale=self.output_size / 416.0)

        c = self.output_size // 2
        gtbbs.rotate(rot, (c, c))
        gtbbs.zoom(zoom, (c, c))

        # Cornell try
        # gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx])
        # center, left, top = self._get_crop_attrs(idx)
        # gtbbs.rotate(rot, center)
        # gtbbs.offset((-top, -left))
        # gtbbs.zoom(zoom, (self.output_size // 2, self.output_size // 2))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        return None

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_file = re.sub(r"_\d{1}_\d{1}\.pt", ".jpg", self.grasp_files[idx])
        rgb_file = rgb_file.replace("grasp_label", "image")
        rgb_img = image.Image.from_file(rgb_file)

        # Jacquard try
        rgb_img.rotate(rot)
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img

        # Cornell try
        # center, left, top = self._get_crop_attrs(idx)
        # rgb_img.rotate(rot, center)
        # rgb_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        # rgb_img.zoom(zoom)
        # rgb_img.resize((self.output_size, self.output_size))
        # if normalise:
        #     rgb_img.normalise()
        #     rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        # return rgb_img.img

    def get_prompt(self, idx):
        with open(self.prompt_files[idx], 'rb') as f:
            prompt = pickle.load(f)
            if not isinstance(prompt, str):
                raise TypeError(f"Prompt must be a string. Type: {type(prompt)}")
        return prompt
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
import random
import json
from avs_3d.echosegnet.audio_informed_gaussian_splatting.utils.system_utils import searchForMaxIteration
from avs_3d.echosegnet.audio_informed_gaussian_splatting.scene.dataset_readers import sceneLoadTypeCallbacks
from avs_3d.echosegnet.audio_informed_gaussian_splatting.scene.gaussian_model import GaussianModel
from avs_3d.echosegnet.audio_informed_gaussian_splatting.arguments import ModelParams
from avs_3d.echosegnet.audio_informed_gaussian_splatting.utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)

        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)

        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())

            self.train_json_cams, self.test_json_cams, self.json_cams = [], [], []
            train_camlist, test_camlist, camlist = [], [], []

            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
                test_camlist.extend(scene_info.test_cameras)

            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
                train_camlist.extend(scene_info.train_cameras)

            for id, cam in enumerate(camlist):
                self.json_cams.append(camera_to_JSON(id, cam))

            for id, cam in enumerate(train_camlist):
                self.train_json_cams.append(camera_to_JSON(id, cam))

            for id, cam in enumerate(test_camlist):
                self.test_json_cams.append(camera_to_JSON(id, cam))

            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(self.json_cams, file)

            with open(os.path.join(self.model_path, "train_cameras.json"), 'w') as file:
                json.dump(self.train_json_cams, file)
            
            with open(os.path.join(self.model_path, "test_cameras.json"), 'w') as file:
                json.dump(self.test_json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(self.model_path,
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                    "point_cloud.ply"
                )
            )
            
            with open(os.path.join(self.model_path, "cameras.json"), 'r') as file:
                 self.json_cams = json.load(file)

            with open(os.path.join(self.model_path, "train_cameras.json"), 'r') as file:
                 self.train_json_cams = json.load(file)

            with open(os.path.join(self.model_path, "test_cameras.json"), 'r') as file:
                 self.test_json_cams = json.load(file)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

        self.scene_audio_train_test = sceneLoadTypeCallbacks["Audio"](os.path.join(args.source_path, 'audio_clips_binaural'))
        self.scene_audio = {}

        for c in self.train_json_cams:
            self.scene_audio[str(int(c['img_name'].split('_')[-1]))] = self.scene_audio_train_test[str(int(c['img_name'].split('_')[-1]))]

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))

        self.gaussians.save_ply(
            os.path.join(point_cloud_path, "point_cloud.ply"), 
            self.scene_audio,
            self.train_json_cams, 
            os.path.join(self.model_path, 'audio_intensity_map_visualisation')
        )

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
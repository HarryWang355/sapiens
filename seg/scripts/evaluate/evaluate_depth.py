# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from mmseg.apis import inference_model, init_model
import os
from tqdm import tqdm
import cv2
import numpy as np
import torchvision
import glob
torchvision.disable_beta_transforms_warning()

import sys

def main():
    parser = ArgumentParser()
    parser.add_argument('--config', help='Config file', default='/data1/users/yuanhao/sapiens/seg/scripts/evaluate/config.py')
    parser.add_argument('--checkpoint', help='Checkpoint file', default='/data1/users/yuanhao/sapiens/seg/Outputs/train/depth_general/sapiens_1b_depth_general-1024x768/node/10-18-2024_22:44:24/epoch_100.pth')
    parser.add_argument('--input', help='Input image dir', default='/data1/datasets/garment-data/iter3-ele0/sapiens-depth-4views/images')
    parser.add_argument('--device', default='cuda:4', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    input = args.input
    image_names = glob.glob(f'{input}/*.png')
    image_names.sort()

    # os.system(f'rm -rf {input.replace("images", "output_depth1")}')
    os.makedirs(input.replace('images', 'output_depth1'), exist_ok=True)

    d_max = 1.50
    d_min = 0.85
    offset = d_min
    scale = d_max - d_min

    for image_path in tqdm(image_names):
        image = cv2.imread(image_path) ## has to be bgr image
        mask_path = image_path.replace('images', 'masks')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(bool)
        cond_depth = np.load(image_path.replace('images', 'depth_conds').replace('.png', '.npy'))
        cond_mask = cv2.imread(image_path.replace('images', 'masks_conds'), cv2.IMREAD_GRAYSCALE).astype(bool)
        cond_mask = np.logical_and(cond_mask, mask)
        
        # process gt
        gt_depth_map = np.load(image_path.replace('images', 'depths').replace('.png', '.npy'))

        gt_depth_map = (gt_depth_map - offset) / scale
        gt_depth_map[~mask] = 0

        # cond_depth[cond_mask] = (cond_depth[cond_mask] - np.min(cond_depth[cond_mask])) / (np.max(cond_depth[cond_mask]) - np.min(cond_depth[cond_mask]))
        cond_depth[cond_mask] = (cond_depth[cond_mask] - offset) / scale
        cond_depth[~cond_mask] = -1
        cond_depth = np.ones_like(cond_depth) * -1

        image = np.concatenate([image, cond_depth.reshape(1024, 1024, 1)], axis=2)

        result = inference_model(model, image)
        result = result.pred_depth_map.data.cpu().numpy()
        depth_map = result[0] ## H x W

        diff = np.abs(depth_map[cond_mask] - gt_depth_map[cond_mask]).mean()
        overall_diff = np.abs(depth_map[mask] - gt_depth_map[mask]).mean()
        print(f'{os.path.basename(image_path)}: {diff}')
        print(f'{os.path.basename(image_path)}: {overall_diff}')

        image_flipped = cv2.flip(image, 1)
        result_flipped = inference_model(model, image_flipped)
        result_flipped = result_flipped.pred_depth_map.data.cpu().numpy()
        depth_map_flipped = result_flipped[0]
        depth_map_flipped = cv2.flip(depth_map_flipped, 1) ## H x W, flip back
        depth_map = (depth_map + depth_map_flipped) / 2 ## H x W, average

        ##-----------save depth_map to disk---------------------
        # save_path = image_path.replace('images', 'output_depth1').replace('.png', '.npy')
        # np.save(save_path, depth_map)
        # depth_map[~mask] = np.nan
        # depth_map = (depth_map - np.nanmin(depth_map)) / (np.nanmax(depth_map) - np.nanmin(depth_map))
        depth_map[~mask] = 0
        depth_map = np.clip(depth_map, 0, 1)
        depth_map = (depth_map * 255).astype(np.uint8)

        # gt_depth_map and depth map are (1024, 1024) and image is (1024, 1024, 4)
        # concatenate all to save as a single image with depths in grayscale
        save_path = image_path.replace('images', 'output_depth1')

        depth_map = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)
        gt_depth_map = np.clip(gt_depth_map, 0, 1)
        gt_depth_map = (gt_depth_map * 255).astype(np.uint8)
        gt_depth_map = cv2.cvtColor(gt_depth_map, cv2.COLOR_GRAY2BGR)

        cond_depth[~cond_mask] = 0
        cond_depth = (cond_depth * 255).astype(np.uint8)
        cond_depth = cv2.cvtColor(cond_depth, cv2.COLOR_GRAY2BGR)
        
        cv2.imwrite(save_path, np.concatenate([image[:, :, :3], depth_map, cond_depth, gt_depth_map], axis=1))
        ##----------------------------------------
        # depth_foreground = depth_map[mask] ## value in range [0, 1]
        # processed_depth = np.full((mask.shape[0], mask.shape[1], 3), 100, dtype=np.uint8)

        # if len(depth_foreground) > 0:
        #     min_val, max_val = np.min(depth_foreground), np.max(depth_foreground)
        #     depth_normalized_foreground = 1 - ((depth_foreground - min_val) / (max_val - min_val)) ## for visualization, foreground is 1 (white), background is 0 (black)
        #     depth_normalized_foreground = (depth_normalized_foreground * 255.0).astype(np.uint8)

        #     print('{}, min_depth:{}, max_depth:{}'.format(os.path.basename(image_path), min_val, max_val))

        #     depth_colored_foreground = cv2.applyColorMap(depth_normalized_foreground, cv2.COLORMAP_INFERNO)
        #     depth_colored_foreground = depth_colored_foreground.reshape(-1, 3)
        #     processed_depth[mask] = depth_colored_foreground

        

        ##----------------------------------------------------
        # output_file = image_path.replace('images', 'output_vis')

        # vis_image = np.concatenate([image[:, :, :3], processed_depth, normal_from_depth], axis=1)
        # cv2.imwrite(output_file, vis_image)

if __name__ == '__main__':
    main()
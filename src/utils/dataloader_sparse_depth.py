import torch
import random
import numpy as np
import cv2

# simulate a sparse dtof depth map from a dense depth map
def get_sparse_depth(depth, rgb):
    c, h, w = depth.shape
    zone_num = [30, 40]
    scale = np.tan(70 / 360 * np.pi) / np.tan(85 / 360 * np.pi)
    offset_y, offset_x = h - scale * h, w - scale * w
    st_y, st_x = int(np.random.rand() * offset_y / 2), int(np.random.rand() * offset_x / 2)
    dist_coef = np.random.rand() * 6e-4 + 3e-4
    dist_coef = 3e-4
    noise = np.random.rand() * 0.3

    x_u = np.linspace(st_x, w - st_x, zone_num[1])[None].repeat(zone_num[0], axis=0).ravel()
    y_u = np.linspace(st_y, h - st_y, zone_num[0])[..., None].repeat(zone_num[1], axis=1).ravel()

    x_c = w // 2 + np.random.rand() * 50 - 25
    y_c = h // 2 + np.random.rand() * 50 - 25
    x_u = x_u - x_c
    y_u = y_u - y_c

    r_u = np.sqrt(x_u ** 2 + y_u ** 2)
    r_d = r_u - dist_coef * r_u ** 2
    num_d = r_d.size
    cos_theta = x_u / r_u
    sin_theta = y_u / r_u

    x_d = np.round(r_d * cos_theta + x_c + np.random.normal(0, noise, num_d))
    y_d = np.round(r_d * sin_theta + y_c + np.random.normal(0, noise, num_d))

    idx_mask = (x_d < w) & (x_d > 0) & (y_d < h) & (y_d > 0)
    x_d = x_d[idx_mask].astype('int')
    y_d = y_d[idx_mask].astype('int')

    sparse_depth = np.zeros_like(depth)
    sparse_depth[:, y_d, x_d] = depth[:, y_d, x_d]

    large_random_map, small_random_map = np.random.rand(1, h, w), np.random.rand(1, h, w)
    binomial_map = np.random.binomial(1, 0.5, h * w).reshape(1, h, w)
    mask = ((sparse_depth > 8.1) & (large_random_map > 0.1)) | (
                (sparse_depth < 8.1) & (binomial_map > 0) & (small_random_map < 0.1))
    sparse_depth[mask] = 0

    albedo = rgb.mean(axis=0, keepdims=True).numpy()
    binomial_map = np.random.binomial(1, 0.5, h * w).reshape(1, h, w)
    mask = ((albedo < 0.5) & (binomial_map > 0))
    sparse_depth[mask] = 0

    max_depth = 10
    error_random_map = np.random.rand(1, h, w)
    error_val_map = np.random.rand(1, h, w) * max_depth
    mask = (sparse_depth > 0) & (error_val_map < 0.01)
    sparse_depth[mask] = error_val_map[mask]

    return torch.from_numpy(sparse_depth)

def get_grid_coordinate():
    center_H, center_W = torch.meshgrid(torch.arange(40, device='cpu'), torch.arange(30, device='cpu'), indexing="xy", )
    center_index = torch.cat((center_W.unsqueeze(2), center_H.unsqueeze(2)), dim=2)
    center_index = (16 * center_index + 8)
    return center_index.view(-1, 2).numpy()

def get_rotate_translate(image):
    # image = h,w,c
    image = image.squeeze()
    height, width = image.shape[:2]

    # define rotation and translation parameters
    min_rotation_angle = -0.9
    max_rotation_angle = 0.9
    min_translation_offset = -12
    max_translation_offset = 12

    # random rotate
    rotation_angle = random.uniform(min_rotation_angle, max_rotation_angle)
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    # random translation
    translation_offset_x = random.uniform(min_translation_offset, max_translation_offset)
    translation_offset_y = random.uniform(min_translation_offset, max_translation_offset)
    translation_matrix = np.float32([[1, 0, translation_offset_x], [0, 1, translation_offset_y]])
    translated_image = cv2.warpAffine(rotated_image, translation_matrix, (width, height))
    return np.expand_dims(translated_image,-1)


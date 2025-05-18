import os
import math
import json
import glob
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from scipy.spatial.transform import Rotation as R
from bpc.inference.utils.camera_utils import load_camera_params

# Make sure to set the OpenGL platform before importing pyrender.
os.environ["PYOPENGL_PLATFORM"] = "egl"
#import pyrender


def compute_2d_center(K, R_mat, t):
    """
    Projects the 3D translation t (of shape (3,1)) into 2D via K.
    Returns (u, v) or None if behind the camera.
    """
    if t[2, 0] <= 0:
        return None
    uv = K @ t
    if uv[2, 0] == 0:
        return None
    uv /= uv[2, 0]
    return uv[0, 0], uv[1, 0]


def letterbox_preserving_aspect_ratio(img, target_size=256, fill_color=(255, 255, 255)):
    h, w = img.shape[:2]
    scale = float(target_size) / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_size, target_size, 3), fill_color, dtype=np.uint8)
    dx = (target_size - new_w) // 2
    dy = (target_size - new_h) // 2
    canvas[dy:dy + new_h, dx:dx + new_w] = resized
    return canvas, scale, dx, dy


def matrix_to_euler_xyz(R_mat):
    """
    Convert a 3x3 rotation matrix to Euler angles (x, y, z) in radians,
    assuming the rotation matrix was built via: R = Rx(x) * Ry(y) * Rz(z).
    """
    sy = R_mat[0, 2]
    eps = 1e-7
    if abs(abs(sy) - 1.0) > eps:
        y = math.asin(sy)
        x = math.atan2(-R_mat[1, 2], R_mat[2, 2])
        z = math.atan2(-R_mat[0, 1], R_mat[0, 0])
    else:
        y = math.pi / 2 if sy > 0 else -math.pi / 2
        x = math.atan2(R_mat[2, 1], R_mat[1, 1])
        z = 0.0
    return (x, y, z)


def euler_to_quat(euler_angles):
    """
    Convert Euler angles (Rx, Ry, Rz) in radians to quaternion [x, y, z, w].
    """
    r = R.from_euler('xyz', euler_angles, degrees=False)
    return r.as_quat()


def euler_to_6d(euler_angles):
    """
    Convert Euler angles to a 6D rotation representation.
    Compute the rotation matrix and take its first two columns flattened.
    """
    r = R.from_euler('xyz', euler_angles, degrees=False)
    R_mat = r.as_matrix()  # shape (3,3)
    return np.concatenate([R_mat[:, 0], R_mat[:, 1]])


class BOPSingleObjDataset(Dataset):
    """
    A dataset for a single object ID from BOP data.
    
    For training (split == "train" and augment==True), each __getitem__ returns a tuple:
        (orig_img_t, aug_img_t, label_dict, meta)
    """
    def __init__(self,
                 root_dir,
                 scene_ids,
                 cam_ids,
                 target_obj_id,
                 target_size=256,
                 augment=False,
                 split="train",
                 max_per_scene=None,
                 train_ratio=0.8,  # Default split ratio: 80% train, 20% val.
                 seed=42,
                 use_real_val=False  # New flag: if True, try to use the real validation dataset.
                 ):
        super().__init__()
        self.root_dir = root_dir
        self.scene_ids = scene_ids
        self.cam_ids = cam_ids
        self.obj_id = target_obj_id
        self.target_size = target_size
        self.augment = augment
        self.split = split.lower()  # "train" or "val"
        self.max_per_scene = max_per_scene
        self.train_ratio = train_ratio
        self.samples = []
        random.seed(seed)

        # Choose dataset path based on split and flag.
        if self.split == "val" and use_real_val:
            real_val_path = os.path.join(root_dir, "val")
            if os.path.exists(real_val_path):
                dataset_path = real_val_path
            else:
                print(f"[WARNING] Real validation directory {real_val_path} not found. Falling back to train_pbr.")
                train_pbr_path = os.path.join(root_dir, "train_pbr")
                if os.path.exists(train_pbr_path):
                    dataset_path = train_pbr_path
                else:
                    raise FileNotFoundError(f"Directory '{train_pbr_path}' not found in {root_dir}")
        else:
            train_pbr_path = os.path.join(root_dir, "train_pbr")
            if os.path.exists(train_pbr_path):
                dataset_path = train_pbr_path
            else:
                raise FileNotFoundError(f"Directory '{train_pbr_path}' not found in {root_dir}")

        # Gather all samples.
        all_samples = []
        for sid in scene_ids:
            scene_path = os.path.join(dataset_path, sid)
            scene_count = 0
            for cam_id in self.cam_ids:
                info_file = os.path.join(scene_path, f"scene_gt_info_{cam_id}.json")
                pose_file = os.path.join(scene_path, f"scene_gt_{cam_id}.json")
                cam_file  = os.path.join(scene_path, f"scene_camera_{cam_id}.json")
                rgb_dir   = os.path.join(scene_path, f"rgb_{cam_id}")
                if not all(os.path.exists(f) for f in [info_file, pose_file, cam_file, rgb_dir]):
                    continue
                with open(info_file, "r") as f1, open(pose_file, "r") as f2, open(cam_file, "r") as f3:
                    info_json = json.load(f1)
                    pose_json = json.load(f2)
                    cam_json  = json.load(f3)
                all_im_ids = sorted(info_json.keys(), key=lambda x: int(x))
                for im_id_s in all_im_ids:
                    im_id = int(im_id_s)
                    if im_id_s not in cam_json:
                        continue
                    K = np.array(cam_json[im_id_s]["cam_K"], dtype=np.float32).reshape(3, 3)
                    
                    img_name_jpg = os.path.join(rgb_dir, f"{im_id:06d}.jpg")
                    img_name_png = os.path.join(rgb_dir, f"{im_id:06d}.png")
                    
                    # Check if JPG exists, otherwise use PNG.
                    if os.path.exists(img_name_jpg):
                        img_path = img_name_jpg
                    elif os.path.exists(img_name_png):
                        img_path = img_name_png
                    else:
                        continue  # Skip if neither format is found.
                    
                    # Loop through all object instances in the image.
                    for inf, pos in zip(info_json[im_id_s], pose_json[im_id_s]):
                        if pos["obj_id"] != self.obj_id:
                            continue
                    
                        # Use bbox_visib for the visible part of the object.
                        x, y, w_, h_ = inf["bbox_visib"]
                        if w_ <= 0 or h_ <= 0:
                            continue
                    
                        # Retrieve additional BOP challenge metrics if available.
                        visib_fract = inf.get("visib_fract", 1.0)
                        px_count_all = inf.get("px_count_all", w_ * h_)
                        px_count_valid = inf.get("px_count_valid", px_count_all)
                    
                        # Apply filtering thresholds.
                        if visib_fract < 0.1 or px_count_valid < 1000:
                            continue
                    
                        R_mat = np.array(pos["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
                        t = np.array(pos["cam_t_m2c"], dtype=np.float32).reshape(3, 1)
                        all_samples.append({
                            "scene_id": sid,
                            "cam_id": cam_id,
                            "im_id": im_id,
                            "img_path": img_path,
                            "K": K,
                            "R": R_mat,
                            "t": t,
                            "bbox_visib": [x, y, w_, h_],
                            "visib_fract": visib_fract,
                            "px_count_all": px_count_all,
                            "px_count_valid": px_count_valid
                        })
                scene_count += 1
                if self.max_per_scene is not None and scene_count >= self.max_per_scene:
                    break

        # ---- SPLITTING LOGIC ----
        # When using real validation data, assume the samples are pre-defined.
        # Otherwise, split the synthetic data using train_ratio.
        from collections import defaultdict
        groups = defaultdict(list)
        for s in all_samples:
            key = (s["scene_id"], s["cam_id"])
            groups[key].append(s)
        final_samples = []
        for group_samples in groups.values():
            random.shuffle(group_samples)
            n_total = len(group_samples)
            n_train = int(round(self.train_ratio * n_total))
            if self.split == "train":
                selected = group_samples[:n_train]
            else:  # self.split == "val"
                selected = group_samples[n_train:]
            final_samples.extend(selected)
        self.samples = final_samples

        print(f"[INFO] BOPSingleObjDataset(split={self.split}, augment={self.augment}): total={len(self.samples)} samples.")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
        img_path = data["img_path"]
        bgr = cv2.imread(img_path)
        if bgr is None:
            raise IOError(f"Cannot read {img_path}")

        K = data["K"]
        R_mat = data["R"]
        t = data["t"]
        x, y, w, h = map(int, data["bbox_visib"])
        H_img, W_img = bgr.shape[:2]

        # ===== Compute the original (non-augmented) crop =====
        orig_crop = bgr[y:y+h, x:x+w]
        if orig_crop.size == 0:
            raise RuntimeError("Empty crop for original image")
        orig_letter_img, orig_scale, orig_dx, orig_dy = letterbox_preserving_aspect_ratio(orig_crop, target_size=self.target_size)
        orig_letter_img_c = np.ascontiguousarray(orig_letter_img, dtype=np.uint8)
        orig_img_t = torch.from_numpy(orig_letter_img_c).permute(2, 0, 1).float() / 255.0

        # ===== Compute the augmented crop (only for training and if augmentation is enabled) =====
        aug_img_t = None
        if self.split == "train" and self.augment:
            scale_factor = 1.0 + 0.2 * random.random()
            aug_w = int(round(w * scale_factor))
            aug_h = int(round(h * scale_factor))
            max_shift_x = int(0.1 * w)
            max_shift_y = int(0.1 * h)
            shift_x = random.randint(-max_shift_x, max_shift_x)
            shift_y = random.randint(-max_shift_y, max_shift_y)
            aug_x = max(0, min(x - shift_x, W_img - 1))
            aug_y = max(0, min(y - shift_y, H_img - 1))
            aug_w = min(aug_w, W_img - aug_x)
            aug_h = min(aug_h, H_img - aug_y)
            aug_crop = bgr[aug_y:aug_y+aug_h, aug_x:aug_x+aug_w]
            if aug_crop.size == 0:
                raise RuntimeError("Empty crop for augmented image")
            aug_letter_img, aug_scale, aug_dx, aug_dy = letterbox_preserving_aspect_ratio(aug_crop, target_size=self.target_size)
            aug_letter_img_c = np.ascontiguousarray(aug_letter_img, dtype=np.uint8)
            aug_img_t = torch.from_numpy(aug_letter_img_c).permute(2, 0, 1).float() / 255.0
            # Apply color jitter augmentation.
            jitter_transform = T.ColorJitter(brightness=0.5, contrast=0.1, saturation=0.05, hue=0.05)
            img_pil = TF.to_pil_image(aug_img_t)
            img_pil = jitter_transform(img_pil)
            aug_img_t = TF.to_tensor(img_pil)
            aug_img_t = TF.normalize(aug_img_t, mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        # Normalize the original image.
        orig_img_t = TF.normalize(orig_img_t, mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])

        # ===== Compute rotation representations =====
        Rx, Ry, Rz = matrix_to_euler_xyz(R_mat)
        euler_angles = np.array([Rx, Ry, Rz], dtype=np.float32)
        quat = euler_to_quat(euler_angles)
        rep6d = euler_to_6d(euler_angles)
        label_dict = {
            "euler": np.array(euler_angles, dtype=np.float32),
            "quat":  np.array(quat, dtype=np.float32),
            "6d":    np.array(rep6d, dtype=np.float32),
            "R":     np.array(R_mat, dtype=np.float32)  # Add the GT rotation matrix directly.
        }
        # Convert all label arrays to torch tensors.
        for key in label_dict:
            label_dict[key] = torch.from_numpy(label_dict[key])
        
        meta = {
            "scene_id": data["scene_id"],
            "cam_id": data["cam_id"],
            "im_id": data["im_id"]
        }
        return (orig_img_t, aug_img_t, label_dict, meta)


def bop_collate_fn(batch):
    """
    Custom collate function that collates images, labels, and metadata.
    For training, it concatenates originals and augmented images.
    Instead of stacking labels into a single dictionary, we return them as a list.
    """
    orig_imgs, aug_imgs, labels, metas = [], [], [], []
    for sample in batch:
        orig, aug, lbl, meta = sample
        orig_imgs.append(orig)
        labels.append(lbl)
        metas.append(meta)
        if aug is not None:
            aug_imgs.append(aug)
    if len(aug_imgs) > 0:
        imgs_t = torch.cat([torch.stack(orig_imgs, dim=0), torch.stack(aug_imgs, dim=0)], dim=0)
        labels = labels + labels
        metas = metas + metas
    else:
        imgs_t = torch.stack(orig_imgs, dim=0)
    # Return labels as a list, not a batched dictionary.
    return imgs_t, labels, metas


def render_mask(mesh, K, camera_pose, imsize, mesh_poses):
    K = K.copy()
    camera_pose = camera_pose.copy()
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.background_color = np.array([0.0, 0.0, 0.0])
    for mesh_pose in mesh_poses:
        scene.add(mesh, pose=mesh_pose)
    camera = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], zfar=10000)
    camera_pose[1, :] = -camera_pose[1, :]
    camera_pose[2, :] = -camera_pose[2, :]
    camera_pose = np.linalg.inv(camera_pose)
    scene.add(camera, pose=camera_pose)
    light_direction = np.array([0, 0, -1])
    light_direction_world = camera_pose.copy()
    light_direction_world[:3, :3] = light_direction_world[:3, :3] @ light_direction
    light = pyrender.DirectionalLight(color=np.array([1.0, 0, 1.0]), intensity=5)
    scene.add(light, pose=light_direction_world)
    renderer = pyrender.OffscreenRenderer(*imsize)
    color, depth = renderer.render(scene)
    return color, depth


def load_gt_poses(scene_dir, scene_id, cam_ids, image_id, obj_id):
    gt_poses = []
    scene_path = os.path.join(scene_dir, scene_id)
    for cam_id in cam_ids[:1]:
        gt_path = os.path.join(scene_path, f"scene_gt_{cam_id}.json")
        info_path = os.path.join(scene_path, f"scene_gt_info_{cam_id}.json")
        if not os.path.exists(gt_path) or not os.path.exists(info_path):
            print(f"Missing GT files for {cam_id}")
            continue
        with open(gt_path, "r") as f:
            gt_data = json.load(f)
        with open(info_path, "r") as f:
            info_data = json.load(f)
        img_key = str(image_id)
        if img_key not in gt_data or img_key not in info_data:
            print(f"Image {image_id} not found in {cam_id}")
            continue
        objects = gt_data[img_key]
        bboxes = info_data[img_key]
        for obj, bbox in zip(objects, bboxes):
            if obj["obj_id"] != obj_id:
                continue
            rotation_matrix = np.array(obj["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
            translation = np.array(obj["cam_t_m2c"], dtype=np.float32)
            gt_poses.append(calc_pose_matrix(rotation_matrix, translation))
    return gt_poses


def calc_pose_matrix(R_mat, t):
    pose = np.eye(4)
    pose[:3, :3] = R_mat
    pose[:3, 3] = t
    return pose


class Capture:
    def __init__(self, images, depths, Ks, RTs, obj_id, gt_poses=None):
        self.images = images
        self.depths = depths
        self.Ks = Ks
        self.RTs = RTs
        print(gt_poses)
        if gt_poses:
            self.gt_poses = np.linalg.inv(RTs[0]) @ gt_poses

    @classmethod
    def from_dir(cls, scene_dir, cam_ids, image_id, obj_id):
        cam_params = load_camera_params(scene_dir, cam_ids)
        Ks = [cam_params[x]['K'][image_id] for x in cam_ids]
        Rs = [cam_params[x]['R'][image_id] for x in cam_ids]
        Ts = [cam_params[x]['t'][image_id] for x in cam_ids]
        RTs = [calc_pose_matrix(r, t) for r, t in zip(Rs, Ts)]
        image_paths = [glob.glob(os.path.join(scene_dir, f"rgb_{cam_id}", f"{image_id:06d}.*g"))[0] for cam_id in cam_ids]
        depth_paths = [glob.glob(os.path.join(scene_dir, f"depth_{cam_id}", f"{image_id:06d}.*g"))[0] for cam_id in cam_ids]
        images = [cv2.imread(x) for x in image_paths]
        depths = []
        for x in depth_paths:
            this_depth = cv2.imread(x, cv2.IMREAD_UNCHANGED)
            depths.append(this_depth)
        gt_poses = load_gt_poses(scene_dir, '', cam_ids, image_id, obj_id)
        return cls(images, depths, Ks, RTs, obj_id, gt_poses)



def transform_depth_image(depth_image, depth_image_scale, max_depth_mm, background_factor=1.1, new_width=None):
    depth_image = depth_image.copy()
    if len(depth_image.shape) and depth_image.shape[-1] == 3:
        depth_image = depth_image[:,:,0] # get only one channel, they are all the same

    if new_width is not None:
        # resize depth_image to (new_width x R), where R is the aspect ratio of the original image:
        h, w = depth_image.shape
        aspect_ratio = w / h
        new_height = int(new_width / aspect_ratio)
        depth_image = cv2.resize(depth_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    depth_image = depth_image.astype(np.float32) * depth_image_scale

    # In the PBR training data, many times background depth is 0.0, but we want to map it to be behind the target objects.
    # That's why we are mapping anything below 10 mm or above 5 meters to the maximum original depth value plus some margin.
    depth_image[(depth_image < 10) & (depth_image > max_depth_mm)] = 0 # anything below 10 mm or above 5 meters gets mapped to zero
    depth_image_max_distance = np.max(depth_image)
    depth_image[depth_image == 0] = depth_image_max_distance * background_factor
    
    return depth_image



def hillshade(elevation, azimuth=135, altitude=45, cellsize=10):
    azimuth_rad = np.radians(360 - azimuth + 90)
    altitude_rad = np.radians(altitude)

    # Compute the gradient
    dy, dx = np.gradient(elevation, cellsize, cellsize)

    # Slope and aspect
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    aspect = np.where(aspect < 0, 2 * np.pi + aspect, aspect)

    # Hillshade calculation
    shaded = (np.sin(altitude_rad) * np.cos(slope) +
              np.cos(altitude_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect))

    shaded = np.clip(shaded, 0, 1)
    return (shaded * 255).astype(np.uint8)


def apply_noise_sim2real(depth_image):
    # Apply Gaussian noise to the depth image:
    # depth_image is expected to be single-channel2400x2400 float32, with values in the range [0.0, 65535.0].

    depth_image = depth_image.copy()
    if len(depth_image.shape) and depth_image.shape[-1] == 3:
        depth_image = depth_image[:,:,0] # get only one channel, they are all the same

    #depth_image = depth_image.astype(np.float32)
    #depth_image = (255.0 * (depth_image / 255.0)).astype(np.uint8)

    #depth_image = cv2.medianBlur(depth_image, 3)
    depth_image = depth_image.astype(np.float32)
    # OpenCV cv2.medianBlur only supports uint8 images.
    # So we need to convert the depth image to uint8, apply the median blur, and then convert back to float32.
    #depth_image = depth_image.astype(np.uint8)
    #depth_image = cv2.medianBlur(depth_image, 7)
   # depth_image = depth_image.astype(np.float32)

    # Emulate quantization noise:
    L = 10
    depth_image = np.floor(depth_image / L) * L

    # Apply a Gaussian blur to the noise to make it more realistic (introduce spatial correlation)
    noise1 = np.random.normal(0, 10, depth_image.shape).astype(np.float32)
    #noise1 = cv2.GaussianBlur(noise1, (5, 5), 0)

    noise2 = np.random.normal(0, 70, depth_image.shape).astype(np.float32)
    noise2 = cv2.GaussianBlur(noise2, (31, 31), 0)

    # Add the noise to the depth image:
    depth_image = depth_image + noise2


    # Apply salt and pepper noise:
    if True:
        salt_noise_probability = 0.02
        pepper_noise_probability = 0.02
        random_image = np.random.random(depth_image.shape).astype(np.float32) # uniform random noise in the range [0, 1]
        pepper_mask = random_image < pepper_noise_probability
        depth_image[pepper_mask] = 0        
        salt_mask = random_image > (1 - salt_noise_probability)
        depth_image[salt_mask] = 65535
    
    return depth_image


def hillshade_depth_image(depth_image, new_width=1280, azimuth=135, is_synthetic=False):
    """
    depth_image is expected to be 2400x2400 uint16, with values in the range [0, 65535].
    """
    elevation = depth_image.copy()
    if len(elevation.shape) and elevation.shape[-1] == 3:
        elevation = elevation[:,:,0] # get only one channel, they are all the same

    elevation = np.array(elevation).astype(np.float32)
    
    if is_synthetic:
        elevation = apply_noise_sim2real(elevation)
    else:     
        pass
        #elevation = cv2.bilateralFilter(elevation, 7, 255, 100) # these values were tuned empirically for 2400x2400 images

    
    hillshade_img = hillshade(elevation, azimuth=azimuth)
        
    # resize depth_image to 1280xR, where R is the aspect ratio of the original image:
    h, w = hillshade_img.shape
    aspect_ratio = w / h
    new_height = int(new_width / aspect_ratio)
    hillshade_img = cv2.resize(hillshade_img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    if is_synthetic:
        # the following filters were tuned empirically for 1280x1280 images
        hillshade_img = cv2.medianBlur(hillshade_img, 9)
        # apply sharpening filter:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        hillshade_img = cv2.filter2D(hillshade_img, -1, kernel)
        hillshade_img = cv2.filter2D(hillshade_img, -1, kernel)
    
    return hillshade_img


def compose_grey_plus_hillshade_depth_image(img_gray_np, img_depth_np_raw_pixel_values, new_width=1280, is_synthetic=False):

    hillshade_img_0 = hillshade_depth_image(img_depth_np_raw_pixel_values, new_width=new_width, azimuth=0, is_synthetic=is_synthetic)
    hillshade_img_135 = hillshade_depth_image(img_depth_np_raw_pixel_values, new_width=new_width, azimuth=135, is_synthetic=is_synthetic)

    w,h = hillshade_img_0.shape

    if len(img_gray_np.shape) and img_gray_np.shape[-1] == 3:
        img_gray_np = img_gray_np[:,:,0] # get only one channel, they are all the same

    img_gray_np = cv2.resize(img_gray_np, (w,h), interpolation=cv2.INTER_LINEAR)

    composed_img_np = np.stack((img_gray_np, hillshade_img_0, hillshade_img_135), axis=-1)
    return composed_img_np

def compose_grey_lograd_depth_image(img_gray_np, img_depth_np_raw_pixel_values, depth_scale_pixel_to_mm=0.1, max_depth_mm=5000.0):
    """
    Concatenates greyscale image with gradients of depth image into a 3-channel image.
    lograd stands for log-gradient operation.
    Processes depth image with Sobel gradients and compresses them to 0-255 range for 0-500 mm range,
    using log-compression with clipping so that they can be written to disk and fed to YOLO model as 8-bit PNG images.
    The log-compression is done with floor(41.0 * log(x + 1)) to map 0-500 mm to 0-255 range.

    Args:
        img_gray_np: Grayscale image as numpy array
        img_depth_np_raw_pixel_values: Raw depth image as numpy array
        depth_scale_pixel_to_mm: Scale factor to convert depth pixels to mm
        max_depth_mm: Maximum depth value in mm
        
    Returns:
        3-channel numpy array with (Grayscale, f(SobelX), f(SobelY)) channels. where f(x) = floor(41.0 * log(x + 1))
    """
    # Transform depth image
    img_depth_np = transform_depth_image(img_depth_np_raw_pixel_values, depth_image_scale=depth_scale_pixel_to_mm, max_depth_mm=max_depth_mm)

    # Calculate Sobel gradients on depth image
    sobel_x = cv2.Sobel(img_depth_np, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_depth_np, cv2.CV_64F, 0, 1, ksize=3)

    # Compress Sobel gradients to 0-255 range for 0-500 mm range, with clipping
    sobel_x_compressed = np.floor(41.0 * np.log(sobel_x + 1))
    sobel_y_compressed = np.floor(41.0 * np.log(sobel_y + 1))
    sobel_x_clipped = np.clip(sobel_x_compressed, 0, 255).astype(np.uint8)                
    sobel_y_clipped = np.clip(sobel_y_compressed, 0, 255).astype(np.uint8)

    # Stack to create 3-channel image: (Grayscale, SobelX, SobelY)
    return np.stack((img_gray_np, sobel_x_clipped, sobel_y_clipped), axis=-1)

def apply_image_transformations(image, apply_affine_transformations=False, apply_clahe=True, apply_negative=True, apply_contrast_gamma_correction=True, apply_pixel_noise=True):
    # Apply distortions to the image:
    # Flip horizontally and vertically:
    # image = cv2.flip(image, -1)

    # Apply CLAHE contrast and brightness:
    if apply_clahe:  # Apply CLAHE:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(24, 24))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_channel = image[:, :, 0]
        y_channel = clahe.apply(y_channel)
        image[:, :, 0] = y_channel
        image = cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)
        # Apply brightness:
        alpha = 1.5
        beta = 0.0
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        image = np.clip(image, 0, 255).astype(np.uint8)
        # Apply brightness and contrast:
        alpha = 0.5
        beta = 0.0
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        image = np.clip(image, 0, 255).astype(np.uint8)

    if apply_negative:  # Apply negative:
        image = cv2.bitwise_not(image)

    if apply_contrast_gamma_correction:  # Apply Contrast gamma correction:
        alpha = 0.8
        beta = -0.3
        gamma = 3.0

        invGamma = 1.0 / gamma
        table = np.array(
            [(alpha * ((i / 255.0) ** invGamma) + beta) * 255 for i in range(256)]
        )
        table = np.clip(table, 0, 255).astype(np.uint8)
        image = cv2.LUT(image, table)
        image = np.clip(image, 0, 255).astype(np.uint8)

    if apply_pixel_noise:  # Apply pixel noise:
        # Apply Gaussian blur:
        image = cv2.GaussianBlur(image, (5, 5), 0)
        # Apply Gaussian noise:
        noise = (255.0 * np.random.normal(0, 0.4, image.shape[:2]) - 200)
        noise = np.clip(noise, 0, 255).astype(np.uint8)
        noise = cv2.GaussianBlur(noise, (3, 3), 0)
        noise = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
        image = cv2.add(image, noise)

    if apply_affine_transformations:
        shear, scale = 0, 1.2
        angle, tx, ty = 30, 300, 300

        # Define the center of the image
        center = (image.shape[1] // 2, image.shape[0] // 2)

        # Create the transformation matrix
        M = np.eye(3)

        # Rotation & scale
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        M[:2, :3] = rotation_matrix

        # Translation
        M[0, 2] += tx
        M[1, 2] += ty

        # Shearing
        shear_matrix = np.float32([[1, shear, 0], [0, 1, 0]])
        M = np.dot(np.vstack((shear_matrix, [0, 0, 1])), M)

        # Apply the unified transformation
        image = cv2.warpAffine(image, M[:2, :3], (image.shape[1], image.shape[0]))

    return image

def get_color_pallete():
    # Generate color pallete, to be indexed by object ID:
    colors = [
        (255, 0, 0),     # Red
        (0, 255, 0),     # Green
        (0, 0, 255),     # Blue
        (255, 255, 0),   # Yellow
        (255, 0, 255),   # Magenta
        (0, 255, 255),   # Cyan
        (192, 192, 192), # Silver
        (128, 0, 0),     # Maroon
        (128, 128, 0),   # Olive
        (128, 0, 128),   # Purple
        (255, 165, 0),   # Orange
        (0, 128, 128),   # Teal
        (255, 192, 203), # Pink
        (128, 128, 128), # Gray
        (0, 0, 0),       # Black
        (255, 255, 255), # White
        (0, 128, 0),     # Dark Green
        (0, 0, 128),     # Navy
        (128, 128, 255), # Light Blue
        (255, 128, 0),   # Coral
        (255, 128, 128), # Light Coral
        (128, 255, 128)  # Light Green
    ]
    return colors

def get_color_for_class_id(class_id, color_pallete=None, as_bgr=False):
    if color_pallete is None:
        color_pallete = get_color_pallete()
    color = color_pallete[class_id % len(color_pallete)]
    if as_bgr:
        return color[::-1]
    else:
        return color

def extract_roi_with_padding(image, roi, color=(127, 127, 127)):
    """
    Extracts a ROI from an image, padding the out-of-bounds area with a specified color using cv2.copyMakeBorder().
    Args:
        image: The input image.
        roi: A tuple (x, y, w, h) defining the ROI rectangle.
        color: The padding color (default is gray).
    Returns:
        The extracted ROI with padding.
    """
    x, y, w, h = roi
    img_height, img_width = image.shape[:2]

    # Calculate padding values for each side
    top = max(0, -y)
    bottom = max(0, (y + h) - img_height)
    left = max(0, -x)
    right = max(0, (x + w) - img_width)

    # Calculate the ROI within the image boundaries
    x_start = max(0, x)
    y_start = max(0, y)
    x_end = min(img_width, x + w)
    y_end = min(img_height, y + h)

    # Extract the ROI from the image
    roi_cropped = image[y_start:y_end, x_start:x_end]

    # Add padding to the ROI
    padded_roi = cv2.copyMakeBorder(
        roi_cropped,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return padded_roi

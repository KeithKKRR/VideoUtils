import argparse
import multiprocessing

import cv2
import dlib
import json
import numpy as np
import os
import torch

from multiprocessing import Pool, Manager, cpu_count
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import utils, transforms

from face_align import get_key_points, image_align_and_crop, paste_face_back_to_image
from tqdm import tqdm
from PIL import Image


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-file', type=str, required=True, help='json file storing which image pair to swap faces')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size for model inference')
    return parser.parse_args()


def init_progress_bar(total, progress_queue):
    """
    进度条更新
    """
    pbar = tqdm(total=total)  # 设置进度条总数
    while True:
        progress_queue.get()  # 获取进度更新
        pbar.update(1)  # 更新进度条
        if pbar.n >= pbar.total:
            break
    pbar.close()


class InferenceDataset(Dataset):
    def __init__(self, json_file):
        super(InferenceDataset, self).__init__()
        self.face_detector = dlib.get_frontal_face_detector()
        self.face_predictor = dlib.shape_predictor('./shape_predictor_81_face_landmarks.dat')
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        with open(json_file, 'r') as f:
            data_in_json = json.load(f)

        print('Initializing Dataset...')
        self.pair_list = []

        manager = Manager()
        progress_queue = manager.Queue()

        progress_process = multiprocessing.Process(target=init_progress_bar, args=(len(data_in_json), progress_queue))
        progress_process.start()

        with Pool(cpu_count()) as pool:
            results = [pool.apply_async(self.check_face_existence, args=(data_item, progress_queue)) for data_item in
                       data_in_json]
            for result in results:
                result.get()

        self.pair_list = [item.get() for item in results if item.get() is not None]
        progress_process.join()

    def __getitem__(self, idx):
        data_item = self.pair_list[idx]

        source_rgb = cv2.imread(data_item['source_path'])
        source_rgb = cv2.cvtColor(source_rgb, cv2.COLOR_BGR2RGB)
        source_landmarks = get_key_points(source_rgb, data_item['source_face'], self.face_predictor)
        cropped_source_image, source_affine_matrix = image_align_and_crop(source_rgb, source_landmarks, (256, 256))
        cropped_source_image = self.transform(Image.fromarray(cropped_source_image))

        target_rgb = cv2.imread(data_item['target_path'])
        target_rgb = cv2.cvtColor(target_rgb, cv2.COLOR_BGR2RGB)
        target_landmarks = get_key_points(target_rgb, data_item['target_face'], self.face_predictor)
        cropped_target_image, target_affine_matrix = image_align_and_crop(target_rgb, target_landmarks, (256, 256))
        cropped_target_image = self.transform(Image.fromarray(cropped_target_image))

        return {'source_image': cropped_source_image,
                'target_image': cropped_target_image,
                'source_affine': source_affine_matrix,
                'target_affine': target_affine_matrix,
                'source_path': data_item['source_path'],
                'target_path': data_item['target_path'],
                'output_path': data_item['output_path']}

    def __len__(self):
        return len(self.pair_list)

    def check_face_existence(self, data_item, progress_queue):
        """
        Check whether face(s) exist in this image.
        Args:
            data_item: (dict with keys 'source_path', 'target_path', 'output_path')
        Returns:
            data_item or None: if both source face and target face exist, update data_item with face rectangles. Otherwise,
            return None
        """
        source_rgb = cv2.imread(data_item['source_path'])
        source_rgb = cv2.cvtColor(source_rgb, cv2.COLOR_BGR2RGB)
        source_faces = self.face_detector(source_rgb, 1)
        if len(source_faces) == 0:
            progress_queue.put(1)
            return None

        target_rgb = cv2.imread(data_item['target_path'])
        target_rgb = cv2.cvtColor(target_rgb, cv2.COLOR_BGR2RGB)
        target_faces = self.face_detector(target_rgb, 1)
        if len(target_faces) == 0:
            progress_queue.put(1)
            return None

        progress_queue.put(1)
        data_item.update({'source_face': max(source_faces, key=lambda rect: rect.width() * rect.height()),
                          'target_face': max(target_faces, key=lambda rect: rect.width() * rect.height())})

        return data_item


if __name__ == '__main__':
    # 4090 support 8.9 CUDA architecture, but default pytorch version 1.4.0 cannot support it. Instead, use
    # "conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch" to install proper version of pytorch
    # corresponding to current CUDA architecture. However, GCC and G++ version is also important, remember to install
    # proper GCC/G++ version and set environment variable here.
    os.environ['CC'] = 'gcc-9'
    os.environ['CXX'] = 'g++-9'

    """
    export CUDA_PATH=/usr/local/cuda
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    """

    args = arg_parse()
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    # load data
    dataset = InferenceDataset(args.json_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=min(4, args.batch_size), shuffle=False)

    # prepare model
    # load encoder
    encoder = GradualStyleEncoder(50, 'ir_se')
    encoder_ckpt = torch.load('checkpoints/encoder.pt')
    encoder.load_state_dict(get_keys(encoder_ckpt, 'encoder'), strict=True)
    # load mlps
    MLPs = []
    mlp_ckpt = torch.load('checkpoints/mlp.pt')
    for i in range(5):
        mlp = define_mlp(4)
        mlp.load_state_dict(get_keys(mlp_ckpt, f'MLP{i}'), strict=True)
        MLPs.append(mlp)
    network_pkl = 'checkpoints/ffhq512-128.pkl'
    # load decoder
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(self.device)
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(self.device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new.requires_grad_(True)
    # gen_w_avg
    intrinsics = FOV_to_intrinsics(18.837, device=self.device)
    cam_pivot = torch.tensor(self.decoder.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=self.device)
    cam_radius = self.decoder.rendering_kwargs.get('avg_camera_radius', 2.7)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2, cam_pivot, radius=cam_radius,
                                                           device=self.device)
    constant_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    num_samples = 10000
    z_samples = np.random.RandomState(123).randn(num_samples, 512)
    w_samples = self.decoder.mapping(torch.from_numpy(z_samples).to(self.device),
                                     constant_params.repeat([num_samples, 1]), truncation_psi=0.7, truncation_cutoff=14)
    w_samples = w_samples[:, :1, :].cpu().detach().numpy().astype(np.float32)
    w_avg = np.mean(w_samples, axis=0, keepdims=True)
    w_avg = np.repeat(w_avg, 14, axis=1)
    w_avg = torch.tensor(w_avg).to(self.device)

    for batch_idx, batch_data in enumerate(tqdm(dataloader)):
        source_image = batch_data['source_image'].half().to(device)
        target_image = batch_data['target_image'].half().to(device)
        with torch.no_grad():
            img_t, img_s, img_r1 = model([target_image, source_image])

        for idx in range(len(img_r1)):
            os.makedirs(os.path.dirname(batch_data['output_path'][idx]), exist_ok=True)
            # utils.save_image(img_r1[idx], batch_data['output_path'][idx], normalize=True, range=(-1, 1))

            swapped_face = img_r1[idx].permute(1, 2, 0).cpu().numpy()
            swapped_face = (swapped_face + 1) / 2 * 255
            swapped_face = swapped_face.astype(np.uint8)
            target_face = cv2.imread(batch_data['target_path'][idx])[:, :, ::-1].astype(np.float32)
            result = paste_face_back_to_image(target_face, swapped_face, batch_data['target_affine'][idx].cpu().numpy())
            cv2.imwrite(batch_data['output_path'][idx], result[:, :, ::-1])

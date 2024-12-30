import argparse

import cv2
import dlib
import json
import numpy as np
import os
import torch

from pathlib import Path
from torch import nn
from torchvision import utils, transforms

from face_align import get_key_points, image_align_and_crop, paste_face_back_to_image
from training.model import Generator_globalatt_return_32 as Generator
from training.model import Encoder_return_32 as Encoder
from tqdm import tqdm
from PIL import Image


transform = transforms.Compose([transforms.Resize((256, 256)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-file', type=str, required=True, help='json file storing which image pair to swap faces')
    return parser.parse_args()


class Colorize(object):
    def __init__(self, n=19):
        cmap = np.array([(0, 0, 0), (255, 0, 0), (76, 153, 0),
                         (204, 204, 0), (51, 51, 255), (204, 0, 204), (0, 255, 255),
                         (51, 255, 255), (102, 51, 0), (255, 0, 0), (102, 204, 0),
                         (255, 255, 0), (0, 0, 153), (0, 0, 204), (255, 51, 153),
                         (0, 204, 204), (0, 51, 0), (255, 153, 51), (0, 204, 0)],
                        dtype=np.uint8)
        self.cmap = cmap
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


class Model(nn.Module):
    def __init__(self, device="cuda"):
        super(Model, self).__init__()
        self.g_ema = Generator(
            args.size,
            args.latent_channel_size,
            args.latent_spatial_size,
            lr_mul=args.lr_mul,
            channel_multiplier=args.channel_multiplier,
            normalize_mode=args.normalize_mode,
            small_generator=args.small_generator,
        )
        self.e_ema = Encoder(
            args.size,
            args.latent_channel_size,
            args.latent_spatial_size,
            channel_multiplier=args.channel_multiplier,
        )

    def tensor2label(self, label_tensor, n_label):
        label_tensor = label_tensor.cpu().float()
        if label_tensor.size()[0] > 1:
            label_tensor = label_tensor.max(0, keepdim=True)[1]
        label_tensor = Colorize(n_label)(label_tensor)
        label_numpy = label_tensor.numpy()

        return label_numpy

    def forward(self, input):
        trg = input[0]
        src = input[1]

        trg_src = torch.cat([trg, src], dim=0)
        # w = self.e_ema(trg_src)

        w, w_feat = self.e_ema(trg_src)
        w_feat_tgt = [torch.chunk(f, 2, dim=0)[0] for f in w_feat][::-1]

        trg_w, src_w = torch.chunk(w, 2, dim=0)

        fake_img = self.g_ema([trg_w, src_w, w_feat_tgt])

        return trg, src, fake_img


def video_swap_face(data, tqdm_prefix):
    # 1. Select, align and crop source frames/image with face
    # As for image case, just align and crop.
    # As for video case, search from the first frame to last one for any frame with detectable face, align and crop.
    cropped_source_image, M = None, None
    if Path(data['source_path']).suffix in ['.mp4']:  # video case
        cap = cv2.VideoCapture(data['source_path'])
        if not cap.isOpened():
            print(f'Cannot open video: {data["source_path"]}')
            return

        for frame_idx in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, frame = cap.read()
            if not ret:
                print(f'Video loading error: {data["source_path"]}')
                return

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            source_faces = face_detector(frame, 1)
            if len(source_faces) > 0:
                source_faces_max_rect = max(source_faces, key=lambda rect: rect.width() * rect.height())
                source_landmark = get_key_points(frame, source_faces_max_rect, face_predictor)
                cropped_source_image, _ = image_align_and_crop(frame, source_landmark, (256, 256))
                break

        cap.release()
        if cropped_source_image is None:
            print(f'No face in all frames in {data["source_path"]}')
            return
    elif Path(data['source_path']).suffix in ['.png', '.jpg', '.jpeg']:  # image case
        frame = cv2.imread(data['source_path'])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        source_faces = face_detector(frame, 1)
        if len(source_faces) == 0:
            print(f'No face in source image: {data["source_path"]}')
            return

        source_faces_max_rect = max(source_faces, key=lambda rect: rect.width() * rect.height())
        source_landmark = get_key_points(frame, source_faces_max_rect, face_predictor)
        cropped_source_face, _ = image_align_and_crop(frame, source_landmark, (256, 256))
    else:
        print(f'Unsupported type: {Path(data["source_path"]).suffix}')
    # end up searching, aligning and cropping source face

    # 2. Swap and write to new video.
    # Iterate over each frame, if any face can be detected, swap face and write this swapped frame. Otherwise, skip
    # this frame.
    os.makedirs(os.path.dirname(data['output_path']), exist_ok=True)
    cap = cv2.VideoCapture(data['target_path'])
    if not cap.isOpened:
        print(f'Cannot open target video: {data["target_path"]}')
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(data['output_path'], fourcc, fps, (width, height))

    result = np.zeros([height, width, 3], dtype=np.uint8)
    for frame_idx in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), desc=tqdm_prefix):
        ret, frame = cap.read()
        if not ret:
            print(f'Video loading error: {data["source_path"]}')
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        target_faces = face_detector(frame, 1)
        if len(target_faces) == 0:
            out.write(result[:, :, ::-1])
            print(f'No face detected in frame {frame_idx} of {data["target_path"]}. Skip this frame.')
            continue

        target_faces_max_rect = max(target_faces, key=lambda rect: rect.width() * rect.height())
        target_landmark = get_key_points(frame, target_faces_max_rect, face_predictor)
        cropped_target_image, target_affine = image_align_and_crop(frame, target_landmark, (256, 256))

        # model inference
        source_input = transform(Image.fromarray(cropped_source_image)).unsqueeze(0).half().to(device)
        target_input = transform(Image.fromarray(cropped_target_image)).unsqueeze(0).half().to(device)
        with torch.no_grad():
            img_t, img_s, img_r1 = model([target_input, source_input])

        swapped_face = img_r1[0].permute(1, 2, 0).cpu().numpy()
        swapped_face = (swapped_face + 1) / 2 * 255
        swapped_face = swapped_face.astype(np.uint8)
        result = paste_face_back_to_image(frame, swapped_face, target_affine)
        out.write(result[:, :, ::-1])

    cap.release()
    out.release()


if __name__ == '__main__':
    os.environ['CC'] = 'gcc-9'
    os.environ['CXX'] = 'g++-9'

    face_detector = dlib.get_frontal_face_detector()
    face_predictor = dlib.shape_predictor('./shape_predictor_81_face_landmarks.dat')

    args = arg_parse()
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    # prepare model
    args.ckpt = 'session/swap/checkpoints/500000.pt'
    ckpt = torch.load(args.ckpt)
    train_args = ckpt["train_args"]
    for key in vars(train_args):
        if not (key in vars(args)):
            setattr(args, key, getattr(train_args, key))
    model = Model().half().to(device)
    model.g_ema.load_state_dict(ckpt["g_ema"])
    model.e_ema.load_state_dict(ckpt["e_ema"])
    model.eval()

    # load data
    with open(args.json_file, 'r') as f:
        data_in_json = json.load(f)

    # sequential video swap (time-consuming)
    for i in range(len(data_in_json)):
        tqdm_prefix = f'Video [{i}/{len(data_in_json)}]'
        video_swap_face(data_in_json[i], tqdm_prefix)

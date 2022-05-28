#!/usr/bin/env python

from __future__ import annotations

import argparse
import functools
import os
import pathlib
import sys
import tarfile

import cv2
import gradio as gr
import huggingface_hub
import numpy as np
import torch

sys.path.insert(0, 'face_detection')
sys.path.insert(0, 'face_alignment')
sys.path.insert(0, 'emotion_recognition')

from ibug.emotion_recognition import EmoNetPredictor
from ibug.face_alignment import FANPredictor
from ibug.face_detection import RetinaFacePredictor

REPO_URL = 'https://github.com/ibug-group/emotion_recognition'
TITLE = 'ibug-group/emotion_recognition'
DESCRIPTION = f'This is a demo for {REPO_URL}.'
ARTICLE = None

TOKEN = os.environ['TOKEN']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    parser.add_argument('--allow-flagging', type=str, default='never')
    return parser.parse_args()


def load_sample_images() -> list[pathlib.Path]:
    image_dir = pathlib.Path('images')
    if not image_dir.exists():
        image_dir.mkdir()
        dataset_repo = 'hysts/input-images'
        filenames = ['004.tar']
        for name in filenames:
            path = huggingface_hub.hf_hub_download(dataset_repo,
                                                   name,
                                                   repo_type='dataset',
                                                   use_auth_token=TOKEN)
            with tarfile.open(path) as f:
                f.extractall(image_dir.as_posix())
    return sorted(image_dir.rglob('*.jpg'))


def load_face_detector(device: torch.device) -> RetinaFacePredictor:
    model = RetinaFacePredictor(
        threshold=0.8,
        device=device,
        model=RetinaFacePredictor.get_model('mobilenet0.25'))
    return model


def load_landmark_detector(device: torch.device) -> FANPredictor:
    model = FANPredictor(device=device, model=FANPredictor.get_model('2dfan2'))
    return model


def load_model(model_name: str, device: torch.device) -> EmoNetPredictor:
    model = EmoNetPredictor(device=device,
                            model=EmoNetPredictor.get_model(model_name))
    return model


def predict(image: np.ndarray, model_name: str, max_num_faces: int,
            face_detector: RetinaFacePredictor,
            landmark_detector: FANPredictor,
            models: dict[str, EmoNetPredictor]) -> np.ndarray:
    model = models[model_name]
    if len(model.config.emotion_labels) == 8:
        colors = (
            (192, 192, 192),
            (0, 255, 0),
            (255, 0, 0),
            (0, 255, 255),
            (0, 128, 255),
            (255, 0, 128),
            (0, 0, 255),
            (128, 255, 0),
        )
    else:
        colors = (
            (192, 192, 192),
            (0, 255, 0),
            (255, 0, 0),
            (0, 255, 255),
            (0, 0, 255),
        )

    # RGB -> BGR
    image = image[:, :, ::-1]

    faces = face_detector(image, rgb=False)
    if len(faces) == 0:
        raise RuntimeError('No face was found.')
    faces = sorted(list(faces), key=lambda x: -x[4])[:max_num_faces]
    faces = np.asarray(faces)
    _, _, features = landmark_detector(image,
                                       faces,
                                       rgb=False,
                                       return_features=True)
    emotions = model(features)

    res = image.copy()
    for index, face in enumerate(faces):
        box = np.round(face[:4]).astype(int)
        cv2.rectangle(res, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 2)

        emotion = emotions['emotion'][index]
        valence = emotions['valence'][index]
        arousal = emotions['arousal'][index]
        emotion_label = model.config.emotion_labels[emotion].title()

        text_content = f'{emotion_label} ({valence: .01f}, {arousal: .01f})'
        cv2.putText(res,
                    text_content, (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1,
                    colors[emotion],
                    lineType=cv2.LINE_AA)

    return res[:, :, ::-1]


def main():
    gr.close_all()

    args = parse_args()
    device = torch.device(args.device)

    face_detector = load_face_detector(device)
    landmark_detector = load_landmark_detector(device)

    model_names = [
        'emonet248',
        'emonet245',
        'emonet248_alt',
        'emonet245_alt',
    ]
    models = {name: load_model(name, device=device) for name in model_names}

    func = functools.partial(predict,
                             face_detector=face_detector,
                             landmark_detector=landmark_detector,
                             models=models)
    func = functools.update_wrapper(func, predict)

    image_paths = load_sample_images()
    examples = [[path.as_posix(), model_names[0], 30] for path in image_paths]

    gr.Interface(
        func,
        [
            gr.inputs.Image(type='numpy', label='Input'),
            gr.inputs.Radio(model_names,
                            type='value',
                            default=model_names[0],
                            label='Model'),
            gr.inputs.Slider(
                1, 30, step=1, default=30, label='Max Number of Faces'),
        ],
        gr.outputs.Image(type='numpy', label='Output'),
        examples=examples,
        title=TITLE,
        description=DESCRIPTION,
        article=ARTICLE,
        theme=args.theme,
        allow_flagging=args.allow_flagging,
        live=args.live,
    ).launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()

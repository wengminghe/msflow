import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
import cv2 as cv
import torchvision.transforms as T
from pathlib import Path

from models.msflow import MSFlow


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_paths, labels, mask_paths = load_dataset_folder(args)
    if args.visualize:
        Path(os.path.join(args.output_dir, 'images', 'anomaly', 'true')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(args.output_dir, 'images', 'anomaly', 'false')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(args.output_dir, 'images', 'good', 'true')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(args.output_dir, 'images', 'good', 'false')).mkdir(parents=True, exist_ok=True)

    model = MSFlow(**vars(args)).to(device)
    state_dict = torch.load(os.path.join(args.output_dir, 'best_recall_model.pth'), map_location='cpu')
    model = load_state_dict(model, state_dict)
    model.eval()

    preds = []
    for i, (image_path, label, mask_path) in enumerate(tqdm(zip(image_paths, labels, mask_paths), total=len(image_paths), ncols=100)):
        ori_image = cv.imread(image_path)
        mask = np.zeros_like(ori_image)
        if mask_path is not None:
            mask = cv.imread(mask_path)

        image = preprocess(ori_image, label, i, input_size=args.input_size)
        image = image.to(device)
        with torch.no_grad():
            score, score_map = model.predict(image, top_k=args.top_k)
        score = np.max(score)
        if score_map.shape[-1] != score_map.shape[-2]:
            score_map = np.concatenate([score_map[0], score_map[1], score_map[2], score_map[3]], axis=-1)
        else:
            score_map = score_map[0]
        pred = 1 if score > args.threshold else 0
        preds.append(pred)
        if args.visualize:
            save_images(ori_image, mask, label, pred, score_map, args.output_dir, i)

    acc, pre, rec = cal_metric(preds, labels)
    print(f'Accuracy: {acc*100:.2f}%, Precision: {pre*100:.2f}%, Recall: {rec*100:.2f}%')


def preprocess(image, label, idx, input_size=512):
    # if label == 0:
    #     if idx % 2 == 0:
    #         t = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0][(idx // 2) % 10]
    #         t = int(image.shape[1] * t)
    #         if (idx // 10) % 2 == 0:
    #             image[:, t:, :] = 0
    #         else:
    #             image[:, :t, :] = 0

    input_size = (input_size * image.shape[0] // 1000, input_size * image.shape[1] // 1000)
    image = cv.resize(image, (input_size[1], input_size[0]), interpolation=cv.INTER_LANCZOS4)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = transform(image)
    if input_size[0] == input_size[1]:
        image = image.unsqueeze(0)
    else:
        image = torch.chunk(image, chunks=4, dim=-1)
        image = torch.stack(image)
    # image = image.unsqueeze(0)

    return image


def load_dataset_folder(args):
    image_dir = os.path.join(args.data_dir, args.class_name, 'test')
    mask_dir = os.path.join(args.data_dir, args.class_name, 'ground_truth')

    image_paths = [os.path.join(image_dir, 'anomaly', path) for path in os.listdir(os.path.join(image_dir, 'anomaly'))]
    labels = [1 for _ in image_paths]
    mask_paths = [os.path.join(mask_dir, 'anomaly', path) for path in os.listdir(os.path.join(mask_dir, 'anomaly'))]

    image_paths += [os.path.join(image_dir, 'good', path) for path in os.listdir(os.path.join(image_dir, 'good'))]
    labels += [0 for _ in range(len(image_paths) - len(labels))]
    mask_paths += [None for _ in range(len(image_paths) - len(mask_paths))]

    return image_paths, labels, mask_paths


def load_state_dict(model, state_dict):
    maps = {}
    for i in range(len(model.parallel_flows)):
        maps[model.fusion_flow.module_list[i].perm.shape[0]] = i
    temp = {}
    for k, v in state_dict.items():
        if 'fusion_flow' in k and 'perm' in k:
            temp[k.replace(k.split('.')[2], str(maps[v.shape[0]]))] = v
    for k, v in temp.items():
        state_dict[k] = v

    model.load_state_dict(state_dict)
    return model


def cal_metric(preds, labels):
    tp, fn, tn, fp = 0, 0, 0, 0
    for pred, label in zip(preds, labels):
        if label == 1 and pred == 1:
            tp += 1
        if label == 0 and pred == 0:
            tn += 1
        if label == 1 and pred == 0:
            fn += 1
        if label == 0 and pred == 1:
            fp += 1

    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = 0
    if tp + fp != 0:
        precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return accuracy, precision, recall


def save_images(image, mask, label, pred, score_map, output_dir, i):
    score_map = score_map / np.max(score_map)
    score_map = np.uint8(255 * score_map)
    score_map = cv.applyColorMap(score_map, cv.COLORMAP_JET)
    image = cv.resize(image, (score_map.shape[1], score_map.shape[0]), interpolation=cv.INTER_LANCZOS4)
    mask = cv.resize(mask, (score_map.shape[1], score_map.shape[0]), interpolation=cv.INTER_NEAREST)
    image = np.concatenate([image, mask, score_map], axis=0)
    if label == 1 and pred == 1:
        cv.imwrite(os.path.join(output_dir, 'images', 'anomaly', f'true/{i}.png'), image)
    if label == 0 and pred == 0:
        cv.imwrite(os.path.join(output_dir, 'images', 'good', f'true/{i}.png'), image)
    if label == 1 and pred == 0:
        cv.imwrite(os.path.join(output_dir, 'images', 'anomaly', f'false/{i}.png'), image)
    if label == 0 and pred == 1:
        cv.imwrite(os.path.join(output_dir, 'images', 'good', f'false/{i}.png'), image)


def get_args():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--class_name', type=str, default='fiber')
    parser.add_argument('--input_size', type=int, default=1000, help='input image size')

    # model
    parser.add_argument('--extractor', type=str, default='wide_resnet50_2',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnext50_32x4d', 'wide_resnet50_2'])
    parser.add_argument('--pool_type', type=str, default='avg', help='pool type for extracted feature maps')
    parser.add_argument('--parallel_blocks', type=int, default=[2, 5, 8])
    parser.add_argument('--c_conds', type=int, default=[64, 64, 64])
    parser.add_argument('--clamp_alpha', type=float, default=1.9)

    # eval
    parser.add_argument('--top_k', type=float, default=0.03)
    parser.add_argument('--threshold', type=float, default=0.8)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--output_dir', type=str, help='output directory')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()


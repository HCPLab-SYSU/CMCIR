import argparse, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import h5py
from skimage.transform import resize 
import skvideo.io
from PIL import Image

import torch
from torch import nn
import torchvision
import random
import numpy as np
import jsonlines
import pandas as pd
import jsonlines

from preprocess.models import resnext
from preprocess.datautils import utils
from preprocess.datautils import tgif_qa
from preprocess.datautils import msrvtt_qa
from preprocess.datautils import msvd_qa
from preprocess.datautils import sutd_qa
from model.vivit import ViViT, ViT
from model.build import build_model, build_model3d, build_volo
def build_mobilenetv2():
    cnn = torch.hub.load("pytorch/vision", "mobilenet_v2", pretrained=True)
    model = torch.nn.Sequential(*list(cnn.children())[:-1])
    model.cuda()
    model.eval()
    return model

def build_resnet():
    if not hasattr(torchvision.models, args.model):
        raise ValueError('Invalid model "%s"' % args.model)
    if not 'resnet' in args.model:
        raise ValueError('Feature extraction only supports ResNets')
    cnn = getattr(torchvision.models, args.model)(pretrained=True)
    model = torch.nn.Sequential(*list(cnn.children())[:-1])
    model.cuda()
    model.eval()
    return model


def build_resnext():
    model = resnext.resnet101(num_classes=400, shortcut_type='B', cardinality=32,
                              sample_size=112, sample_duration=16,
                              last_fc=False)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    assert os.path.exists('data/preprocess/pretrained/resnext-101-kinetics.pth')
    model_data = torch.load('data/preprocess/pretrained/resnext-101-kinetics.pth', map_location='cpu')
    model.load_state_dict(model_data['state_dict'])
    model.eval()
    return model

def build_vit():
    model = ViT()
    model.train()
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    return model


def run_batch(cur_batch, model):
    """
    Args:
        cur_batch: treat a video as a batch of images
        model: ResNet model for feature extraction
    Returns:
        ResNet extracted feature.
    """
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)

    image_batch = np.concatenate(cur_batch, 0).astype(np.float32)
    image_batch = (image_batch / 255.0 - mean) / std
    image_batch = torch.FloatTensor(image_batch).cuda()
    with torch.no_grad():
        image_batch = torch.autograd.Variable(image_batch)

    feats = model(image_batch)
    feats = feats.data.cpu().clone().numpy()

    return feats

def run_batch_sutd(cur_batch, model):
    """
    Args:
        cur_batch: treat a video as a batch of images
        model: ResNet model for feature extraction
    Returns:
        ResNet extracted feature.
    """
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)

    cur_batch = (cur_batch / 255.0 - mean) / std
    image_batch = np.stack(cur_batch, axis=0).astype(np.float32)
    image_batch = torch.FloatTensor(image_batch)
    with torch.no_grad():
        image_batch = torch.autograd.Variable(image_batch).cuda()
    
    #image_batch = image_batch.unsqueeze(0)
    feats = model(image_batch)
    feats = feats.detach().cpu().numpy()

    return feats


def extract_clips_with_consecutive_frames(path, num_clips=8, num_frames_per_clip=16):
    """
    Args:
        path: path of a video
        num_clips: expected numbers of splitted clips
        num_frames_per_clip: number of frames in a single clip, pretrained model only supports 16 frames
    Returns:
        A list of raw features of clips.
    """
    valid = True
    clips = list()
    try:
        video_data = skvideo.io.vread(path)
    except:
        print('file {} error'.format(path))
        valid = False
        if args.model == 'resnext101':
            return list(np.zeros(shape=(num_clips, 3, num_frames_per_clip, 112, 112))), valid
        else:
            return list(np.zeros(shape=(num_clips, num_frames_per_clip, 3, 224, 224))), valid
    total_frames = video_data.shape[0]
    img_size = (args.image_height, args.image_width)
    for i in np.linspace(0, total_frames, num_clips + 2, dtype=np.int32)[1:num_clips + 1]:
        clip_start = int(i) - int(num_frames_per_clip / 2)
        clip_end = int(i) + int(num_frames_per_clip / 2)
        if clip_start < 0:
            clip_start = 0
        if clip_end > total_frames:
            clip_end = total_frames - 1
        clip = video_data[clip_start:clip_end]
        if clip_start == 0:
            shortage = num_frames_per_clip - (clip_end - clip_start)
            added_frames = []
            for _ in range(shortage):
                added_frames.append(np.expand_dims(video_data[clip_start], axis=0))
            if len(added_frames) > 0:
                added_frames = np.concatenate(added_frames, axis=0)
                clip = np.concatenate((added_frames, clip), axis=0)
        if clip_end == (total_frames - 1):
            shortage = num_frames_per_clip - (clip_end - clip_start)
            added_frames = []
            for _ in range(shortage):
                added_frames.append(np.expand_dims(video_data[clip_end], axis=0))
            if len(added_frames) > 0:
                added_frames = np.concatenate(added_frames, axis=0)
                clip = np.concatenate((clip, added_frames), axis=0)
        new_clip = []
        for j in range(num_frames_per_clip):
            frame_data = clip[j]
            img = Image.fromarray(frame_data).resize(size=img_size)
            frame_data = np.asarray(img)
            frame_data = np.transpose(frame_data, (2, 0, 1))
            new_clip.append(frame_data)
        new_clip = np.asarray(new_clip)  # (num_frames, width, height, channels)
        if args.model in ['resnext101']:
            new_clip = np.squeeze(new_clip)
            new_clip = np.transpose(new_clip, axes=(1, 0, 2, 3))
        clips.append(new_clip)
    return clips, valid

def generate_h5(model, video_ids, num_clips, outfile):
    """
    Args:
        model: loaded pretrained model for feature extraction
        video_ids: list of video ids
        num_clips: expected numbers of splitted clips
        outfile: path of output file to be written
    Returns:
        h5 file containing visual features of splitted clips.
    """
    if args.dataset == "tgif-qa":
        if not os.path.exists('data/tgif-qa/{}'.format(args.question_type)):
            os.makedirs('data/tgif-qa/{}'.format(args.question_type))
    else:
        if not os.path.exists('data/{}'.format(args.dataset)):
            os.makedirs('data/{}'.format(args.dataset))

    dataset_size = len(video_ids)
    print("total_num: ", dataset_size)

    with h5py.File(outfile, 'w') as fd:
        feat_dset = None
        video_ids_dset = None
        i0 = 0
        _t = {'misc': utils.Timer()}
        for i, (video_path, video_id) in enumerate(video_ids):
            _t['misc'].tic()
            clips, valid = extract_clips_with_consecutive_frames(video_path, num_clips=num_clips, num_frames_per_clip=16)
            if args.feature_type == 'appearance':
                clip_feat = []
                if valid:
                    for clip_id, clip in enumerate(clips):
                        if args.dataset=='sutd-qa':
                            feats = run_batch_sutd(clip, model)  # (16, 2048)
                        else:
                            feats = run_batch_sutd(clip, model)  # (16, 2048)
                        feats = feats.squeeze()
                        clip_feat.append(feats)
                else:
                    clip_feat = np.zeros(shape=(num_clips, 16, 2048))
                clip_feat = np.asarray(clip_feat)  # (8, 16, 2048)
                if feat_dset is None:
                    C, F, D = clip_feat.shape
                    feat_dset = fd.create_dataset('resnet_features', (dataset_size, C, F, D),
                                                  dtype=np.float32)
                    video_ids_dset = fd.create_dataset('ids', shape=(dataset_size,), dtype=np.int)
            elif args.feature_type == 'motion':
                clip_torch = torch.FloatTensor(np.asarray(clips)).cuda()
                if valid:
                    clip_feat = model(clip_torch)  # (8, 2048)
                    clip_feat = clip_feat.squeeze()
                    clip_feat = clip_feat.detach().cpu().numpy()
                else:
                    clip_feat = np.zeros(shape=(num_clips, 2048))
                if feat_dset is None:
                    C, D = clip_feat.shape
                    feat_dset = fd.create_dataset('resnext_features', (dataset_size, C, D),
                                                  dtype=np.float32)
                    video_ids_dset = fd.create_dataset('ids', shape=(dataset_size,), dtype=np.int)

            i1 = i0 + 1
            feat_dset[i0:i1] = clip_feat
            video_ids_dset[i0:i1] = video_id
            i0 = i1
            _t['misc'].toc()
            if (i % 1 == 0):
                print('{:d}/{:d} {:.3f}s (projected finish: {:.2f} hours)' \
                      .format(i1, dataset_size, _t['misc'].average_time,
                              _t['misc'].average_time * (dataset_size - i1) / 3600))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='specify which gpu will be used')
    # dataset info
    parser.add_argument('--dataset', default='tgif-qa', choices=['sutd-qa', 'tgif-qa', 'msvd-qa', 'msrvtt-qa'], type=str)
    parser.add_argument('--question_type', default='count', choices=['frameqa', 'count', 'transition', 'action', 'none'], type=str)
    # output
    parser.add_argument('--out', dest='outfile',
                        help='output filepath',
                        default="data/{}/{}_{}_feat_resnet_all.h5", type=str)
    parser.add_argument("--video_dir", help="raw video path", type=str)   
    parser.add_argument("--video_file", help="raw video file path", type=str)       
    # image sizes
    parser.add_argument('--num_clips', default=8, type=int)
    parser.add_argument('--image_height', default=224, type=int)
    parser.add_argument('--image_width', default=224, type=int)

    # network params
    parser.add_argument('--model', default='resnet101', choices=['resnet101', 'resnext101','ViViT', 'Swin'], type=str)
    parser.add_argument('--seed', default='666', type=int, help='random seed')
    args = parser.parse_args()
    if args.model == 'resnet101':
        args.feature_type = 'appearance'
    elif args.model == 'ViViT':
        args.feature_type = 'appearance'
    elif args.model == 'Swin':
        args.feature_type = 'appearance'
    elif args.model == 'resnext101':
        args.feature_type = 'motion'
    else:
        raise Exception('Feature type not supported!')
    # set gpu
    torch.backends.cudnn.benchmark = True
    # Force the pytorch to create context on the specific device 
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if args.model != 'Swin':
        torch.cuda.set_device(args.gpu_id)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # annotation files
    if args.dataset == 'sutd-qa':
        args.video_file = 'datasets/SUTD-TrafficQA/annotations/R3_all.jsonl'
        args.video_dir = 'datasets/SUTD-TrafficQA/raw_videos/'
        #args.video_file='E:/Video-QA/datasets/SUTD-TrafficQA/annotations/vid_filename_to_id.json'
        video_paths = sutd_qa.load_video_path(args)
        random.shuffle(video_paths)
        if args.model == 'resnet101':
            model = build_resnet()
        elif args.model == 'resnext101':
            model = build_resnext()
        elif args.model == 'ViViT':
            model = build_vit()
        elif args.model == 'Swin':
            model = build_model()
        generate_h5(model, video_paths, args.num_clips,
                    args.outfile.format(args.dataset, args.dataset, args.feature_type))
    elif args.dataset == 'tgif-qa':
        args.annotation_file = 'datasets/TGIF-QA/csv/Total_{}_question.csv'
        args.video_dir = 'datasets/TGIF-QA/gifs'
        args.outfile = 'data/{}/{}/{}_{}_{}_feat_resnet.h5'
        video_paths = tgif_qa.load_video_paths(args)
        random.shuffle(video_paths)
        # load model
        if args.model == 'resnet101':
            model = build_resnet()
        elif args.model == 'resnext101':
            model = build_resnext()
        elif args.model == 'Swin':
            model = build_model()
        generate_h5(model, video_paths, args.num_clips,
                    args.outfile.format(args.dataset, args.question_type, args.dataset, args.question_type, args.feature_type))
    elif args.dataset == 'msrvtt-qa':
        args.annotation_file = 'datasets/msrvtt/annotations/{}_qa.json'
        args.video_dir = 'datasets/msrvtt/videos/'
        video_paths = msrvtt_qa.load_video_paths(args)
        random.shuffle(video_paths)
        # load model
        if args.model == 'resnet101':
            model = build_resnet()
        elif args.model == 'resnext101':
            model = build_resnext()
        elif args.model == 'Swin':
            model = build_model()
        generate_h5(model, video_paths, args.num_clips,
                    args.outfile.format(args.dataset, args.dataset, args.feature_type))

    elif args.dataset == 'msvd-qa':
        args.annotation_file = 'datasets/msvd/MSVD-QA/{}_qa.json'
        args.video_dir = 'datasets/msvd/MSVD-QA/video/'
        args.video_name_mapping = 'datasets/msvd/youtube_mapping.txt'
        video_paths = msvd_qa.load_video_paths(args)
        random.shuffle(video_paths)
        # load model
        if args.model == 'resnet101':
            model = build_resnet()
        elif args.model == 'resnext101':
            model = build_resnext()
        elif args.model == 'Swin':
            model = build_model()
        generate_h5(model, video_paths, args.num_clips,
                    args.outfile.format(args.dataset, args.dataset, args.feature_type))

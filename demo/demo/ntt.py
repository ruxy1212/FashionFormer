import os
import sys

# Add the paths to the projects and models directories to the Python path
# project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../mmdetection/projects'))
models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
datasets_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets'))

# sys.path.insert(project_path)
sys.path.insert(0, models_path)
sys.path.insert(0, datasets_path)
import asyncio
from argparse import ArgumentParser

from mmdet.apis import  inference_detector,init_detector


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default='demo.png', help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    # show the results
    ress = model._show_result(
        args.img,
        result,
        score_thr=args.score_thr,
        out_file=args.out_file)
    print("done")
    # ress = model._show_result(
    #     args.img,
    #     result[:2],
    #     score_thr=args.score_thr,
    #     out_file=args.out_file)
    # print(ress[1])

if __name__ == '__main__':
    args = parse_args()
    main(args)
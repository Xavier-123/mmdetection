# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)

"""
E:\\work\\AI_Project\\mmlab\\mmdetection-master\\mmdet\\data\\labelme-data\\coco-formatcoco\\images\\val2017\\19.jpg
E:\\work\\AI_Project\\mmlab\\mmdetection-master\\1.jpeg
E:\\work\\AI_Project\\mmlab\\mmdetection-master\\configs\\detr\\my_detr_r50_8x2_150e_coco.py
E:\\work\\AI_Project\\mmlab\\mmdetection-master\\tools\\work_dirs\\detr_r50_8x2_150e_coco\\latest.pth
"""

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        # '--score-thr', type=float, default=0.2, help='bbox score threshold')
        '--score-thr', type=float, default=0.1, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    print("result:", result[0].shape, result[1].shape)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result,
        palette=args.palette,
        score_thr=args.score_thr)


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result[0],
        palette=args.palette,
        score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
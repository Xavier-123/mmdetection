_base_ = './faster_rcnn_r50_fpn_1x_coco20230206.py'
# fp16 settings
fp16 = dict(loss_scale=512.)

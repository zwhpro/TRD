_base_ = '../yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py'
data_root = 'PATH TO VisDrone/'

class_name = ('pedestrian', 'people', 'bicycle', 'car', 'van','truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')
num_classes = len(class_name)

metainfo = dict(classes=class_name, palette=[(20, 220, 60),(0, 220, 60),(20, 0, 60),(20, 220, 0),(20, 100, 60),(20, 220, 100),(56, 21, 60),(12, 100, 0),(2, 235, 22),(2, 20, 135)])
# -----------train_cfg-----------

max_epochs = 300

save_epoch_intervals = 20

num_last_epochs = 30
# -----------train_cfg-----------
# -----------dataloader-----------

train_batch_size_per_gpu = 8 

train_num_workers = 8  

val_batch_size_per_gpu = 8

val_num_workers = 8
# -----------dataloader-----------

# ========================modified parameters======================
deepen_factor = 0.33
# The scaling factor that controls the width of the network structure
widen_factor = 0.5
# The output channel of the last stage
last_stage_out_channels = 1024

affine_scale = 0.5  # YOLOv5RandomAffine scaling ratio
mixup_prob = 0.1
loss_bbox_weight = 7.5
# =======================Unmodified in most cases==================
img_scale = _base_.img_scale
pre_transform = _base_.pre_transform
last_transform = _base_.last_transform

model = dict(
    backbone=dict(
        type='YOLOv8CSPDarknetTRD',
        last_stage_out_channels=last_stage_out_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, last_stage_out_channels],
        out_channels=[256, 512, last_stage_out_channels]),
    bbox_head=dict(
        head_module=dict(
            widen_factor=widen_factor,
            in_channels=[256, 512, last_stage_out_channels],
            num_classes=num_classes),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xyxy',
            reduction='sum',
            loss_weight=loss_bbox_weight,
            return_iou=False)),
    train_cfg=dict(assigner=dict(num_classes=num_classes)))

mosaic_affine_transform = [
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_aspect_ratio=100,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114))
]

# enable mixup
train_pipeline = [
    *pre_transform, *mosaic_affine_transform,
    dict(
        type='YOLOv5MixUp',
        prob=mixup_prob,
        pre_transform=[*pre_transform, *mosaic_affine_transform]),
    *last_transform
]

train_pipeline_stage2 = [
    *pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114.0)),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=100,
        border_val=(114, 114, 114)), *last_transform
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
_base_.custom_hooks[1].switch_pipeline = train_pipeline_stage2
_base_.custom_hooks[1].switch_epoch = max_epochs - num_last_epochs

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='coco/train.json',
        data_prefix=dict(img='YOLO/images/train/')))

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='coco/val.json',
        data_prefix=dict(img='YOLO/images/val/')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'coco/val.json')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=save_epoch_intervals,
    dynamic_intervals=[(max_epochs - num_last_epochs, 2)])

visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]) # noqa
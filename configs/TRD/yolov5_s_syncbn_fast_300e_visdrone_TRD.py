_base_ = '../yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'
data_root = 'path to VisDrone/'

class_name = ('pedestrian', 'people', 'bicycle', 'car', 'van','truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')
num_classes = len(class_name)

metainfo = dict(classes=class_name, palette=[(20, 220, 60),(0, 220, 60),(20, 0, 60),(20, 220, 0),(20, 100, 60),(20, 220, 100),(56, 21, 60),(12, 100, 0),(2, 235, 22),(2, 20, 135)])
# ===============train_cfg==================
max_epochs = 300
save_epoch_intervals = 20
num_last_epochs = 30
# ===============train_cfg====================
# ===============dataloader=====================
train_batch_size_per_gpu = 12 
train_num_workers = 8  
val_batch_size_per_gpu = 8
val_num_workers = 8
# ==================dataloader===================

# ========================modified parameters======================

deepen_factor = 0.33
# The scaling factor that controls the width of the network structure
widen_factor = 0.5
lr_factor = 0.01
affine_scale = 0.5  # YOLOv5RandomAffine scaling ratio
loss_cls_weight = 0.5
loss_bbox_weight = 0.05
loss_obj_weight = 1.0
mixup_prob = 0.1
anchors = [[(3, 4), (4, 9), (8, 6)], [(8, 14), (16, 9), (15, 19)], [(32, 17), (24, 34), (54, 42)]]
num_det_layers = _base_.num_det_layers
img_scale = _base_.img_scale

model = dict(
    backbone=dict(
        type='YOLOv5CSPDarknetTRD',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    bbox_head=dict(
        head_module=dict(num_classes=num_classes,
            widen_factor=widen_factor),
        prior_generator=dict(base_sizes=anchors),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xywh',
            eps=1e-7,
            reduction='mean',
            return_iou=True),
        loss_cls=dict(loss_weight=loss_cls_weight *
                      (num_classes / 80 * 3 / num_det_layers)),
        loss_obj=dict(loss_weight=loss_obj_weight *
                      ((img_scale[0] / 640)**2 * 3 / num_det_layers))))

# =======================Unmodified in most cases==================
pre_transform = _base_.pre_transform
albu_train_transforms = _base_.albu_train_transforms

mosaic_affine_pipeline = [
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114))
]

# enable mixup
train_pipeline = [
    *pre_transform, *mosaic_affine_pipeline,
    dict(
        type='YOLOv5MixUp',
        prob=mixup_prob,
        pre_transform=[*pre_transform, *mosaic_affine_pipeline]),
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]


train_dataloader = dict(dataset=dict(pipeline=train_pipeline))


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
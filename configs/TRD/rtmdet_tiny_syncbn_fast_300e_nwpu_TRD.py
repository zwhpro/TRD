_base_ = '../rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco.py'

data_root = 'Path to NWPU/'

class_name = ('airplane', 'ship', 'storage tank', 'baseball diamond', 'tennis court','basketball court', 'ground track field', 'harbor', 'bridge', 'vehicle')
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
deepen_factor = 0.167
widen_factor = 0.375

# ratio range for random resize
random_resize_ratio_range = (0.5, 2.0)
# Number of cached images in mosaic
mosaic_max_cached_images = 20
# Number of cached images in mixup
mixup_max_cached_images = 10

# =======================Unmodified in most cases==================
model = dict(
    backbone=dict(
        type='CSPNeXtTRD',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor
        ),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))


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
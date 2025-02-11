_base_ = './yolov5_s_syncbn_fast_300e_visdrone_TRD.py'

# -----------train_cfg-----------
# -----------dataloader-----------

train_batch_size_per_gpu = 4
train_num_workers = 4
val_batch_size_per_gpu = 4
val_num_workers = 4

deepen_factor = 1.33
widen_factor = 1.25

model = dict(
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers
   )

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers
    )
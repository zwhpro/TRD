_base_ = './yolov8_s_syncbn_fast_300e_visdrone_TRD.py'


# -----------train_cfg-----------
# -----------dataloader-----------

train_batch_size_per_gpu = 4
train_num_workers = 4
val_batch_size_per_gpu = 4
val_num_workers = 4


deepen_factor = 1.00
widen_factor = 1.25

model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))



train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers
   )

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers
    )

# img_scale = (640, 640)
# test_pipeline = [
#     dict(
#         type='LoadImageFromFile',
#         file_client_args=_base_.file_client_args),
#     dict(type='mmdet.Resize', scale=img_scale, keep_ratio=False), # 这里将 LetterResize 修改成 mmdet.Resize
#     dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
#     dict(
#         type='mmdet.PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor'))
# ]
# val_dataloader = dict(
#     dataset=dict(pipeline=test_pipeline))
# test_dataloader = val_dataloader
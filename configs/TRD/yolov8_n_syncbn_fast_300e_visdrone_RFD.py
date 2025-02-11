_base_ = './yolov8_s_syncbn_fast_300e_visdrone_TRD.py'

# -----------train_cfg-----------
# -----------dataloader-----------

train_batch_size_per_gpu = 12 
train_num_workers = 8  
val_batch_size_per_gpu = 12
val_num_workers = 8



deepen_factor = 0.33
widen_factor = 0.25
model = dict(
    backbone=dict( type='YOLOv8CSPDarknetRFD',deepen_factor=deepen_factor, widen_factor=widen_factor),
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
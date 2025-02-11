_base_ = [
    '../../../configs/tod-yolo/yolov5_s_syncbn_fast_8xb8-300e_visdrone.py'
]

custom_imports = dict(imports=[
    'projects.assigner_visualization.detectors',
    'projects.assigner_visualization.dense_heads'
])

model = dict(
    type='YOLODetectorAssigner', bbox_head=dict(type='YOLOv5HeadAssigner'))

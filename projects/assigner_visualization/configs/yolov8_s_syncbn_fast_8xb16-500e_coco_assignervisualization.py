_base_ = ['../../../configs/tod-yolo/yolov8_s_syncbn_fast_8xb16-300e_visdrone.py']

custom_imports = dict(imports=[
    'projects.assigner_visualization.detectors',
    'projects.assigner_visualization.dense_heads'
])

model = dict(
    type='YOLODetectorAssigner', bbox_head=dict(type='YOLOv8HeadAssigner'))

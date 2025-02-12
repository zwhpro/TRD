# Copyright (c) OpenMMLab. All rights reserved.
from .base_backbone import BaseBackbone
from .csp_darknet import YOLOv5CSPDarknet, YOLOv8CSPDarknet, YOLOXCSPDarknet
from .trd_darknet import YOLOv8CSPDarknetTRD,YOLOv8CSPDarknetHWD,YOLOv8CSPDarknetRFD,YOLOv8CSPDarknetSCDown,YOLOv8CSPDarknetADown,YOLOv8CSPDarknetFouriDown
from .trd_cspnext  import CSPNeXtTRD
from .csp_resnet import PPYOLOECSPResNet
from .cspnext import CSPNeXt
from .efficient_rep import YOLOv6CSPBep, YOLOv6EfficientRep
from .yolov7_backbone import YOLOv7Backbone


__all__ = [
    'YOLOv5CSPDarknet', 'BaseBackbone', 'YOLOv6EfficientRep', 'YOLOv6CSPBep',
    'YOLOXCSPDarknet', 'CSPNeXt', 'YOLOv7Backbone', 'PPYOLOECSPResNet','YOLOv8CSPDarknetSCDown','YOLOv8CSPDarknetADown','YOLOv8CSPDarknetFouriDown',
    'YOLOv8CSPDarknet','YOLOv8CSPDarknetTRD','YOLOv8CSPDarknetHWD','YOLOv8CSPDarknetRFD','CSPNeXtTRD'
]

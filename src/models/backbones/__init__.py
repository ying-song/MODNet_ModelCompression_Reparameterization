from .wrapper import *
from .wrapper2 import *

SUPPORTED_BACKBONES = {
    'mobilenetv2': MobileNetV2Backbone,
    'mobilenetv2_auto': MobileNetV2BackboneAuto,
    'repmobilenetv2_auto': RepMobileNetV2BackboneAuto
}

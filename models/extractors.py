from .resnet import *


def build_extractor(extractor_name, pretrained=False):
    if extractor_name == 'resnet18':
        extractor = resnet18(pretrained=pretrained, progress=True)
    elif extractor_name == 'resnet34':
        extractor = resnet34(pretrained=pretrained, progress=True)
    elif extractor_name == 'resnet50':
        extractor = resnet50(pretrained=pretrained, progress=True)
    elif extractor_name == 'resnext50_32x4d':
        extractor = resnext50_32x4d(pretrained=pretrained, progress=True)
    elif extractor_name == 'wide_resnet50_2':
        extractor = wide_resnet50_2(pretrained=pretrained, progress=True)

    # output_channels = []
    # if 'wide' in extractor_name:
    #     for i in range(3):
    #         output_channels.append(eval('extractor.layer{}[-1].conv3.out_channels'.format(i+1)))
    # else:
    #     for i in range(3):
    #         output_channels.append(extractor.eval('layer{}'.format(i+1))[-1].conv2.out_channels)

    return extractor

# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py.py    
@Contact :   guzhouweihu@163.com

@Modify Time      @Author        @Version    @Desciption
------------      -----------    --------    -----------
2020/10/7 19:27   guzhouweihu      1.0         None
'''


from .densenet2 import densenet121, densenet161, densenet169, densenet201
from .densenet3 import densenet121 as densenet121v2
from .densenet import densenetd40k12, densenetd100k12, densenetd40k40, densenetd100k40
from .resnet import resnet32, wide_resnet20_8
from .resnetv2 import resnet18, resnet50, resnet34, resnet101, resnet152
from .resnetv3 import ResNet18 as PreActResnet18
from .label_smoothing_resnet import resnet18 as LabelResNet18
from .label_smo_resnet18 import ResNet18 as label_smoothing_resnet18
from .resnet_cluster import resnet18 as resnet18_cluster
from .resnet_cluster2 import resnet18 as resnet18_cluster2
from .resnet_clusterv5 import resnet18 as resnet18_cluster5
from .resnet_clusterv4 import resnet56
from .resnet_multiCluster import resnet18 as multiResnet18
from .densenet_cluster import densenet121 as densenet121_cluster
from .densenet_clusterv2 import densenetd40k12 as densenetd40k12_cluster
from .wrn_cluster import wide_resnet20_8 as wide_resnet_20_8_cluster
from .resnet_clusterCl import resnet18 as resnet18_clusterClv1
from .resnet_clusterClv2 import resnet18 as resnet18_clusterClv2
from .resnet_clusterClv3 import resnet18 as resnet18_clusterClv3

model_dict ={
    'densenet121': densenet121,
    'densenetd40k40': densenetd40k40,
    'densenetd100k40': densenetd100k40,
    'densenet121v2': densenet121v2,
    'densenetd40k12': densenetd40k12,
    'densenetd100k12': densenetd100k12,
    'resnet32': resnet32,
    'wide_resnet20_8': wide_resnet20_8,
    'PreActResnet18': PreActResnet18,
    'resnet18': resnet18,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnet34': resnet34,
    'LabelResNet18': LabelResNet18,
    'label_smoothing_resnet18': label_smoothing_resnet18,
    'resnet18_cluster1': resnet18_cluster,
    'resnet18_cluster2': resnet18_cluster2,
    'resnet56': resnet56,
    'multiResnet18': multiResnet18,
    'resnet18_cluster5': resnet18_cluster5,
    'densenet121_cluster': densenet121_cluster,
    'densenetd40k12_cluster': densenetd40k12_cluster,
    'wide_resnet_20_8_cluster': wide_resnet_20_8_cluster,
    'resnet18_clusterClv1': resnet18_clusterClv1,
    'resnet18_clusterClv2': resnet18_clusterClv2,
    'resnet18_clusterClv3': resnet18_clusterClv3,

}

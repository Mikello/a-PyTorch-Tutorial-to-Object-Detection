import torch
import torch.nn as nn
import math
import torchvision


model_urls = {
'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
}

path_to_save = "/home/z/project/ssd300/data/"

load_state = {'None', 'Pretrained', 'Load'}


def conv3x3(in_planes, out_planes, stride=1):
    
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class tuned_ResNet18(nn.Module):

    def __init__(self, state = 'None', model_path = None):
        assert state in load_state
        
        block = BasicBlock
        layers = [2, 2, 2]
        self.modelPath = model_path
        self.inplanes = 64
        super(tuned_ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        
        self.out_layer1 = conv3x3(128, 512, stride=1)
        self.out_layer2 = conv3x3(256, 1024, stride=1)
        
        if state == 'None':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                    
        elif state == 'Pretrained':
            self.load_pretrained_layers()
        
        elif state == 'Load':
            self.load_state_dict(torch.load(self.modelPath))
        
                
    def save_model(self, model_path = None):
        if model_path == None:
            return
        torch.save(self.state_dict(), model_path)
    
    def load_pretrained_layers(self):
        """
        Из модуля resnet18 используются три из четырез блоков. Выходы будем брать со второго и третьего блока.
        Размерность фич для блока 2 - 256, для блока 3 - 512. Требуется согласование по расмерностям фич, 
        так как для SSD300 требуется размер фич 512 и 1024. Для этого введем дополнительные слои которые 
        принимают на вход  выходы с блоков 2 и 3 и согласовывают по размеру фич. 
        Предобученые веса для этих слоев будем брать со следующего слоя.
        """
        # Current state of base
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # Pretrained Resnet base
        pretrained_state_dict = torchvision.models.resnet18(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())
        
        # Transfer conv. parameters from pretrained model to current model
        for i, param in enumerate(param_names[:90]):  # excluding layer4 and fc parameters
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]
        
        layer3_0_conv1_weight = pretrained_state_dict['layer3.0.conv1.weight']  # (256, 128, 3, 3)
        state_dict['out_layer1.weight']  = torch.cat(
                (layer3_0_conv1_weight, layer3_0_conv1_weight), 0) # (512, 128, 3, 3)
        
        layer4_0_conv1_weight = pretrained_state_dict['layer4.0.conv1.weight'] # (512, 256, 3, 3)
        state_dict['out_layer2.weight']  = torch.cat(
                (layer4_0_conv1_weight, layer4_0_conv1_weight), 0)   # (1024, 256, 3, 3)

        self.load_state_dict(state_dict)

        print("\nLoaded base model.\n")
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        out1 = self.out_layer1(x)
        x = self.layer3(x)
        out2 = self.out_layer2(x)

        return out1, out2


def tuned_resnet18(pretrained=False, **kwargs):
    
    model = tuned_ResNet18(BasicBlock, [2, 2, 2], **kwargs)
    if pretrained:
        model.load_pretrained_layers()
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

if __name__ == '__main__':
    inp = torch.randn(1, 3, 480, 300)
    model_resnet = tuned_ResNet18(state = 'Pretrained')
    out1, out2 = model_resnet(inp)
    print(out1.shape, out2.shape)
    model_resnet.save_model(path_to_save + 'model.pkl')
    
    
    model_resnet = tuned_ResNet18(state = 'Load', model_path = path_to_save + 'model.pkl')
    out1, out2 = model_resnet(inp)
    print(out1.shape, out2.shape)
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ViT import VisionTransformer as VT, vit_model
from model.grl import WarmStartGradientReverseLayer


class ViTgrlNet(nn.Module):
    def __init__(self, base_net='vit_base_patch16_224', use_bottleneck=True, bottleneck_dim=1024, width=1024, class_num=31, args=None):
        super(ViTgrlNet, self).__init__()

        self.base_network = vit_model[base_net](pretrained=True, args=args, VisionTransformerModule=VT)
        self.use_bottleneck = use_bottleneck
        self.grl = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=True)
        if self.use_bottleneck:
            self.bottleneck_layer = [nn.Linear(self.base_network.embed_dim, bottleneck_dim), nn.BatchNorm1d(bottleneck_dim), nn.ReLU(), nn.Dropout(0.5)]
            self.bottleneck = nn.Sequential(*self.bottleneck_layer)

        classifier_dim = bottleneck_dim if use_bottleneck else self.base_network.embed_dim
        self.classifier_layer = [nn.Linear(classifier_dim, width), nn.ReLU(), nn.Dropout(0.5), nn.Linear(width, class_num)]
        self.classifier = nn.Sequential(*self.classifier_layer)

        self.discriminator_layer = [nn.Linear(classifier_dim, width), nn.ReLU(), nn.Dropout(0.5), nn.Linear(width, 1)]
        self.discriminator = nn.Sequential(*self.discriminator_layer)

        if self.use_bottleneck:
            self.bottleneck[0].weight.data.normal_(0, 0.005)
            self.bottleneck[0].bias.data.fill_(0.1)

        for dep in range(2):
            self.discriminator[dep * 3].weight.data.normal_(0, 0.01)
            self.discriminator[dep * 3].bias.data.fill_(0.0)
            self.classifier[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier[dep * 3].bias.data.fill_(0.0)

        self.parameter_list = [
                             {"params":self.base_network.parameters(), "lr":0.1},
                             {"params":self.classifier.parameters(), "lr":1},
                             {"params":self.discriminator.parameters(), "lr":1}]
        if self.use_bottleneck:
            self.parameter_list.extend([{"params":self.bottleneck.parameters(), "lr":1}])


    def forward(self, inputs):
        features = self.base_network.forward_features(inputs)
        if self.use_bottleneck:
            features = self.bottleneck(features)

        outputs = self.classifier(features)

        if self.training:
            outputs_dc = self.discriminator(self.grl(features))
            return outputs, outputs_dc
        else:
            return outputs

class ViTgrl(object):
    def __init__(self, base_net='vit_base_patch16_224', bottleneck_dim=1024, class_num=31, use_gpu=True, args=None):
        self.c_net = ViTgrlNet(base_net, args.use_bottleneck, bottleneck_dim, class_num, args)
        self.use_gpu = use_gpu
        self.is_train = False
        self.iter_num = 0
        self.class_num = class_num
        if self.use_gpu:
            self.c_net = self.c_net.cuda()

    def to_dicts(self):
        return self.c_net.state_dict()

    def from_dicts(self, dicts):
        self.c_net.load_state_dict(dicts)

    def get_loss(self, inputs_source, inputs_target, labels_source, labels_target, args=None):

        inputs = torch.cat((inputs_source, inputs_target))
        outputs, outputs_dc = self.c_net(inputs)

        classifier_loss = nn.CrossEntropyLoss()(outputs.narrow(0, 0, labels_source.size(0)), labels_source)

        domain_loss = 0.
        if args.domain_loss_weight > 0:
            domain_labels = torch.cat(
                (torch.ones(inputs.shape[0] // 2, device=inputs.device, dtype=torch.float),
                 torch.zeros(inputs.shape[0] // 2, device=inputs.device, dtype=torch.float)),
                0)
            domain_loss = nn.BCELoss()(F.sigmoid(outputs_dc).squeeze(), domain_labels) * 2

        self.iter_num += 1

        total_loss = classifier_loss * args.classifier_loss_weight + domain_loss * args.domain_loss_weight

        if args.use_tensorboard:
            all_losses = {}
            all_losses.update({'classifier_loss': classifier_loss})
            all_losses.update({'domain_loss': domain_loss})

            for key, value in all_losses.items():
                if torch.is_tensor(value):
                    args.writer.add_scalar(key, value.item(), self.iter_num)
                else:
                    args.writer.add_scalar(key, value, self.iter_num)
            args.writer.flush()

        return total_loss

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        outputs = self.c_net(inputs)
        return outputs

    def predict(self, inputs, domain='target', output='prob'):
        outputs = self.c_net(inputs)
        if output == 'prob':
            softmax_outputs = F.softmax(outputs)
            return softmax_outputs
        elif output == 'score':
            return outputs
        else:
            raise NotImplementedError('Invalid output')

    def get_parameter_list(self):
        return self.c_net.parameter_list

    def set_train(self, mode):
        self.c_net.train(mode)
        self.is_train = mode

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
import random
import numpy as np
import logging

from model.ViT import Block, PatchEmbed, VisionTransformer, vit_model
from model.grl import WarmStartGradientReverseLayer

class VT(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., distilled=False,
                 args=None):

        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.distilled = distilled

        self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if distilled:
            self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.pre_logits = nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.sr_alpha = args.sr_alpha
        self.sr_layers = args.sr_layers
        self.sr_alpha_adap = self.sr_alpha
        self.iter_num = 0


    def forward_features(self, x):
        B = x.shape[0]

        if self.training and len(self.sr_layers) > 0:
            perturb_layer = random.choice(self.sr_layers)
        else:
            perturb_layer = None

        # perturbing raw input image
        if perturb_layer == -1:
            idx = torch.flip(torch.arange(B // 2, B), dims=[0])
            xm = x[B // 2:] + (x[idx] - x[B // 2:]).detach() * self.sr_alpha_adap
            x = torch.cat((x, xm))

        y = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(y.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        if self.distilled:
            dist_tokens = self.dist_token.expand(y.shape[0], -1, -1)
            y = torch.cat((cls_tokens, dist_tokens, y), dim=1)
        else:
            y = torch.cat((cls_tokens, y), dim=1)
        y = y + self.pos_embed
        y = self.pos_drop(y)


        for layer, blk in enumerate(self.blocks):
            if self.training:
                if layer == perturb_layer:
                    idx = torch.flip(torch.arange(B // 2, B), dims=[0])
                    ym = y[B // 2:] + (y[idx]-y[B // 2:]).detach() * self.sr_alpha_adap
                    y = torch.cat((y, ym))
                y = blk(y)
            else:
                y = blk(y)

        y = self.norm(y)
        y = y[:, 0]
        self.iter_num += 1

        return y


class SSRTNet(nn.Module):
    def __init__(self, base_net='vit_base_patch16_224', use_bottleneck=True, bottleneck_dim=1024, width=1024, class_num=31, args=None):
        super(SSRTNet, self).__init__()

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

        outputs_dc = self.discriminator(self.grl(features))
        outputs = self.classifier(features)

        if self.training:
            return features, outputs, outputs_dc
        else:
            return outputs

class SSRT(object):
    def __init__(self, base_net='vit_base_patch16_224', bottleneck_dim=1024, class_num=31, use_gpu=True, args=None):
        self.net = SSRTNet(base_net, args.use_bottleneck, bottleneck_dim, bottleneck_dim, class_num, args)

        self.use_gpu = use_gpu
        self.is_train = False
        self.iter_num = 0
        self.class_num = class_num
        if self.use_gpu:
            self.net = self.net.cuda()

        self.use_safe_training = args.use_safe_training
        self.sr_loss_weight = args.sr_loss_weight
        self.sr_loss_weight_adap = self.sr_loss_weight

        if self.use_safe_training:
            self.snap_shot = None
            self.restore = False
            self.r = 0.0
            self.r_period = args.adap_adjust_T
            self.r_phase = 0
            self.r_mag = 1.0
            self.adap_adjust_T = args.adap_adjust_T
            self.adap_adjust_L = args.adap_adjust_L
            self.adap_adjust_append_last_subintervals = args.adap_adjust_append_last_subintervals
            self.adap_adjust_last_restore_iter = 0
            self.divs = []
            self.divs_last_period = None


    def to_dicts(self):
        return self.net.state_dict()

    def from_dicts(self, dicts):
        self.net.load_state_dict(dicts, strict=False)

    def get_adjust(self, iter):
        if iter >= self.r_period+self.r_phase:
            return self.r_mag
        return np.sin((iter-self.r_phase)/self.r_period*np.pi/2) * self.r_mag

    def save_snapshot(self):
        self.snap_shot = self.net.state_dict()

    def restore_snapshot(self):
        self.net.load_state_dict(self.snap_shot)
        self.adap_adjust_last_restore_iter = self.iter_num

    def check_div_drop(self):
        flag = False

        for l in range(self.adap_adjust_L+1):
            chunk = np.power(2, l)
            divs_ = np.array_split(np.array(self.divs), chunk)
            divs_ = [d.mean() for d in divs_]

            if self.adap_adjust_append_last_subintervals and self.divs_last_period is not None:
                divs_last_period = np.array_split(np.array(self.divs_last_period), chunk)
                divs_last_period = [d.mean() for d in divs_last_period]
                divs_.insert(0, divs_last_period[-1])

            for i in range(len(divs_)-1):
                if divs_[i+1] < divs_[i] - 1.0:
                    flag = True

        if self.r <= 0.1:
            flag = False

        if flag:
            self.restore = True
            self.r_phase = self.iter_num
            if self.iter_num - self.adap_adjust_last_restore_iter <= self.r_period:
                self.r_period *= 2


    def get_sr_loss(self, out1, out2, sr_epsilon=0.4, sr_loss_p=0.5, args=None):
        prob1_t = F.softmax(out1, dim=1)
        prob2_t = F.softmax(out2, dim=1)

        prob1 = F.softmax(out1, dim=1)
        log_prob1 = F.log_softmax(out1, dim=1)
        prob2 = F.softmax(out2, dim=1)
        log_prob2 = F.log_softmax(out2, dim=1)

        if random.random() <= sr_loss_p:
              log_prob2 = F.log_softmax(out2, dim=1)
              mask1 = (prob1_t.max(-1)[0] > sr_epsilon).float()
              aug_loss = ((prob1 * (log_prob1 - log_prob2)).sum(-1) * mask1).sum() / (mask1.sum() + 1e-6)
        else:
              log_prob1 = F.log_softmax(out1, dim=1)
              mask2 = (prob2_t.max(-1)[0] > sr_epsilon).float()
              aug_loss = ((prob2 * (log_prob2 - log_prob1)).sum(-1) * mask2).sum() / (mask2.sum()+1e-6)

        if args.use_safe_training:
            self.r = self.get_adjust(self.iter_num)
            self.net.base_network.sr_alpha_adap = self.net.base_network.sr_alpha * self.r
            self.sr_loss_weight_adap = self.sr_loss_weight * self.r

            div_unique = prob1.argmax(-1).unique().shape[0]
            self.divs.append(div_unique)

            if (self.iter_num+1) % self.adap_adjust_T == 0 and self.iter_num > 0:
                self.check_div_drop()
                if not self.restore:
                    self.divs_last_period = self.divs

            if args.use_tensorboard:
                args.writer.add_scalar('div_unique', div_unique, self.iter_num)
                args.writer.flush()

        return aug_loss


    def get_loss(self, inputs_source, inputs_target, labels_source, labels_target=None, args=None):
        if self.use_safe_training:
            if self.restore and self.iter_num > 0 and self.sr_loss_weight > 0:
                self.restore_snapshot()
                self.restore = False
                logging.info('Train iter={}:restore model snapshot:r={}'.format(self.iter_num, self.r))

            if self.iter_num % self.adap_adjust_T == 0 and self.sr_loss_weight > 0:
                self.save_snapshot()
                self.divs = []
                logging.info('Train iter={}:save model snapshot:r={}'.format(self.iter_num, self.r))

        inputs = torch.cat((inputs_source, inputs_target))
        _, outputs, outputs_dc = self.net(inputs)

        classification_loss = nn.CrossEntropyLoss()(outputs.narrow(0, 0, labels_source.size(0)), labels_source)

        domain_loss = 0.
        if args.domain_loss_weight > 0:
            domain_labels = torch.cat(
                (torch.ones(inputs_source.shape[0], device=inputs.device, dtype=torch.float),
                 torch.zeros(inputs_target.shape[0], device=inputs.device, dtype=torch.float)),
                0)
            domain_loss = nn.BCELoss()(F.sigmoid(outputs_dc.narrow(0, 0, inputs.size(0))).squeeze(), domain_labels) * 2

        total_loss = classification_loss * args.classification_loss_weight + domain_loss * args.domain_loss_weight

        sr_loss = 0.
        if args.sr_loss_weight > 0:
            outputs_tgt = outputs.narrow(0, labels_source.size(0), inputs.size(0)-labels_source.size(0))
            outputs_tgt_perturb = outputs.narrow(0, inputs.size(0),
                                             inputs.size(0) - labels_source.size(0))

            sr_loss = self.get_sr_loss(outputs_tgt, outputs_tgt_perturb, sr_epsilon=args.sr_epsilon,
                                       sr_loss_p=args.sr_loss_p, args=args)
            total_loss += self.sr_loss_weight_adap * sr_loss

        # mi loss
        if args.mi_loss_weight > 0:
            softmax_out = F.softmax(
                outputs.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0)))
            entropy_loss = torch.mean(torch.sum(-softmax_out * torch.log(softmax_out+1e-6), dim=1))
            msoftmax = softmax_out.mean(dim=0)
            gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-6))
            entropy_loss -= gentropy_loss
            total_loss += args.mi_loss_weight * entropy_loss


        if args.use_tensorboard:
            all_losses = {}
            all_losses.update({'classification_loss': classification_loss})
            all_losses.update({'domain_loss': domain_loss})
            all_losses.update({'sr_loss': sr_loss})

            for key, value in all_losses.items():
                if torch.is_tensor(value):
                    args.writer.add_scalar(key, value.item(), self.iter_num)
                else:
                    args.writer.add_scalar(key, value, self.iter_num)

            args.writer.add_scalar('sr_alpha_adap', self.net.base_network.sr_alpha_adap, self.iter_num)
            args.writer.add_scalar('sr_loss_weight_adap', self.sr_loss_weight_adap, self.iter_num)

            args.writer.flush()

        self.iter_num += 1

        return total_loss


    def predict(self, inputs, output='prob'):
        outputs = self.net(inputs)
        if output == 'prob':
            softmax_outputs = F.softmax(outputs)
            return softmax_outputs
        elif output == 'score':
            return outputs
        else:
            raise NotImplementedError('Invalid output')

    def get_parameter_list(self):
        return self.net.parameter_list

    def set_train(self, mode):
        self.net.train(mode)
        self.is_train = mode

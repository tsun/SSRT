import argparse
from utils.utils import str2bool, str2list, strlist
from torch.utils.tensorboard import SummaryWriter
import time

def argument_parse(args_):
    _timestamp = time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime())
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--model', default=None, type=str,
                        help='uda model')
    parser.add_argument('--base_net', default='vit_base_patch16_224', type=str,
                        help='vit backbone')
    parser.add_argument('--restore_checkpoint', default=None, type=str,
                        help='checkpoint to restore weights')
    parser.add_argument('--use_bottleneck', default=True, type=str2bool,
                        help='whether use bottleneck layer')
    parser.add_argument('--bottleneck_dim', default=None, type=int,
                        help="the dim of the bottleneck layer")

    # dataset
    parser.add_argument('--dataset', default='Office-31', type=str,
                        help='dataset')
    parser.add_argument('--source_path', default=None, type=str,
                        help='path to source (train) image list')
    parser.add_argument('--target_path', default=None, type=str,
                        help='path to target (train) image list')
    parser.add_argument('--test_path', default=None, type=strlist,
                        help='path to (target) test image list')
    parser.add_argument('--rand_aug', default='False', type=str2bool,
                        help='whether use RandAug for target images')
    parser.add_argument('--center_crop', default=False, type=str2bool,
                        help='whether use center crop for images')
    parser.add_argument('--random_resized_crop', default=False, type=str2bool,
                        help='whether use RandomResizedCrop for images')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='number of workers for dataloader')

    # training configuration
    parser.add_argument('--lr', default=0.004, type=float,
                        help='learning rate')
    parser.add_argument('--lr_wd', default=0.0005, type=float,
                        help='weight decay')
    parser.add_argument('--lr_momentum', default=0.9, type=float,
                        help='lr schedule momentum')
    parser.add_argument('--lr_scheduler_gamma', default=0.001, type=float,
                        help='lr scheduler gamma')
    parser.add_argument('--lr_scheduler_decay_rate', default=0.75, type=float,
                        help='lr schedule decay rate')
    parser.add_argument('--lr_scheduler_rate', default=1, type=int,
                        help='lr schedule rate')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch size')
    parser.add_argument('--class_num', default=-1, type=int,
                        help='class number')
    parser.add_argument('--eval_source', default='True', type=str2bool,
                        help='whether evaluate on source data')
    parser.add_argument('--eval_target', default='True', type=str2bool,
                        help='whether evaluate on target data')
    parser.add_argument('--eval_test', default='True', type=str2bool,
                        help='whether evaluate on test data')
    parser.add_argument('--save_checkpoint', default='True', type=str2bool,
                        help='whether save checkpoint')
    parser.add_argument('--iters_per_epoch', default=1000, type=int,
                        help='number of iterations per epoch')
    parser.add_argument('--save_epoch', default='50', type=int,
                        help='interval of saving checkpoint')
    parser.add_argument('--eval_epoch', default='10', type=int,
                        help='interval of evaluating')
    parser.add_argument('--train_epoch', default=50, type=int,
                        help='number of training epochs')

    # environment
    parser.add_argument('--gpu_id', default='0', type=str,
                        help='which gpu to use')
    parser.add_argument('--random_seed', default='0', type=int,
                        help='random seed')
    parser.add_argument('--timestamp', default=_timestamp, type=str,
                        help='timestamp')

    # tensorboard and logger
    parser.add_argument('--use_file_logger', default='True', type=str2bool,
                        help='whether use file logger')
    parser.add_argument('--log_dir', default='log', type=str,
                        help='logging directory')
    parser.add_argument('--use_tensorboard', default='False', type=str2bool,
                        help='whether use tensorboard')
    parser.add_argument('--tensorboard_dir', default='tensorboard', type=str,
                        help='tensorboard directory')
    parser.add_argument('--writer', default=None, type=SummaryWriter,
                        help='tensorboard writer')

    # losses
    parser.add_argument('--classification_loss_weight', default='1.00', type=float,
                        help='weight of semantic classification loss')
    parser.add_argument('--domain_loss_weight', default='1.00', type=float,
                        help='weight of domain classification loss')
    parser.add_argument('--mi_loss_weight', default='0.00', type=float,
                        help='weight of mutual information maximization loss')

    # self refinement
    parser.add_argument('--sr_alpha', default='0.3', type=float,
                        help='self refinement alpha (perturbation magnitude)')
    parser.add_argument('--sr_layers', default='[0,4,8]', type=str2list,
                        help='transformer layers to add perturbation (0 to 11; -1 means raw input images)')
    parser.add_argument('--sr_loss_p', default='0.5', type=float,
                        help='self refinement loss sampling probability')
    parser.add_argument('--sr_loss_weight', default='0.2', type=float,
                        help='weight of self refinement loss')
    parser.add_argument('--sr_epsilon', default='0.4', type=float,
                        help='self refinement epsilon (confidence threshold)')

    # safe training
    parser.add_argument('--use_safe_training', default='True', type=str2bool,
                        help='whether use safe training')
    parser.add_argument('--adap_adjust_restore_optimizor', default='False', type=str2bool,
                        help='whether save and restore snapshot of optimizor')
    parser.add_argument('--adap_adjust_T', default='1000', type=int,
                        help='adaptive adjustment T (interval of saving/restoring snapshot and detecting diversity drop)')
    parser.add_argument('--adap_adjust_L', default='4', type=int,
                        help='adaptive adjustment L (multi-scale detection of diversity dropping)')
    parser.add_argument('--adap_adjust_append_last_subintervals', default='True', type=str2bool,
                        help='whether detect diversity drop along with last sub-intervals')

    args = parser.parse_args(args_)

    # default configurations
    if args.dataset == 'Office-31':
        class_num = 31
        bottleneck_dim = 1024
        center_crop = False
    elif args.dataset == 'Office-Home':
        class_num = 65
        bottleneck_dim = 2048
        center_crop = False
    elif args.dataset == 'visda':
        class_num = 12
        bottleneck_dim = 1024
        center_crop = True
    elif args.dataset == 'DomainNet':
        class_num = 345
        bottleneck_dim = 1024
        center_crop = False
    else:
        raise NotImplementedError('Unsupported dataset')

    args.bottleneck_dim = bottleneck_dim if args.bottleneck_dim is None else args.bottleneck_dim
    args.center_crop = center_crop if args.center_crop is None else args.center_crop
    args.class_num = class_num

    return args
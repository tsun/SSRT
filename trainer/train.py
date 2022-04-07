import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from importlib import import_module
import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import logging
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

from utils.utils import *
from trainer.argument_parser import argument_parse
from trainer.evaluate import evaluate_all
from dataset.data_provider import get_dataloaders, ForeverDataIterator

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train_source(model_instance, dataloaders, optimizer, lr_scheduler, args):
    model_instance.set_train(True)
    print("start train source model...")
    iter_per_epoch = args.iters_per_epoch
    max_iter = args.train_epoch * iter_per_epoch
    iter_num = 0

    total_progress_bar = tqdm.tqdm(desc='Train iter', total=max_iter, initial=0)

    iter_source = ForeverDataIterator(dataloaders["source_tr"])
    for epoch in range(args.train_epoch):
        for _ in tqdm.tqdm(
                range(iter_per_epoch),
                total=iter_per_epoch,
                desc='Train epoch = {}'.format(epoch), ncols=80, leave=False):

            datas = next(iter_source)
            inputs_source, labels_source, indexes_source = datas

            inputs_source = inputs_source.cuda()
            labels_source = labels_source.cuda()

            optimizer.zero_grad()
            outputs_source = model_instance.forward(inputs_source)
            classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
            classifier_loss.backward()
            optimizer.step()

            iter_num += 1
            total_progress_bar.update(1)


        if (epoch+1) % args.eval_epoch == 0:
            evaluate_all(model_instance, dataloaders, epoch+1, args)

        if (epoch+1) % args.save_epoch == 0 and args.save_checkpoint:
            checkpoint_dir = "checkpoint_source/{}/".format(args.base_net)
            checkpoint_name = checkpoint_dir + args_to_str_src(args) + '_' + args.timestamp + '_' + str(
                args.random_seed) + '_epoch_' + str(epoch+1) + '.pth'
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            save_checkpoint(model_instance, checkpoint_name)
            logging.info('Train iter={}:Checkpoint saved to {}'.format(epoch+1, checkpoint_name))

    print('finish source train')



def train(model_instance, dataloaders, optimizer, lr_scheduler, args):
    model_instance.set_train(True)
    logging.info("start training ...")
    iter_num = 0
    iter_per_epoch = args.iters_per_epoch
    max_iter = args.train_epoch * iter_per_epoch
    total_progress_bar = tqdm.tqdm(desc='Train iter', total=max_iter, initial=0)

    iter_source = ForeverDataIterator(dataloaders["source_tr"])
    iter_target = ForeverDataIterator(dataloaders["target_tr"])

    for epoch in range(args.train_epoch):
        for _ in tqdm.tqdm(
                range(iter_per_epoch),
                total=iter_per_epoch,
                desc='Train epoch = {}'.format(epoch), ncols=80, leave=False):

            inputs_source, labels_source, indexes_source = next(iter_source)
            if args.rand_aug:
                inputs_target, labels_target, indexes_target, inputs_rand_target = next(iter_target)
            else:
                inputs_target, labels_target, indexes_target = next(iter_target)
                inputs_rand_target = None

            inputs_source = inputs_source.cuda()
            inputs_target = inputs_target.cuda()
            labels_source = labels_source.cuda()
            labels_target = labels_target.cuda()
            if args.rand_aug:
                inputs_rand_target = inputs_rand_target.cuda()

            # safe training
            if args.use_safe_training and args.adap_adjust_restore_optimizor:
                if model_instance.restore and iter_num > 0 and args.sr_loss_weight > 0:
                    optimizer.load_state_dict(optimizer_snapshot)
                    logging.info('Train iter={}:restore optimizor snapshot'.format(iter_num))

                if iter_num % args.adap_adjust_T == 0 and args.sr_loss_weight > 0:
                    optimizer_snapshot = optimizer.state_dict()
                    logging.info('Train iter={}:save optimizor snapshot'.format(iter_num))

            optimizer.zero_grad()
            if args.rand_aug:
                total_loss = model_instance.get_loss(inputs_source, inputs_target, labels_source, labels_target,
                                                     inputs_rand_target, args=args)
            else:
                total_loss = model_instance.get_loss(inputs_source, inputs_target, labels_source, labels_target, args=args)
            total_loss.backward()
            optimizer.step()

            if iter_num % args.lr_scheduler_rate == 0:
                lr_scheduler.step()

            iter_num += 1
            total_progress_bar.update(1)

        if (epoch+1) % args.eval_epoch == 0 and epoch!=0:
            evaluate_all(model_instance, dataloaders, (epoch+1), args)

        if (epoch+1) % args.save_epoch == 0 and args.save_checkpoint:
            checkpoint_dir = "./checkpoint/{}/".format(args.base_net)
            checkpoint_name = checkpoint_dir+args_to_str(args)+'_'+args.timestamp+'_'+ str(args.random_seed)+'_epoch_'+str(epoch+1)+'.pth'
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            save_checkpoint(model_instance, checkpoint_name)
            logging.info('Train epoch={}:Checkpoint saved to {}'.format((epoch+1), checkpoint_name))


    logging.info('finish training.')


def _init_(args_, header):
    args = argument_parse(args_)

    resetRNGseed(args.random_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    dir = '{}_{}'.format(args.timestamp, '_'.join([_ for _ in [args.model or args.base_model, parse_path(args.source_path),
                                                   parse_path(args.target_path)] if _!='']))

    if not logger_init:
        init_logger(dir, args.use_file_logger, args.log_dir)

    if args.use_tensorboard:
        args.writer = init_tensorboard_writer(args.tensorboard_dir, dir + '_' + str(args.random_seed))

    logging.info(header)
    logging.info(args)

    return args

def train_source_main(args_, header=''):
    args = _init_(args_, header)

    try:
        model_module = import_module('model.'+args.model)
        Model = getattr(model_module, args.model)
        model_instance = Model(base_net=args.base_net, bottleneck_dim=args.bottleneck_dim, use_gpu=True, class_num=args.class_num, args=args)
    except:
        raise NotImplementedError('Unsupported model')

    dataloaders = get_dataloaders(args)
    param_groups = model_instance.get_parameter_list()

    optimizer = torch.optim.SGD(param_groups, args.lr, momentum=args.lr_momentum, weight_decay=args.lr_wd, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_scheduler_gamma * float(x)) ** (-args.lr_scheduler_decay_rate))

    train_source(model_instance, dataloaders, optimizer=optimizer, lr_scheduler=lr_scheduler, args=args)


def train_main(args_, header=''):
    args = _init_(args_, header)

    try:
        model_module = import_module('model.'+args.model)
        Model = getattr(model_module, args.model)
        model_instance = Model(base_net=args.base_net, bottleneck_dim=args.bottleneck_dim, use_gpu=True, class_num=args.class_num, args=args)
    except:
        raise NotImplementedError('Unsupported model')

    dataloaders = get_dataloaders(args)
    param_groups = model_instance.get_parameter_list()

    optimizer = torch.optim.SGD(param_groups, args.lr, momentum=args.lr_momentum, weight_decay=args.lr_wd, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_scheduler_gamma * float(x)) ** (-args.lr_scheduler_decay_rate))

    if args.restore_checkpoint is not None:
        load_checkpoint(model_instance, args.restore_checkpoint)
        logging.info('Model weights restored from: {}'.format(args.restore_checkpoint))

    train(model_instance, dataloaders, optimizer=optimizer, lr_scheduler=lr_scheduler, args=args)
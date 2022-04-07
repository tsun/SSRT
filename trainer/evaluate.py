import torch
import logging
import sklearn
import sklearn.metrics
from utils.utils import parse_path

def evaluate(model_instance, input_loader):
    ori_train_state = model_instance.is_train
    model_instance.set_train(False)
    num_iter = len(input_loader)
    iter_test = iter(input_loader)
    first_test = True

    with torch.no_grad():
        for i in range(num_iter):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            if model_instance.use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            probabilities = model_instance.predict(inputs)

            probabilities = probabilities.data.float()
            labels = labels.data.float()
            if first_test:
                all_probs = probabilities
                all_labels = labels
                first_test = False
            else:
                all_probs = torch.cat((all_probs, probabilities), 0)
                all_labels = torch.cat((all_labels, labels), 0)

    _, predict = torch.max(all_probs, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_labels) / float(all_labels.size()[0])

    avg_acc = sklearn.metrics.balanced_accuracy_score(all_labels.cpu().numpy(),
                                                      torch.squeeze(predict).float().cpu().numpy())

    cm = sklearn.metrics.confusion_matrix(all_labels.cpu().numpy(),
                                          torch.squeeze(predict).float().cpu().numpy())
    accuracies = cm.diagonal() / cm.sum(1)

    model_instance.set_train(ori_train_state)

    # return {'accuracy': np.round(100*accuracy, decimals=2), 'per_class_accuracy': np.round(100*avg_acc, decimals=2)}
    return {'accuracy': accuracy.item(), 'per_class_accuracy': avg_acc, 'accuracies': accuracies}

def format_evaluate_result(eval_result, flag=False):
    if flag:
        return 'Accuracy={}:Per-class accuracy={}:Accs={}'.format(eval_result['accuracy'],
                                                 eval_result['per_class_accuracy'], eval_result['accuracies'])
    else:
        return 'Accuracy={}:Per-class accuracy={}'.format(eval_result['accuracy'], eval_result['per_class_accuracy'])

def evaluate_all(model_instance, dataloaders, iter_num, args):
    flag = args.dataset=='visda'
    if args.eval_source:
        eval_result = evaluate(model_instance, dataloaders["source_val"])
        if args.use_tensorboard:
            args.writer.add_scalar('source_accuracy', eval_result['accuracy'], iter_num)
            args.writer.add_scalar('per_class_source_accuracy', eval_result['per_class_accuracy'], iter_num)
            args.writer.flush()
        print('\n')
        logging.info('Train epoch={}:Source {}'.format(iter_num, format_evaluate_result(eval_result, flag)))

    if args.eval_target and dataloaders["target_val"] is not None:
        eval_result = evaluate(model_instance, dataloaders["target_val"])
        if args.use_tensorboard:
            args.writer.add_scalar('target_accuracy', eval_result['accuracy'], iter_num)
            args.writer.add_scalar('per_class_target_accuracy', eval_result['per_class_accuracy'], iter_num)
            args.writer.flush()
        print('\n')
        logging.info('Train epoch={}:Target {}'.format(iter_num, format_evaluate_result(eval_result, flag)))

    if args.eval_test and dataloaders["test"] is not None:
        if type(dataloaders["test"]) is list or type(dataloaders["test"]) is tuple:
            for i, t_test_loader in enumerate(dataloaders["test"]):
                eval_result = evaluate(model_instance, t_test_loader)
                ext = parse_path(args.test_path[i])
                if args.use_tensorboard:
                    args.writer.add_scalar('test_accuracy_{}'.format(ext), eval_result['accuracy'], iter_num)
                    args.writer.add_scalar('per_class_test_accuracy_{}'.format(ext), eval_result['per_class_accuracy'], iter_num)
                    args.writer.flush()
                print('\n')
                logging.info('Train epoch={}:Test {} {}'.format(iter_num, ext, format_evaluate_result(eval_result, flag)))
        else:
            eval_result = evaluate(model_instance, dataloaders["test"])
            if args.use_tensorboard:
                args.writer.add_scalar('test_accuracy', eval_result['accuracy'], iter_num)
                args.writer.add_scalar('per_class_test_accuracy', eval_result['per_class_accuracy'], iter_num)
                args.writer.flush()
            print('\n')
            logging.info('Train epoch={}:Test {}'.format(iter_num, format_evaluate_result(eval_result, flag)))

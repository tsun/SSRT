from trainer.train import train_main
import time
import socket
import os

timestamp = time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime())
hostName = socket.gethostname()
pid = os.getpid()

domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']

for src in domains:
    for tgt in domains:

        if src == tgt:
            continue

        header = '''
        ++++++++++++++++++++++++++++++++++++++++++++++++       
        {}       
        ++++++++++++++++++++++++++++++++++++++++++++++++   
        @{}:{}
        '''.format

        args = ['--model=ViTgrl',
                '--base_net=vit_base_patch16_224',

                '--gpu=0',
                '--timestamp={}'.format(timestamp),

                '--dataset=DomainNet',
                '--source_path=data/{}_train.txt'.format(src),
                '--target_path=data/{}_train.txt'.format(tgt),
                '--test_path=data/{}_test.txt'.format(tgt),
                '--batch_size=32',

                '--lr=0.004',
                '--train_epoch=40',
                '--save_epoch=40',
                '--eval_epoch=5',
                '--iters_per_epoch=1000',

                '--use_tensorboard=False',
                '--use_file_logger=True',
                '--log_dir=logs/ViTgrl']
        train_main(args, header('\n\t\t'.join(args), hostName, pid))
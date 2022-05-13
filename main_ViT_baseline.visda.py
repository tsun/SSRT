from trainer.train import train_main
import time
import socket
import os

timestamp = time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime())
hostName = socket.gethostname()
pid = os.getpid()

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

        '--dataset=visda',
        '--source_path=data/VisDA2017_train.txt',
        '--target_path=data/VisDA2017_valid.txt',
        '--batch_size=32',

        '--lr=0.002',
        '--train_epoch=20',
        '--save_epoch=20',
        '--eval_epoch=5',
        '--iters_per_epoch=1000',

        '--use_tensorboard=False',
        '--use_file_logger=True',
        '--log_dir=logs/ViTgrl' ]
train_main(args, header('\n\t'.join(args), hostName, pid))
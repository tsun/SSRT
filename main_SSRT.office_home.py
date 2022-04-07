from trainer.train import train_main
import time
import socket
import os

timestamp = time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime())
hostName = socket.gethostname()
pid = os.getpid()

domains = ['Product', 'Clipart', 'Art', 'Real_World']

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

        args = ['--model=SSRT',
                '--base_net=vit_base_patch16_224',

                '--gpu=0',
                '--timestamp={}'.format(timestamp),

                '--dataset=Office-Home',
                '--source_path=data/{}.txt'.format(src),
                '--target_path=data/{}.txt'.format(tgt),
                '--batch_size=32',

                '--lr=0.004',
                '--train_epoch=20',
                '--save_epoch=20',
                '--eval_epoch=5',
                '--iters_per_epoch=1000',

                '--sr_loss_weight=0.2',
                '--sr_alpha=0.3',
                '--sr_layers=[0,4,8]',
                '--sr_epsilon=0.4',

                '--use_safe_training=True',
                '--adap_adjust_T=1000',
                '--adap_adjust_L=4',

                '--use_tensorboard=False',
                '--tensorboard_dir=tbs/SSRT',
                '--use_file_logger=True',
                '--log_dir=logs/SSRT']
        train_main(args, header('\n\t\t'.join(args), hostName, pid))
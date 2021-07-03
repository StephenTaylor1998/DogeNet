
#python train_imagenet.py -d cifar10 -a doge_net18_cifar -b 2048 -j 2 -c 10 --epoch 250 \
#--dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 \
# ./data/dataset

python train_imagenet.py -d cifar10 -a doge_net50_cifar -b 2048 -j 2 -c 10 --epoch 250 \
--dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 \
./data/dataset

#python train_imagenet.py -d cifar10 -a bot_net50_l1_cifar -b 2048 -j 2 -c 10 --epoch 250 \
#--dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 \
#./data/dataset


# transformer models are defined in file "core/models/transformer.py"
#python train_imagenet.py -d cifar10 -a doge_net50_cifar -b 512 -j 2 -c 10 --epoch 400 --dist-url 'tcp://127.0.0.1:8889' \
#--dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset --weight-decay 5e-4 \

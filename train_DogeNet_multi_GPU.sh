#!/bin/bash
#/home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/sub_imagenet/
#/home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/svhn/


python train_imagenet.py -d image_folder -a bot_net50_l1 -b 512 -j 2 -c 10 --epoch 1 --in-shape 3 224 224 \
--data-path /home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/sub_imagenet/ > bot_net50_l1.log

python train_imagenet.py -d image_folder -a bot_net50_l2 -b 512 -j 2 -c 10 --epoch 1 --in-shape 3 224 224 \
--data-path /home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/sub_imagenet/ > bot_net50_l2.log

python train_imagenet.py -d image_folder -a doge_net26 -b 512 -j 2 -c 10 --epoch 1 --in-shape 3 224 224 \
--data-path /home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/sub_imagenet/ > doge_net26.log

python train_imagenet.py -d image_folder -a doge_net50 -b 512 -j 2 -c 10 --epoch 2 --in-shape 3 224 224 \
--data-path /home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/sub_imagenet/ > doge_net50.log
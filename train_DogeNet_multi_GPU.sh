#!/bin/bash
#/home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/sub_imagenet/
#/home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/svhn/

echo "[INFO] Starting..."

#python train_imagenet.py -d image_folder -a bot_net50_l1 -b 512 -j 2 -c 10 --epoch 300 --in-shape 3 224 224 \
#--data-path /home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/sub_imagenet/ > bot_net50_l1.log

#python train_imagenet.py -d image_folder -a bot_net50_l2 -b 512 -j 2 -c 10 --epoch 300 --in-shape 3 224 224 \
#--data-path /home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/sub_imagenet/ > bot_net50_l2.log

#python train_imagenet.py -d image_folder -a doge_net26 -b 512 -j 2 -c 10 --epoch 300 --in-shape 3 224 224 \
#--data-path /home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/sub_imagenet/ > doge_net26.log

#python train_imagenet.py -d image_folder -a doge_net_2x1x3x2 -b 512 -j 2 -c 10 --epoch 300 --in-shape 3 224 224 \
#--data-path /home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/sub_imagenet/ > doge_net_2x1x3x2.log

#python train_imagenet.py -d image_folder -a doge_net50 -b 512 -j 2 -c 10 --epoch 300 --in-shape 3 224 224 \
#--data-path /home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/sub_imagenet/ > doge_net50.log

python train_imagenet.py -d image_folder -a res_net50 -b 512 -j 2 -c 10 --epoch 300 --in-shape 3 224 224 \
--data-path /home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/sub_imagenet/ > res_net50.log

python train_imagenet.py -d image_folder -a res_net26 -b 512 -j 2 -c 10 --epoch 300 --in-shape 3 224 224 \
--data-path /home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/sub_imagenet/ > res_net26.log

echo "[INFO] Done."

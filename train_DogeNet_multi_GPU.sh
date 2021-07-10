#!/bin/bash
#/home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/sub_imagenet/
#/home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/svhn/

echo "[INFO] Starting..."

#python train_imagenet.py -d image_folder -a bot_net50_l1 -b 512 -j 2 -c 10 --epoch 300 --in-shape 3 224 224 \
#--data-path /home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/sub_imagenet/ \
# > data/logs/bot_net50_l1.log

#python train_imagenet.py -d image_folder -a bot_net50_l2 -b 512 -j 2 -c 10 --epoch 300 --in-shape 3 224 224 \
#--data-path /home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/sub_imagenet/ \
# > data/logs/bot_net50_l2.log

#python train_imagenet.py -d image_folder -a doge_net26 -b 512 -j 2 -c 10 --epoch 300 --in-shape 3 224 224 \
#--data-path /home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/sub_imagenet/ \
# > data/logs/doge_net26.log

#python train_imagenet.py -d image_folder -a doge_net_2x1x3x2 -b 512 -j 2 -c 10 --epoch 300 --in-shape 3 224 224 \
#--data-path /home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/sub_imagenet/ \
# > data/logs/doge_net_2x1x3x2.log

#python train_imagenet.py -d image_folder -a doge_net50 -b 512 -j 2 -c 10 --epoch 300 --in-shape 3 224 224 \
#--data-path /home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/sub_imagenet/ \
# > data/logs/doge_net50.log

#python train_imagenet.py -d image_folder -a res_net50 -b 512 -j 2 -c 10 --epoch 300 --in-shape 3 224 224 \
#--data-path /home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/sub_imagenet/ \
# > data/logs/res_net50.log

#python train_imagenet.py -d image_folder -a res_net26 -b 512 -j 2 -c 10 --epoch 300 --in-shape 3 224 224 \
#--data-path /home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/sub_imagenet/ \
# > data/logs/res_net26.log

#python train_imagenet.py -d image_folder -a shibax26 -b 512 -j 2 -c 10 --epoch 300 --in-shape 3 224 224 \
#--data-path /home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/sub_imagenet/ \
#> data/logs/shibax26.log

#python train_imagenet.py -d image_folder -a shibax50 -b 512 -j 2 -c 10 --epoch 300 --in-shape 3 224 224 \
#--data-path /home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/sub_imagenet/ \
#> data/logs/shibax50.log

#python train_imagenet.py -d image_folder -a dogex26 -b 512 -j 2 -c 10 --epoch 300 --in-shape 3 224 224 \
#--data-path /home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/sub_imagenet/ \
#> data/logs/dogex26.log

#python train_imagenet.py -d image_folder -a dogex50 -b 512 -j 2 -c 10 --epoch 300 --in-shape 3 224 224 \
#--data-path /home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/sub_imagenet/ \
#> data/logs/dogex50.log

python train_imagenet.py -d image_folder -a b0 -b 512 -j 2 -c 10 --epoch 300 --in-shape 3 224 224 \
--data-path /home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/sub_imagenet/ \
> data/logs/b0.log

python train_imagenet.py -d image_folder -a shiba26 -b 512 -j 2 -c 10 --epoch 300 --in-shape 3 224 224 \
--data-path /home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/sub_imagenet/ \
> data/logs/shiba26.log

python train_imagenet.py -d image_folder -a shiba50 -b 512 -j 2 -c 10 --epoch 300 --in-shape 3 224 224 \
--data-path /home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/sub_imagenet/ \
> data/logs/shiba50.log

echo "[INFO] Done."

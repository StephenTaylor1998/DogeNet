#!/bin/bash
#/home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/sub_imagenet/
#/home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/svhn/

echo "[INFO] Starting..."

# 75.6   r90 - 51.8
#python train_imagenet.py -d image_folder -a bot_net50_l1 -b 512 -j 2 -c 10 --epoch 300 --in-shape 3 224 224 \
#--data-path /home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/sub_imagenet/ \
#-t --resume data/weights/bot_net50_l1_epoch300_bs51_lr1.0e-01_image_folder/checkpoint_epoch300.pth.tar

# 75    r90 - 52.4
#python train_imagenet.py -d image_folder -a bot_net50_l2 -b 512 -j 2 -c 10 --epoch 300 --in-shape 3 224 224 \
#--data-path /home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/sub_imagenet/ \
#-t --resume data/weights/bot_net50_l2_epoch300_bs51_lr1.0e-01_image_folder/checkpoint_epoch300.pth.tar

## 94.2  r90 - 83.4 - 85.4
#python train_imagenet.py -d image_folder -a doge_net26 -b 512 -j 2 -c 10 --epoch 300 --in-shape 3 224 224 \
#--data-path /home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/sub_imagenet/ \
#-t --resume data/weights/doge_net26_epoch300_bs51_lr1.0e-01_image_folder/checkpoint_epoch300.pth.tar

# 90.4  r90 - 79.4
#python train_imagenet.py -d image_folder -a doge_net_2x1x3x2 -b 512 -j 2 -c 10 --epoch 300 --in-shape 3 224 224 \
#--data-path /home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/sub_imagenet/ \
#-t --resume data/weights/doge_net_2x1x3x2_epoch300_bs51_lr1.0e-01_image_folder/checkpoint_epoch300.pth.tar

# 91.4  r90 - 77.4 - 71.8
#python train_imagenet.py -d image_folder -a doge_net50 -b 512 -j 2 -c 10 --epoch 300 --in-shape 3 224 224 \
#--data-path /home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/sub_imagenet/ \
#-t --resume data/weights/doge_net50_epoch300_bs51_lr1.0e-01_image_folder/checkpoint_epoch300.pth.tar

# 94.2  r90 - 83.4 - 85.4
python train_imagenet.py -d image_folder -a res_net50 -b 512 -j 2 -c 10 --epoch 300 --in-shape 3 224 224 \
--data-path /home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/sub_imagenet/ \
-t --resume data/weights/res_net50_epoch300_bs51_lr1.0e-01_image_folder/checkpoint_epoch300.pth.tar

# 94.2  r90 - 83.4 - 85.4
python train_imagenet.py -d image_folder -a res_net26 -b 512 -j 2 -c 10 --epoch 300 --in-shape 3 224 224 \
--data-path /home/aistudio/Desktop/remote/high-resolution-capsule/data/dataset/sub_imagenet/ \
-t --resume data/weights/res_net26_epoch300_bs51_lr1.0e-01_image_folder/checkpoint_epoch300.pth.tar

echo "[INFO] Done."

# <<if you put this script into source root remove this command
cd ..
# if you put this script into source root remove this command>>


# multi_gpu
# python main.py -a resnet18 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /home/aistudio/Desktop/datasets/ILSVRC2012/ -e

# single_gpu
# python main.py -a resnet18 /home/aistudio/Desktop/datasets/ILSVRC2012/ --resume ./data/model_best.pth.tar -e --gpu 0

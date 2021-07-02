# <<if you put this script into source root remove this command
cd ..
# if you put this script into source root remove this command>>

# train multi-GPU
#python custom_model_trainer.py -a resnet18 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' --epoch 1 --multiprocessing-distributed --world-size 1 --rank 0 /home/aistudio/Desktop/datasets/ILSVRC2012/

# eval multi-GPU
# python custom_model_trainer.py -a resnet18 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' --epoch 1 --multiprocessing-distributed --world-size 1 --rank 0 /home/aistudio/Desktop/datasets/ILSVRC2012/ -e

# train single-GPU
#python custom_model_trainer.py /home/aistudio/Desktop/datasets/ILSVRC2012/ --gpu 0 -b 1

# eval single-GPU
#python custom_model_trainer.py /home/aistudio/Desktop/datasets/ILSVRC2012/ -e --gpu 0 -b 1
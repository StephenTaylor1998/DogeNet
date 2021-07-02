import numpy as np
from matplotlib import pyplot as plt
from utils import get_np_array_from_file


def draw_log(name):
    data, index = get_np_array_from_file('./modified/%s.log.txt' % name)
    plt.plot(index, data)
    plt.savefig('./plt/%s.png' % name)
    print('file %s.png saved in folder ./plt' % name)
    plt.show()


def draw_all():
    # draw test_acc1_log
    draw_log(name='test_acc1_log')
    # draw test_loss_log
    draw_log(name='test_loss_log')
    # draw train
    draw_log(name='train')


if __name__ == '__main__':
    draw_all()

import os
from utils import unrepeated


def get_data_from_log_file(source_root='../logs', log_file='densenet121-train90-lr0.1-batch768.txt'):
    log_path = os.path.join(source_root, log_file)
    print('process data in log %s' % log_path)
    with open(log_path, 'r') as log_file:
        train_log = open('./train.log.txt', 'w')
        test_loss_log = open('./test_loss_log.log.txt', 'w')
        test_acc_log = open('./test_acc_log.log.txt', 'w')
        lines = log_file.readlines()
        new_lines = unrepeated(lines)

        for line in new_lines:

            if line.startswith('Epoch'):
                train_log.writelines(line)

            if line.startswith('Test'):
                test_loss_log.writelines(line)

            if line.startswith(' * '):
                test_acc_log.writelines(line)

        train_log.close()
        test_loss_log.close()
        test_acc_log.close()
    print('temp file ./train.log.txt created')
    print('temp file ./test_acc_log.log.txt created')
    print('temp file ./test_loss_log.log.txt created')

    print('process data in ./train.log.txt...', end='\t')
    with open('./train.log.txt', 'r') as train:
        new_file = open('./modified/train.log.txt', 'w')
        lines = train.readlines()
        # lines = unrepeated(lines, start=-40)
        for line in lines:
            items = line.split('Loss ')
            items = items[1].split(' (')
            # print(items)
            new_file.writelines(items[0] + '\n')
        # new_file.writelines(new_lines)
        new_file.close()

    print('done!')

    print('process data in ./test_loss_log.log.txt', end='\t')
    with open('./test_loss_log.log.txt', 'r') as test_loss_log:
        new_file = open('./modified/test_loss_log.log.txt', 'w')
        lines = test_loss_log.readlines()
        lines = unrepeated(lines, start=-40)

        for line in lines:
            items = line.split('Loss ')
            items = items[1].split(' (')
            # print(items)
            new_file.writelines(items[0] + '\n')
        # new_file.writelines(new_lines)
        new_file.close()

    print('done!')

    print('process data in ./test_acc_log.log.txt', end='\t')
    with open('./test_acc_log.log.txt', 'r') as test_acc_log:
        test_acc1_log = open('./modified/test_acc1_log.log.txt', 'w')
        test_acc5_log = open('./modified/test_acc5_log.log.txt', 'w')
        lines = test_acc_log.readlines()
        for line in lines:
            items = line.split(' ')
            # print(items)
            test_acc1_log.writelines(items[3] + '\n')
            test_acc5_log.writelines(items[5])

        test_acc1_log.close()
        test_acc5_log.close()

    print('done!')


if __name__ == '__main__':
    get_data_from_log_file()

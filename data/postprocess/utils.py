import os
import shutil
import numpy as np


# use in file run_postprocess.py
def check_dir():
    print("file checking...\nWaring: if files don't exist, will be create!")
    if os.path.exists('./modified'):
        print('./modified exist')
    else:
        os.mkdir('./modified')
        print('create ./modified')

    if os.path.exists('./plt'):
        print('./plt exist')
    else:
        os.mkdir('./plt')
        print('create ./plt')

    if os.path.exists('../processed'):
        print('../processed exist')
    else:
        os.mkdir('../processed')
        print('create ../processed')

    print('done!')


# use in file get_data_from_log_file.py
def unrepeated(lines, start=0, stop=-1):
    new_lines = []
    former = ''
    for line in lines:
        if line[start:stop] == former[start:stop]:
            pass
        else:
            new_lines.append(line)

        former = line

    return new_lines


# use in file draw.py, you can use np.loadtxt() instead
def get_np_array_from_file(filename, dtype=np.float):
    data = np.loadtxt(filename, dtype=dtype, delimiter=',')
    index = np.linspace(1, data.size, data.size)
    return data, index


# use in file run_postprocess.py
def copy_resources_to_processed(model_name, file_name, replace=False):
    if os.path.exists('../processed'):
        pass
    else:
        print('../processed not exist, creating...')
        os.mkdir('../processed')
    if os.path.exists('../processed/%s/%s' % (model_name, file_name)):
        print('../processed/%s/%s exist' % (model_name, file_name))

        if replace:
            print('replace former file in ../processed/%s/%s' % (model_name, file_name))
            shutil.rmtree('../processed/%s/%s' % (model_name, file_name))
            shutil.copytree(file_name, '../processed/%s/%s' % (model_name, file_name))

    else:
        print("copy file from '%s' to '../processed/%s/%s'" % (file_name, model_name, file_name))
        shutil.copytree(file_name, '../processed/%s/%s' % (model_name, file_name))


# use in file run_postprocess.py
def clean_temp_files():
    def remove(file_name):
        print('removing temp file %s...' % file_name, end=' ')
        os.remove(file_name)
        print('done!')
    try:
        remove('./train.log.txt')
        remove('./test_acc_log.log.txt')
        remove('./test_loss_log.log.txt')
        remove('./modified/train.log.txt')
        remove('./modified/test_acc1_log.log.txt')
        remove('./modified/test_acc5_log.log.txt')
        remove('./modified/test_loss_log.log.txt')
        remove('./plt/train.png')
        remove('./plt/test_acc1_log.png')
        remove('./plt/test_loss_log.png')

    except Warning:
        print('some error occured when clean files in dir postprocess')


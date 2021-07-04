import os
import shutil


def copy_weights(arg, epoch):
    model_name = arg.arch
    epochs = arg.epochs
    batch_size = arg.batch_size
    learning_rate = arg.lr
    datasets = arg.data_format.strip()

    folder_name = '%s_epoch%d_bs%d_lr%.1e_%s' % \
                  (model_name, epochs, batch_size, learning_rate, datasets)
    # print(folder_name)
    folder_path = os.path.join('./data/weights', folder_name)
    # print('making dir ', folder_path)
    os.makedirs(folder_path, exist_ok=True)

    new_checkpoint = f'checkpoint_epoch{epoch+1}.pth.tar'
    origin_checkpoint = 'checkpoint.pth.tar'
    model_best_name = 'model_best.pth.tar'

    print("copy file from %s to %s" % (
        os.path.join('./data', origin_checkpoint),
        os.path.join(folder_path, new_checkpoint)))
    shutil.copyfile(os.path.join('./data', origin_checkpoint),
                    os.path.join(folder_path, new_checkpoint))

    # print("copy file from %s to %s" % (
    #     os.path.join('./data', model_best_name),
    #     os.path.join(folder_path, model_best_name)))
    # shutil.copyfile(os.path.join('./data', model_best_name),
    #                 os.path.join(folder_path, model_best_name))

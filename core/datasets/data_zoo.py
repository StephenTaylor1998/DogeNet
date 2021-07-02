from . import data


def get_data_by_name(name: str, **kwargs):
    if name.islower() and not name.startswith("_"):
        data_file = data.__dict__[name]
        train_dataset = data_file.classify_train_dataset(**kwargs)
        val_dataset = data_file.classify_val_dataset(**kwargs)
        test_dataset = data_file.classify_test_dataset(**kwargs)
    else:
        print("[ERROR] Data name you selected is not support, but can be registered.")
        print("[WARNING] Custom dataset loading file should be add in 'core/dataset/data/*', "
              "and import in file 'core/dataset/data/__init__.py'.")
        raise NameError

    return train_dataset, val_dataset, test_dataset

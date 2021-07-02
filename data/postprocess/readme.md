# Data post processing script

## This is a script used to process the data in the training log

#### Getting start
 - Modify variable 'model_name' in 'data/postprocess/run_postprocess.py' and makes him point to a log in 'data/logs/'

 - Open terminal in 'data/postprocess/' run this command

```bash
   python run_postprocess.py
```

 -  View results in data/processed

## tips
 - Source of file in data/logs
 
```
    After using the command python main.py to train the model, 
    the training log will be printed but not saved,
    logs can be obtained by a relocating command.
    Log files should be put into data/logs.
    In Linux, the log can be obtained by sh xxx.sh > xxx.log.txt command.
```

Example

```bash
sh train_resnet18_multi_GPU.sh > ./data/logs/resnet18.log.txt
```

 - Annotations

```python
from draw import draw_all
from utils import check_dir
from utils import clean_temp_files
from utils import copy_resources_to_processed
from get_data_from_log_file import get_data_from_log_file


if __name__ == '__main__':
    # You can modify the 'model_name' here in need
    model_name = 'densenet121-train90-lr0.1-batch768'
    # model_name = 'restnet18-train90-lr0.1'
    check_dir()
    get_data_from_log_file(log_file='%s.txt' % model_name)
    draw_all()
    copy_resources_to_processed(model_name=model_name, file_name='modified', replace=True)
    copy_resources_to_processed(model_name=model_name, file_name='plt', replace=True)
    # If you want to clean temp files after postprocess, use this command
    clean_temp_files()
    print("All Done!")
    print("you can check resources in '../processed'")
    print("path in project is 'data/postprocess'")


```




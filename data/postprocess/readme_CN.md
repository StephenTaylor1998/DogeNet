# 数据后处理脚本
## 这是一个用来处理训练日志中的数据的脚本
#### 开始使用
 - 修改 data/postprocess/run_postprocess.py中的model_name使他指向data/logs/中的一个日志

 - 在data/postprocess/路径下打开终端，运行如下命令

```bash
   python run_postprocess.py
```

 -  在data/processed下查看结果

## tips
 - data/logs下的文件来源
    使用命令 python main.py 模型训练后会打印训练日志但不会保存，可以通过重定位获取日志。日志文件放入data/logs中即可。linux用户可尝试bash xxx.sh > xxx.log.txt获取日志。例如

```bash
sh train_resnet18_multi_GPU.sh > ./data/logs/resnet18.log.txt
```

 - 注释

```python
from draw import draw_all
from utils import check_dir
from utils import clean_temp_files
from utils import copy_resources_to_processed
from get_data_from_log_file import get_data_from_log_file


if __name__ == '__main__':
    # 可以根据需要修改这里的model_name
    model_name = 'densenet121-train90-lr0.1-batch768'
    # model_name = 'restnet18-train90-lr0.1'
    check_dir()
    get_data_from_log_file(log_file='%s.txt' % model_name)
    draw_all()
    copy_resources_to_processed(model_name=model_name, file_name='modified', replace=True)
    copy_resources_to_processed(model_name=model_name, file_name='plt', replace=True)
    # 如果需要清理临时文件可以使用如下命令
    # if you want to clean temp file under postprocess, use this
    clean_temp_files()
    print("All Done!")
    print("you can check resources in '../processed'")
    print("path in project is 'data/postprocess'")


```




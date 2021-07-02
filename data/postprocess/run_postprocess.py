from draw import draw_all
from utils import check_dir
from utils import clean_temp_files
from utils import copy_resources_to_processed
from get_data_from_log_file import get_data_from_log_file


if __name__ == '__main__':
    model_name = 'densenet121-train90-lr0.1-batch768'
    # model_name = 'restnet18-train90-lr0.1'
    check_dir()
    get_data_from_log_file(log_file='%s.txt' % model_name)
    draw_all()
    copy_resources_to_processed(model_name=model_name, file_name='modified', replace=True)
    copy_resources_to_processed(model_name=model_name, file_name='plt', replace=True)
    # if you want to clean temp file under postprocess, use this
    clean_temp_files()
    print("All Done!")
    print("you can check resources in '../processed'")
    print("path in project is 'data/postprocess'")



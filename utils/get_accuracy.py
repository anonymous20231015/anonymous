import os
import re
import csv
import numpy as np

# ==================================================================
location = 1001                                    # 总共有 ?? 个epoch
numb_of_epochs = location                 # 表里包含从后往前 ?? 个epoch
num_avg_accuracy = 2               # 计算从后往前 ?? 个epoch的平均精度
num_decimals_avg_accuracy = 4                  # 平均精度保留 ?? 位小数
top_directory = '../result'                 # 原始数据存放的文件夹是 ??
hyper_param_header = ['Dataset', 'Model', 'Method', 'Distribution',
                      'Lamda_Decay', 'B_Decay', 'Lamda_Lr', 'B_Lr']
num_of_hyper_parameters = len(hyper_param_header)
path_csv_file = '../running_data.csv'         # 输出csv文件的路径是 ??
num_empty_lines = 2                                    # 中间空 ?? 行
# ==================================================================

# CHECK the validity of settings above
def check():
    assert isinstance(location, int)
    assert isinstance(numb_of_epochs, int)
    assert isinstance(num_avg_accuracy, int)
    assert isinstance(num_decimals_avg_accuracy, int)
    assert isinstance(top_directory, str)
    assert isinstance(hyper_param_header, list)
    assert isinstance(num_of_hyper_parameters, int)
    assert isinstance(path_csv_file, str)
    assert isinstance(num_empty_lines, int)
    assert location >= 0
    assert numb_of_epochs <= location
    assert num_avg_accuracy <= numb_of_epochs
    assert num_avg_accuracy >= 0
    assert num_empty_lines >= 0
    assert num_decimals_avg_accuracy > 0
    assert 'Distribution' in hyper_param_header

# CHECK if the number of hyper_parameters matches with the header
def check_sequence(params):
    if not len(params) == num_of_hyper_parameters:
        print("ERROR: hyper_parameters and header are MISALIGNED in at least one row!!!")

# INPUT: path of one file
# OUTPUT: a list of hyper_parameters extracted from the path of the file
def clean_directories(directory):
    # split directory by '/', '_', and 'decay'
    directory = re.split('[/_]|(decay)', directory)
    # words TO BE DELETED from file directory
    del_list = ['', '..', None, 'decay', 'result', 'lam-', 'b-', 'val', 'accuracy.txt', 'fedDH']
    # delete words mentioned above
    directory = [i for i in directory if i not in del_list]
    # append None at the end of Fedavg
    while len(directory) < num_of_hyper_parameters:
        directory.append(None)
    return directory


# OUTPUT: a list of paths for all 'accuracy.txt'
def get_file_directories():
    # get ALL file directories under specified directory
    command = f'find {top_directory} -name \'*accuracy*\''
    directories = os.popen(command, 'r').read()
    # split each file as a unique line in a list of file directories
    directories = directories.split("\n")

    # with open('file_directories.txt', 'w', newline='') as file:
    #     file.write(result)
    return directories


# WRITE hyper_parameters and last few epochs' accuracy and AVERAGE accuracy
# ONE file as ONE line
def write_data_as_csv():
    all_acc = []
    # WRITE HEADER TO running_data.csv
    # define column names of hyper_parameters
    header = hyper_param_header
    # add column names for last selected epoch accuracy
    for epoch in range(location - numb_of_epochs + 1, location + 1):
        header.append(str(epoch))
    # add AVERAGE accuracy at the end
    header.append('avg_accuracy')
    # write header to .csv file
    with open(path_csv_file, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(header)

    # get directories of 'accuracy.txt' from all routes
    directories = get_file_directories()

    # record the group information of PREVIOUS line
    prev_method = []

    for directory in sorted(directories, reverse=True):
        # directory is NON-EMPTY
        if len(directory) > 1:
            # open each 'accuracy.txt'
            with open(directory, 'r') as file:
                # store file content in a tuple
                all_loss = eval(file.read())
                # select the last 'number_of_epochs' epochs' accuracy, except the last one
                loss = all_loss[:location][-numb_of_epochs:]
                loss = list(loss)
                # calculate the mean accuracy, keep ?? decimals
                avg_loss = str(round(np.mean(loss[-num_avg_accuracy:]), (num_decimals_avg_accuracy)))
                all_acc.append(avg_loss)

            # get details of hyper_parameters
            hyper_params = clean_directories(directory)
            # CHECK misalignment between header and hyper_params
            check_sequence(hyper_params)
            data_row = hyper_params + avg_loss.split(' ')
            data_row = [data_row]
            print(data_row)

            # GROUP_BY first few hyper_parameters until 'Distribution'
            curr_method = hyper_params[0:hyper_param_header.index('Distribution')+1]
            if curr_method != prev_method and len(prev_method) > 1 and num_empty_lines > 0:
                for i in range(num_empty_lines):
                    data_row.insert(0, [])
            prev_method = curr_method

            # write down each line to .csv file
            with open(path_csv_file, 'a') as file:
                writer = csv.writer(file)
                writer.writerows(data_row)


def main():
    # CHECK if the hyper_parameters defined above are VALID
    check()
    # Create .csv file with observations
    write_data_as_csv()


if __name__ == '__main__':
    main()

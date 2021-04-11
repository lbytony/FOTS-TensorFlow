import codecs
import glob
import json
import operator
import os
import pathlib
from functools import reduce

from natsort import natsorted
from tqdm import tqdm


def get_file_list(folder_path: str, p_postfix: list = None) -> list:
    """
    获取所给文件目录里的指定后缀的文件,读取文件列表目前使用的是 os.walk 和 os.listdir ，这两个目前比 pathlib 快很多
    :param filder_path: 文件夹名称
    :param p_postfix: 文件后缀,如果为 [.*]将返回全部文件
    :return: 获取到的指定类型的文件列表
    """
    assert os.path.exists(folder_path) and os.path.isdir(folder_path)
    if p_postfix is None:
        p_postfix = ['.jpg']
    if isinstance(p_postfix, str):
        p_postfix = [p_postfix]
    file_list = [x for x in glob.glob(folder_path + '/**/*.*', recursive=True) if
                 os.path.splitext(x)[-1] in p_postfix or '.*' in p_postfix]
    return natsorted(file_list)


img_path = r'F:\Code\HealthHelper\Dataset\ICDAR 2019 - LSVT\train_full_images_0'
save_path = r'F:\Code\HealthHelper\Dataset\ICDAR 2019 - LSVT\txt'
json_path = r'F:\Code\HealthHelper\Dataset\ICDAR 2019 - LSVT\train_full_labels.json'

with open(json_path, 'r', encoding='utf-8')as fp:
    json_data = json.load(fp)
    # print('json_data',json_data)

    for file_path in tqdm(get_file_list(img_path, p_postfix=['.jpg'])):
        print('file_path----------------------------', file_path)
        # content = load(file_path)
        file_path = pathlib.Path(file_path)
        image_name = file_path.stem

        new_save_path = save_path + os.sep + image_name + '.txt'
        with codecs.open(new_save_path, mode='w', encoding='utf-8') as fw:

            for piece_data in json_data[image_name]:
                print('piece_data', piece_data)
                label = piece_data['transcription']
                points = piece_data['points']
                # print('points', points)
                points_str_temp = reduce(operator.add, points)
                # print('points_str_temp', points_str_temp)
                points_str = [str(i) + ',' for i in points_str_temp]

                dataset_line = ''.join(points_str) + label
                print('dataset_line', dataset_line)

                # fw = open(new_save_path, 'w', encoding='utf-8')
                print('new_save_path', new_save_path)
                # with open(image_name + '.txt', 'a+') as fw:

                fw.write(dataset_line + '\n')

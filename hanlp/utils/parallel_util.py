# -*- coding:utf-8 -*-
# Author: illusion_Shen
# Time  : 2020-03-07 17:14
import multiprocessing as mps
from numpy import linspace


def parallel_task(data):
    # data = data.to_list()   # 注意data到底是list还是numpy.array还是tf.Tensor？
    convert_arr = ['<pad>', 'O', 'S-NS', 'B-NS', 'E-NS', 'B-NT', 'M-NT', 'E-NT', 'M-NS', 'B-NR', 'M-NR', 'E-NR', 'S-NR', 'S-NT']
    res = [convert_arr[tid] for tid in data]
    return res


def split_data(data):
    """
    设cpu数量是m，将数据等分成m块
    param data: 待分数据
    return: 被分成m块的data数据，形式为list
    """
    cores = int(mps.cpu_count() * 0.9)   # 决定使用多少cpu做计算——这里显然应该优化，就试试
    split_num = linspace(0, len(data), cores + 1, dtype=int)
    data_seg = [data[split_num[j]:split_num[j + 1]] for j in range(len(split_num) - 1)]
    return data_seg, cores


def do_parallel(func, Y):
    data_segment, cpus = split_data(Y)
    pool = mps.Pool(processes=cpus)
    r = []
    for i in data_segment:
        i_ex = pool.apply_async(func, args=(i,))
        r.append(i_ex)
    pool.close()
    pool.join()
    res = []
    for j in r:
        res += j.get()
    return res

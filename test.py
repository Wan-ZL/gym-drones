# import pandas as pd
# from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
from os import listdir
import tensorflow as tf
import glob
from tensorflow.python.summary.summary_iterator import summary_iterator



print(tf. __version__)

def get_average_value(setting_folder, scheme_name, trial_id, tag_name, print_log=False):
    path_list = glob.glob('data/'+setting_folder+'/runs_'+scheme_name+'/*--Trial_'+str(trial_id)+'-eps')
    path = path_list[0] + '/'   # this should only have one result
    file_name = listdir(path)[0]        # only one file in the directory
    if print_log:
        print(path + file_name)
    value_set = []
    for summary_set in tf.compat.v1.train.summary_iterator(path + file_name):
        for value in summary_set.summary.value:
            if value.tag == tag_name:
                value_set.append(value.simple_value)
                if print_log:
                    print("step", summary_set.step, tag_name+" value", value.simple_value)


    ave_value = sum(value_set)/len(value_set) if len(value_set) else 0
    if print_log:
        print("average", ave_value)
    return ave_value



setting_folder = '30_5'
scheme_name = 'def'
trial_id = 98
tag_name = "Energy Consumption"
print('30_5:', get_average_value('30_5', scheme_name, trial_id, tag_name))
print('35_5:', get_average_value('35_5', scheme_name, trial_id, tag_name))
print('40_5:', get_average_value('40_5', scheme_name, trial_id, tag_name))
print('45_5:', get_average_value('45_5', scheme_name, trial_id, tag_name))
print('50_5:', get_average_value('50_5', scheme_name, trial_id, tag_name))
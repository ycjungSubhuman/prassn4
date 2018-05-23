import numpy as np
import random

def shuffle(classified_data_list):
    for i in range(len(classified_data_list)):
        rows = np.split(classified_data_list[i], classified_data_list[i].shape[0])
        random.shuffle(rows)
        classified_data_list[i] = np.vstack(rows)

def take_n(classified_data_list, n):
    for i in range(len(classified_data_list)):
        classified_data_list[i] = classified_data_list[i][:n, :]

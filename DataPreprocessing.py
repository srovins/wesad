#!/usr/bin/env python
# coding: utf-8

# # WESAD dataset preprocessing and exploratory data analysis 



import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')



DATA_PATH = 'data/WESAD/'
DATA_PATH = 'data/WESAD/'
chest_columns=['sid', 'acc1', 'acc2', 'acc3', 'ecg', 'emg', 'eda', 'temp', 'resp', 'label']
all_columns =['sid', 'c_acc_x', 'c_acc_y', 'c_acc_z', 'ecg', 'emg', 'c_eda', 'c_temp', 'resp', 'w_acc_x' , 'w_acc_y', 'w_acc_z', 'bvp', 'w_eda', 'w_temp', 'label']
ids = [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]



sf_BVP = 64
sf_EDA = 4
sf_TEMP = 4
sf_ACC = 32
sf_chest = 700 



# convert data from pickle dictionary format to dataframe for wrist
def pkl_to_np_wrist(filename, subject_id):
    unpickled_df = pd.read_pickle(filename)
    wrist_acc = unpickled_df["signal"]["wrist"]["ACC"]
    wrist_bvp = unpickled_df["signal"]["wrist"]["BVP"]
    wrist_eda = unpickled_df["signal"]["wrist"]["EDA"]
    wrist_temp = unpickled_df["signal"]["wrist"]["TEMP"]
    lbl = unpickled_df["label"].reshape(unpickled_df["label"].shape[0],1)
    
    n_wrist_acc = len(wrist_acc)
    n_wrist_bvp = len(wrist_bvp)
    n_wrist_eda = len(wrist_eda)
    n_wrist_temp = len(wrist_temp)

    print("wrist_bvp shape: ", wrist_bvp.shape)
    print("wrist_eda shape: ", wrist_eda.shape)
    print("wrist_temp shape: ", wrist_temp.shape)
    print("wrist_acc shape: ", wrist_acc.shape)
    
    
    
    sid_acc = np.repeat(subject_id, n_wrist_acc).reshape((n_wrist_acc,1))
    #lbl_acc = signal.resample(lbl, n_wrist_acc)
    batch_size = sf_chest/sf_ACC
    lbl_m = np.zeros((n_wrist_acc,1))
    for i in range(n_wrist_acc):
        lbl_m[i] = (stats.mode(lbl[round(i * batch_size) : round((i + 1) * batch_size) - 1]))[0].squeeze()
    lbl_acc = lbl_m 
    print("lbl_acc.shape :", lbl_acc.shape)

    
    sid_bvp = np.repeat(subject_id, n_wrist_bvp).reshape((n_wrist_bvp,1))
    #lbl_bvp = signal.resample(lbl, n_wrist_bvp)
    batch_size = sf_chest/sf_BVP
    lbl_m = np.zeros((n_wrist_bvp,1))
    for i in range(n_wrist_bvp):
        lbl_m[i] = (stats.mode(lbl[round(i * batch_size) : round((i + 1) * batch_size) - 1]))[0].squeeze()
    lbl_bvp = lbl_m 
    print("lbl_bvp.shape :", lbl_bvp.shape)
    
    sid_eda_temp = np.repeat(subject_id, n_wrist_eda).reshape((n_wrist_eda,1))
    #lbl_eda_temp = signal.resample(lbl, n_wrist_eda)
    batch_size = sf_chest/sf_EDA
    lbl_m = np.zeros((n_wrist_eda,1))
    for i in range(n_wrist_eda):
        lbl_m[i] = (stats.mode(lbl[round(i * batch_size) : round((i + 1) * batch_size) - 1]))[0].squeeze()
    lbl_eda_temp = lbl_m 
    print("lbl_eda_temp.shape :", lbl_eda_temp.shape)
    
    
    data1 = np.concatenate((sid_acc, wrist_acc, lbl_acc), axis=1)
    data2 = np.concatenate((sid_bvp, wrist_bvp, lbl_bvp), axis=1)
    data3 = np.concatenate((sid_eda_temp, wrist_eda, wrist_temp, lbl_eda_temp), axis=1)

    return data1, data2, data3



def merge_wrist_data():
    for i, sid in enumerate(ids):
        file = DATA_PATH + 'S' + str(sid) + '/S' + str(sid) + '.pkl'
        print("")
        print("processing file: ", file)
        if i == 0: 
            md1, md2, md3 = pkl_to_np_wrist(file, sid)
            print("md1.shape: ", md1.shape)
            print("md2.shape: ", md2.shape)
            print("md3.shape: ", md3.shape)
        else:
            last_subj1, last_subj2, last_subj3 = pkl_to_np_wrist(file, sid)
            print("last_subj1.shape: ",last_subj1.shape)
            print("last_subj2.shape: ",last_subj2.shape)
            print("last_subj3.shape: ",last_subj3.shape)
            md1 = np.concatenate((md1, last_subj1), axis=0)
            md2 = np.concatenate((md2, last_subj2), axis=0)
            md3 = np.concatenate((md3, last_subj3), axis=0)
            print("md1.shape: ", md1.shape)
            print("md2.shape: ", md2.shape)
            print("md3.shape: ", md3.shape)
            
    fn_merged1 = 'data/subj_merged_acc_w.pkl'
    fn_merged2 = 'data/subj_merged_bvp_w.pkl'
    fn_merged3 = 'data/subj_merged_eda_temp_w.pkl'
    all_columns1 = ['sid', 'w_acc_x' , 'w_acc_y', 'w_acc_z', 'label']
    all_columns2 = ['sid', 'bvp', 'label']
    all_columns3 = ['sid', 'w_eda' , 'w_temp', 'label']
    pd.DataFrame(md1, columns=all_columns1).to_pickle(fn_merged1)
    pd.DataFrame(md2, columns=all_columns2).to_pickle(fn_merged2)
    pd.DataFrame(md3, columns=all_columns3).to_pickle(fn_merged3)
    
        


# convert data from pickle dictionary format to dataframe
def pkl_to_np_chest(filename, subject_id):
    unpickled_df = pd.read_pickle(filename)
    chest_acc = unpickled_df["signal"]["chest"]["ACC"]
    chest_ecg = unpickled_df["signal"]["chest"]["ECG"]
    chest_emg = unpickled_df["signal"]["chest"]["EMG"]
    chest_eda = unpickled_df["signal"]["chest"]["EDA"]
    chest_temp = unpickled_df["signal"]["chest"]["Temp"]
    chest_resp = unpickled_df["signal"]["chest"]["Resp"]
    lbl = unpickled_df["label"].reshape(unpickled_df["label"].shape[0],1)
    sid = np.full((lbl.shape[0],1), subject_id)
    chest_all = np.concatenate((sid, chest_acc,chest_ecg, chest_emg,chest_eda, chest_temp,chest_resp, lbl), axis=1)
    #new_fn = 'chest_all_' + filename
    #pd.DataFrame(chest_all, columns=['acc1', 'acc2', 'acc3', 'ecg', 'emg', 'eda', 'temp', 'resp', 'label']).to_pickle(new_fn)
    return chest_all




def merge_chest_data():
    for i, sid in enumerate(ids):
        file = DATA_PATH + 'S' + str(sid) + '/S' + str(sid) + '.pkl'
        print("")
        print("processing file: ", file)
        if i == 0: 
            merged_data = pkl_to_np_chest(file, sid)
            print("merged_data.shape: ", merged_data.shape)
        else:
            last_subj = pkl_to_np_chest(file, sid)
            print("last_subj.shape: ",last_subj.shape)
            merged_data = np.concatenate((merged_data, last_subj), axis=0)
            print("merged_data.shape: ", merged_data.shape)
            
    fn_merged = 'data/merged_chest.pkl'
    pd.DataFrame(merged_data, columns=chest_columns).to_pickle(fn_merged)
    



def filter_chest_data():
    df = pd.read_pickle(("merged_chest.pkl"))
    df_fltr = df[df["label"].isin([1,2,3])]
    df_fltr = df_fltr[df_fltr["temp"]>0]
    pd.DataFrame(df_fltr, columns=chest_columns).to_pickle("data/merged_chest_fltr.pkl")



def preprocess():
    merge_wrist_data()
    merge_chest_data()
    filter_chest_data()


#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Dense, Flatten, Reshape, BatchNormalization
from keras.layers import Conv1D, UpSampling1D, MaxPooling1D, AveragePooling1D
from keras.models import Model, Sequential
from keras.layers.advanced_activations import ReLU
from keras.optimizers import RMSprop
import tensorflow as tf


C_PATH = "data/merged_chest_fltr.pkl"
W1_PATH = "data/subj_merged_bvp_w.pkl"
W2_PATH = "data/subj_merged_eda_temp_w.pkl"
feat_sf700 = ['ecg', 'emg', 'eda', 'temp', 'resp']
feat_sf64 = ['bvp']
feat_sf4 = ['w_eda', 'w_temp']
sf_chest = 700 #sampling frequency for measurements collected from chest device
sf_BVP = 64
sf_EDA = 4
sf_TEMP = 4

window = 0.25 # sampling window


class autoencoder:
    def __init__(self, **kwargs):
        self.df_c = pd.read_pickle(C_PATH)
        self.df_w1 = pd.read_pickle(W1_PATH)
        self.df_w2 = pd.read_pickle(W2_PATH)
        self.df_w1 = self.df_w1[self.df_w1["label"].isin([1,2,3])]
        self.df_w2 = self.df_w2[self.df_w2["label"].isin([1,2,3])]
        
        self.batch_size = int(sf_chest*window) 
        self.batch_size_bvp =  int(sf_BVP*window) 
        self.batch_size_eda =  int(sf_EDA*window) 
        self.batch_size_temp =  int(sf_TEMP*window) 

        self.ids = self.df_c["sid"].unique().astype(int)
        self.K = len(self.df_c["label"].unique())
        
    def one_hot_enc(self, r, k):
        new_r = np.zeros((r.shape[0],k))
        for i, val in enumerate(r):
            new_r[i, val-1] = 1

        return new_r
    
    def get_data(self, test_id, v_batch_size, v_feat_list, df):
        
        cnt=0
        
        for j in self.ids:
            df_s = df[df["sid"] == j]

            n = (len(df_s)//v_batch_size)*v_batch_size
            df_s = df_s[:n]
            s = StandardScaler().fit_transform(df_s[v_feat_list])
            s = s.reshape(int(s.shape[0]/v_batch_size), s.shape[1],  v_batch_size)

            lbl_m = np.zeros((s.shape[0],1))
            lbl = df_s["label"].values.astype(int)
            for i in range(s.shape[0]):
                lbl_m[i] = int((stats.mode(lbl[i * v_batch_size : (i + 1) * v_batch_size - 1]))[0].squeeze())
            y_k = lbl_m.astype(int)
            s_y = self.one_hot_enc(lbl_m.astype(int), self.K).astype(int)
            #print("subject ", j)
            #print(s.shape, s_y.shape)
            if j==test_id:
                x_test = s
                y_test = s_y
                yk_test = y_k
            else:
                if cnt:
                    merged = np.concatenate((merged, s), axis=0)
                    merged_y = np.concatenate((merged_y, s_y), axis=0)
                    merged_yk = np.concatenate((merged_yk, y_k), axis=0)
                else:
                    merged = s
                    merged_y = s_y
                    merged_yk = y_k
                cnt +=1


        print ("merged train:", merged.shape, merged_y.shape)
        print ("merged test :", x_test.shape, y_test.shape)
        return merged, merged_y, x_test, y_test, merged_yk, yk_test

    # train and store autoencoder model for chest modalities
    def train_model_c(self):   
        # leave one out method
        scores = []

        for sid in self.ids:
            x_train, y_train, x_test, y_test, yk, yk_test = self.get_data (test_id =sid, 
                                                                       v_batch_size=sf_chest, 
                                                                       v_feat_list=feat_sf700, 
                                                                       df=self.df_c)

            encoder, model = self.autoenc_model_chest(v_batch_size=sf_chest, n_feat=len(feat_sf700))
            model.compile(optimizer=RMSprop(lr=0.00025), loss="mse")
            history = model.fit(x_train, x_train, epochs=10)
            m_name = "trained_models/c/encoder_loso"+str(sid)+".h5"

            encoder.save(m_name)
            print("saved ", m_name)
            
    def autoenc_model_w1(self, v_batch_size, n_feat):
    
        input_sig = Input(shape=(n_feat, v_batch_size))
        x = Conv1D(v_batch_size,6, activation='relu', padding='same')(input_sig)
        x1 = BatchNormalization()(x)
        x2 = Conv1D(v_batch_size,3, activation='relu', padding='same')(x1)
        flat = Flatten()(x2)

        encoded = Dense(40, activation='relu')(flat)

        encoder = Model(input_sig, encoded)

        d1 = Dense(v_batch_size*n_feat)(encoded)
        d2 = Reshape((n_feat,v_batch_size))(d1)
        d3 = Conv1D(v_batch_size,3, activation='relu', padding='same')(d2)
        d4 = BatchNormalization()(d3)
        d5 = Conv1D(v_batch_size,6, activation='sigmoid', padding='same',  name='reconst_output')(d4)

        model= Model(input_sig, d5)

        return encoder, model
    
    def autoenc_model_w2(self, v_batch_size, n_feat):
        
        input_sig = Input(shape=(n_feat, v_batch_size))
        x = Conv1D(v_batch_size,4, activation='relu', padding='same')(input_sig)

        x1 = BatchNormalization()(x)
        flat = Flatten()(x1)
        encoded = Dense(4, activation='relu')(flat)

        encoder = Model(input_sig, encoded)

        d1 = Dense(v_batch_size*n_feat)(encoded)
        d2 = Reshape((n_feat,v_batch_size))(d1)
        d5 = Conv1D(v_batch_size,4, activation='sigmoid', padding='same',  name='reconst_output')(d2)

        model= Model(input_sig, d5)

        return encoder, model   
    
    def autoenc_model_chest(self, v_batch_size, n_feat):
    
        input_sig = Input(shape=(n_feat, v_batch_size))
        x = Conv1D(v_batch_size,6, activation='relu', padding='same')(input_sig)

        x1 = BatchNormalization()(x)
        x2 = Conv1D(v_batch_size,3, activation='relu', padding='same')(x1)
        flat = Flatten()(x2)
        
        encoded = Dense(80, activation='relu')(flat)

        encoder = Model(input_sig, encoded)

        d1 = Dense(v_batch_size*n_feat)(encoded)
        d2 = Reshape((n_feat,v_batch_size))(d1)
        d3 = Conv1D(v_batch_size,3, activation='relu', padding='same')(d2)
        d4 = BatchNormalization()(d3)
        d5 = Conv1D(v_batch_size,6, activation='sigmoid', padding='same')(d4)

        model= Model(input_sig, d5)

        return encoder, model
    
    def extract_features (self):
        
        for sid in self.ids:
            print("============= test subject " +str(sid)+ " ==================")
            x_train, y_train, x_test, y_test, yk, yk_test = self.get_data (test_id = sid,
                                                                           v_batch_size=sf_chest,
                                                                           v_feat_list=feat_sf700, 
                                                                           df=self.df_c)
            x_trainw1, y_trainw1, x_testw1, y_testw1, yk1w1, yk_test1w1 = self.get_data (test_id = sid,
                                                                           v_batch_size=sf_BVP,
                                                                           v_feat_list=feat_sf64, 
                                                                           df=self.df_w1)
            x_trainw2, y_trainw2, x_testw2, y_testw2, yk2w2, yk_test2w2 = self.get_data (test_id = sid,
                                                                           v_batch_size=sf_EDA,
                                                                           v_feat_list=feat_sf4, 
                                                                           df=self.df_w2)

            encoderw1, modelw1 = self.autoenc_model_w1(v_batch_size=sf_BVP, n_feat=len(feat_sf64))
            modelw1.compile(optimizer=RMSprop(lr=0.00025), loss="mse")
            history = modelw1.fit(x_trainw1, x_trainw1, epochs=4)

            emb_trainw1 = encoderw1.predict(x_trainw1)
            emb_testw1 = encoderw1.predict(x_testw1)

            encoderw2, modelw2 = self.autoenc_model_w2(v_batch_size=sf_EDA, n_feat=len(feat_sf4))
            modelw2.compile(optimizer=RMSprop(lr=0.00025), loss="mse")
            history = modelw2.fit(x_trainw2, x_trainw2, epochs=4)

            emb_trainw2 = encoderw2.predict(x_trainw2)
            emb_testw2 = encoderw2.predict(x_testw2)

            m_name = "trained_models/c/encoder_loso"+str(sid)+".h5"
            encoder = tf.keras.models.load_model(m_name)
            print("loaded: ", m_name)

            emb_train = encoder.predict(x_train)
            emb_test = encoder.predict(x_test)

            print("emb_trainw1.shape: ", emb_trainw1.shape)
            print("emb_trainw2.shape: ", emb_trainw2.shape)
            print("emb_train.shape: ", emb_train.shape)

            print("emb_testw1.shape: ", emb_testw1.shape)
            print("emb_testw2.shape: ", emb_testw2.shape)
            print("emb_test.shape: ", emb_test.shape)

            last_inx_train = int(min(emb_trainw1.shape[0], emb_trainw2.shape[0], emb_train.shape[0]))
            last_inx_test = int(min(emb_testw1.shape[0], emb_testw2.shape[0], emb_test.shape[0]))

            emb_train_all = np.concatenate ((emb_train[:last_inx_train], emb_trainw1[:last_inx_train], emb_trainw2[:last_inx_train], yk[:last_inx_train]), axis=1)
            emb_test_all = np.concatenate ((emb_test[:last_inx_test], emb_testw1[:last_inx_test], emb_testw2[:last_inx_test], yk_test[:last_inx_test]), axis=1)

            train_feat_file = "features/train/feat_loso"+str(sid)+".pkl"
            test_feat_file = "features/test/feat_loso"+str(sid)+".pkl"
            pd.DataFrame(emb_train_all).to_pickle(train_feat_file)
            pd.DataFrame(emb_test_all).to_pickle(test_feat_file)
    
        


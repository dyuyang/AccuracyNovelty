from NovResysClassifier import NovResysClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import time
import random
import tensorflow as tf
from scipy import spatial
import os
import scipy
import copy
from tqdm import * 
from sklearn.preprocessing import LabelEncoder

DATA_DIR='./ml-100k/'

class MovieLens:
    def load_raw_data(self):
        f=tf.gfile.Open(DATA_DIR + 'u.data',"r")
        self.df_rating = pd.read_csv(
            f,
            sep='\t',
            names=['uid', 'itemid', 'rating', 'time'])
        
        f=tf.gfile.Open(DATA_DIR + 'u.user',"r")
        self.df_userinfo = pd.read_csv(
            f,
            sep='|',
            names=['uid', 'age', 'sex', 'occupation', 'zip_code'])
        list_item_attr = [
            'itemid', 'title', 'rel_date', 'video_rel_date', 'imdb_url',
            "unknown", "Action", "Adventure", "Animation", "Children's",
            "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
            "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller",
            "War", "Western"
        ]
        f=tf.gfile.Open(DATA_DIR + 'u.item',"r")
        self.df_iteminfo = pd.read_csv(
            f,
            sep='|',
            names=list_item_attr)
        self.df_userinfo = self.df_userinfo.fillna(0)
        self.df_iteminfo = self.df_iteminfo.fillna(0)

    def minmax_scaler(self, list_attr, df):
        for attr in list_attr:
            df[attr] = df[attr] - min(df[attr])
    def feature_engineering(self):

        ##iteminfo
        df_all = self.df_iteminfo
        df_date = df_all["rel_date"]
        df_date = pd.to_datetime(df_date)
        df_all["year"] = df_date.apply(lambda x: x.year)
        df_all["month"] = df_date.apply(lambda x: x.month)
        df_all["day"] = df_date.apply(lambda x: x.day)
        df_all.drop(
            ["rel_date", "imdb_url", "video_rel_date", "title"],
            axis=1,
            inplace=True)
        self.minmax_scaler(["year", "month", "day"], df_all)
        df_numeric = df_all.select_dtypes(exclude=['object'])
        df_obj = df_all.select_dtypes(include=['object']).copy()
        for c in df_obj:
            df_obj[c] = (pd.factorize(df_obj[c])[0])
        self.df_iteminfo = pd.concat([df_numeric, df_obj], axis=1)

        df_all = self.df_userinfo
        self.minmax_scaler(["age"], df_all)
        df_numeric = df_all.select_dtypes(exclude=['object'])
        df_obj = df_all.select_dtypes(include=['object']).copy()
        for c in df_obj:
            df_obj[c] = (pd.factorize(df_obj[c])[0])
        self.df_userinfo = pd.concat([df_numeric, df_obj], axis=1)

    def __init__(self):
        self.rating_threshold = 3
        self.load_raw_data()
        self.df_iteminfo["itemid"]=self.df_iteminfo["itemid"]
        self.df_userinfo["uid"]=self.df_userinfo["uid"]
        self.df_rating["itemid"]=self.df_rating["itemid"]
        self.df_rating["uid"]=self.df_rating["uid"]
        self.feature_engineering()
        
        self.user_numerical_attr =  ["age"]
        self.item_numerical_attr = ["year", "month", "day"]




def early_stopping(train_loss_history,val_loss_history):
    if len(train_loss_history)!=0:
        print(train_loss_history[-1])
    if len(train_loss_history)!=0 and train_loss_history[-1]<0.48:
        return 1,len(train_loss_history)-1
    else:
        return 0,-1

movielens =MovieLens()

user_num_inds=[]
item_num_inds=[]
for idx,feat in enumerate(movielens.df_userinfo.columns):
    if feat in movielens.user_numerical_attr:
        user_num_inds.append(idx)
        
for idx,feat in enumerate(movielens.df_iteminfo.columns):
    if feat in movielens.item_numerical_attr:
        item_num_inds.append(idx)
    
resys=NovResysClassifier(0,0,
                         movielens.df_userinfo.values,0,user_num_inds,
                         movielens.df_iteminfo.values,0,item_num_inds,
                        movielens.df_rating.values)

resys.preprocess()

resys.precalculate()

resys.fit(epochs=300,early_stop_method=early_stopping,val_size=5/7)




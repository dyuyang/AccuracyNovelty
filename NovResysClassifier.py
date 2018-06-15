
# coding: utf-8

# In[1]:


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


class NovResysClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 NOVELTY_TYPE,
                 DISTANT_TYPE,
                 user_info=None,
                 uid_ind=None,
                 user_num_inds=None,
                 item_info=None,
                 itemid_ind=None,
                 item_num_inds=None,
                 rating_info=None,
                 rating_threshold=3,
                 test_size=0.3,
                 random_state=2018):
        self.NOVELTY_TYPE=NOVELTY_TYPE
        self.DISTANT_TYPE=DISTANT_TYPE
        self.user_info = user_info
        self.uid_ind = uid_ind
        self.user_num_inds = user_num_inds
        self.item_info = item_info
        self.itemid_ind = itemid_ind
        self.item_num_inds = item_num_inds
        self.rating_info = rating_info
        self.rating_threshold = rating_threshold
        self.ratio = test_size
        self.seed = random_state
        
        for ind in range(np.shape(self.user_info)[1]):
            le=LabelEncoder()
            le.fit(self.user_info[:,ind])
            if ind == uid_ind:
                self.uid_le=le
            self.user_info[:,ind]=le.transform(self.user_info[:,ind])
            
        for ind in range(np.shape(self.item_info)[1]):
            le=LabelEncoder()
            le.fit(self.item_info[:,ind])
            if ind == itemid_ind:
                self.itemid_le=le
            self.item_info[:,ind]=le.transform(self.item_info[:,ind])
            
        self.rating_info[:,0]=self.uid_le.transform(self.rating_info[:,0])
        self.rating_info[:,1]=self.itemid_le.transform(self.rating_info[:,1])
        self.first_fit=1
            
    def _split_history(self,dic,ratio):
        seed = self.seed
        dic1 = {}
        dic2 = {}
        for ky in dic:
            lst = dic[ky]
            lenoflist = len(lst)
            if lenoflist != 0:
                random.Random(seed).shuffle(lst)
                dic1[ky] = lst[:int(ratio * lenoflist)]
                dic2[ky] = lst[int(ratio * lenoflist):]
            else:
                dic1[ky] = []
                dic2[ky] = []
        return dic1, dic2

    def _merge_history(self,dic1, dic2):
        return {ky: list(set(dic1[ky]) | set(dic2[ky])) for ky in dic1}

    def _reverse_history(self,dict_byuid):
        result = {itemid: [] for itemid in self.itemid_list}
        for uid in dict_byuid:
            for itemid in dict_byuid[uid]:
                result[itemid].append(uid)
        return result

    def preprocess(self):

        self.uid_list = self.user_info[:,self.uid_ind]
        self.itemid_list = self.item_info[:,self.itemid_ind]

        self.all_posuser_byitemid = {itemid: [] for itemid in self.itemid_list}
        self.all_positem_byuid = {uid: [] for uid in self.uid_list}
        self.all_neguser_byitemid = {itemid: [] for itemid in self.itemid_list}
        self.all_negitem_byuid = {uid: [] for uid in self.uid_list}
        sz1 = len(self.uid_list)
        sz2 = len(self.itemid_list)
        df_all = self.rating_info

        sz = len(self.rating_info)

        self.ratings_byitemid = [[0.0 for uid in self.uid_list]
                                 for itemid in self.itemid_list]
        
        print('preprocess rating info...')
        time.sleep(1)
        for  row in tqdm(self.rating_info):

            rating = row[2]
            uid = int(row[0])
            itemid = int(row[1])

            self.ratings_byitemid[itemid][uid] = rating
            #self.rating_bypair[uid][itemid] = rating
            if rating > self.rating_threshold:
                self.all_posuser_byitemid[itemid].append(uid)
                self.all_positem_byuid[uid].append(itemid)
            else:
                self.all_neguser_byitemid[itemid].append(uid)
                self.all_negitem_byuid[uid].append(itemid)
        
        print('preprocess rating info succeed.')

        print('preprocess field info...')
        self._USER_SIZE_ONLY_NUM = len(self.user_num_inds)
        self._USER_SIZE_OF_FIELDS = []
        for ind in range(np.shape(self.user_info)[1]):
            if ind in self.user_num_inds:
                self._USER_SIZE_OF_FIELDS.append(1)
            else:
                self._USER_SIZE_OF_FIELDS.append(
                    len(np.unique(self.user_info[:,ind])))
                
        self._USER_SIZE = len(self._USER_SIZE_OF_FIELDS)
        self._USER_SIZE_OF_MASK_FIELDS = self._USER_SIZE_OF_FIELDS[:-self.
                                                                   _USER_SIZE_ONLY_NUM]
        self._USER_SIZE_BIN = sum(self._USER_SIZE_OF_FIELDS)

        self._ITEM_SIZE_ONLY_NUM = len(self.item_num_inds)

        self._ITEM_SIZE_OF_FIELDS = []
        for ind in range(np.shape(self.item_info)[1]):
            if ind in self.item_num_inds:
                self._ITEM_SIZE_OF_FIELDS.append(1)
            else:
                self._ITEM_SIZE_OF_FIELDS.append(
                    len(np.unique(self.item_info[:,ind])))

        self._ITEM_SIZE = len(self._ITEM_SIZE_OF_FIELDS)
        self._ITEM_SIZE_OF_MASK_FIELDS = self._ITEM_SIZE_OF_FIELDS[:-self.
                                                                   _ITEM_SIZE_ONLY_NUM]
        self._ITEM_SIZE_BIN = sum(self._ITEM_SIZE_OF_FIELDS)
        
        print('preprocess field info succeed.')
        
        print('build index pools...')

        self.train_positem_byuid, self.test_positem_byuid = self._split_history(
            self.all_positem_byuid, self.ratio)

        self.train_posuser_byitemid, self.test_posuser_byitemid = self._reverse_history(
            self.train_positem_byuid), self._reverse_history(
                self.test_positem_byuid)

        self.train_negitem_byuid, self.test_negitem_byuid = self._split_history(
            self.all_negitem_byuid, self.ratio)

        self.train_neguser_byitemid, self.test_neguser_byitemid = self._reverse_history(
            self.train_negitem_byuid), self._reverse_history(
                self.test_negitem_byuid)

        self.train_rateduser_byitemid = self._merge_history(
            self.train_posuser_byitemid, self.train_neguser_byitemid)

        self.test_rateduser_byitemid = self._merge_history(
            self.test_posuser_byitemid, self.test_neguser_byitemid)

        self.train_rateditem_byuid = self._merge_history(self.train_positem_byuid,
                                                     self.train_negitem_byuid)

        self.test_rateditem_byuid = self._merge_history(self.test_positem_byuid,
                                                    self.test_negitem_byuid)

        print('build index pools succeed')
        
        
        print('reorder feature columns...')
        for ind in range(np.shape(self.user_info)[1]):
            if ind in self.user_num_inds:
                tmp = self.user_info[:,ind].copy()
                tmp=np.reshape(tmp,(-1,1))
                self.user_info=np.delete(self.user_info,[ind], axis=1)
                #print(np.shape(tmp),np.shape(self.user_info))
                self.user_info=np.concatenate((self.user_info,tmp),axis=1)
        for ind in range(np.shape(self.item_info)[1]):
            if ind in self.item_num_inds:
                tmp = self.item_info[:,ind].copy()
                tmp=np.reshape(tmp,(-1,1))
                self.item_info=np.delete(self.item_info,[ind], axis=1)
                self.item_info=np.concatenate((self.item_info,tmp),axis=1)    
        print('reorder feature columns succeed.')
        
    def _set_distant(self, i, j):
        users_i = self.train_rateduser_byitemid[i]
        users_j = self.train_rateduser_byitemid[j]
        if (len(users_j) != 0):
            return 1 - 1.0 * len(set(users_i) & set(users_j)) / len(users_j)
        else:
            return 1.0

    def _cosine_distant(self, i, j):
        vec_i = self.ratings_byitemid[i]
        vec_j = self.ratings_byitemid[j]
        return 1 - np.dot(vec_i, vec_j)

    def _distant(self, i, j):
        if self.DISTANT_TYPE == 0:
            return self._set_distant(i, j)
        else:
            return self._cosine_distant(i, j)

    def novelty(self, uid, item_i):
        items_byu = self.train_rateditem_byuid[uid]
        if self.NOVELTY_TYPE == 0:
            return np.mean(
                [self._distant_mat[item_i][item_j] for item_j in items_byu])
        else:
            return -np.log2(
                len(self.train_rateduser_byitemid[item_i]
                    ) / len(self.uid_list) + pow(10, -9))

    def _item_vectorize(self, itemid):
        return self.item_info[itemid]

    def _user_vectorize(self, uid):
        return self.user_info[uid]

    def _save_vec(self):
        self.uid_to_vec = {}
        self.itemid_to_vec = {}
        sz = len(self.uid_list)
        print('precalculate user vector...')
        time.sleep(0.5)
        for uid in tqdm(self.uid_list):
            self.uid_to_vec[uid] = self._user_vectorize(uid)
        time.sleep(0.5)
        print('precalculate user vector succeed.')
        print('precalculate item vector...')
        time.sleep(0.5)
        sz = len(self.itemid_list)
        for itemid in tqdm(self.itemid_list):
            self.itemid_to_vec[itemid] = self._item_vectorize(itemid)
        time.sleep(0.5)
        print('precalculate item vector succeed.')   
          
    def _save_distance(self):
        new_ratings_byitemid=[]
        print('precalculate rating vector...')
        time.sleep(0.5)
        for itemid in tqdm(self.itemid_list):
            vec=[0.0 for uid in self.uid_list]
            for uid in self.train_rateduser_byitemid[itemid]:
                vec[uid]=self.ratings_byitemid[itemid][uid]
            vec=np.array(vec)+pow(10,-9)
            new_ratings_byitemid.append(vec/ np.linalg.norm(vec))
        time.sleep(0.5)
        print('precalculate rating vector succeed')
        
        self.ratings_byitemid = new_ratings_byitemid
        self._distant_mat=[]
        print('precalculate distance...')
        time.sleep(0.5)
        for index_i,i in enumerate(tqdm(self.itemid_list)):
            self._distant_mat.append([])
            for index_j,j in enumerate(self.itemid_list):
                if index_j>index_i:
                    self._distant_mat[index_i].append(self._distant(index_i,index_j))
                elif index_j==index_i:
                    self._distant_mat[index_i].append(0)
                else:
                    self._distant_mat[index_i].append(self._distant_mat[index_j][index_i])
        time.sleep(0.5)
        print('precalculate distance succeed.')      
    def precalculate(self):
        self._save_vec()
        self._save_distance()
    
    
    def _get_novelty_distribution(self, u):
        list_positemid = self.train_positem_byuid[u]
        list_negitemid = self.train_negitem_byuid[u]
        positem_novdistr = [
            pow(self.novelty(u, itemid), self.beta)
            for itemid in list_positemid
        ]
        negitem_novdistr = [1.0 for itemid in list_negitemid]
        return positem_novdistr / np.sum(
            positem_novdistr), negitem_novdistr / np.sum(negitem_novdistr)

    def _load_distribution(self):
        pos_distr=[[] for uid in self.uid_list]
        neg_distr=[[] for uid in self.uid_list]
        for uid in self.uid_list:
            #print('load the novelty distribution of user', uid)
            pos_distr[uid], neg_distr[
                uid] = self._get_novelty_distribution(uid)
        return pos_distr,neg_distr

    def _predict_mat(self, uid_list, itemid_list):

        user_batch = [self.uid_to_vec[uid] for uid in uid_list]

        item_batch = []

        for itemid in itemid_list:
            item_batch.append(self.itemid_to_vec[itemid])

        label_batch = [[1] * len(itemid_list) for uid in uid_list]
        #print(np.shape(user_batch),np.shape(item_batch),np.shape(label_batch))
        prob_matrix = self.prob.eval(
            session=self.sess,
            feed_dict={
                self.user_input: user_batch,
                self.item_input: item_batch,
                self.label: label_batch
            })

        return prob_matrix

    def _predict_mat_by_queue(self, uid_list, itemid_list):
        sz = len(itemid_list)
        batch_sz = 10000
        bins = int(sz / batch_sz)
        ret = []
        for idx in range(bins):
            #print('_predict_mat_by_queue %d/%d' % (idx, bins))
            tmp = self._predict_mat(
                uid_list, itemid_list[idx * batch_sz:(idx + 1) * batch_sz])
            if ret != []:
                ret = np.concatenate((ret, tmp), axis=1)
            else:
                ret = tmp

        tmp = self._predict_mat(uid_list, itemid_list[bins * batch_sz:])

        if ret != []:
            ret = np.concatenate((ret, tmp), axis=1)
        else:
            ret = tmp
        return ret

    def eval_performance(self):

        self.prob_by_uitem = self._predict_mat_by_queue(self.uid_list, self.itemid_list)

        self.uid_to_recomm = self._base_recommend(self.prob_by_uitem,
                                                 self.top_N)
        #print(uid_list)
        #print(uid_to_recomm)
        acc = self._print_accuracy(self.uid_to_recomm, self.prob_by_uitem)
        reward0, reward1, agg_div, entro_div = self._print_diversity(
            self.uid_to_recomm)
        return reward0, reward1, agg_div, entro_div

    def _print_accuracy(self, uid_to_recomm, prob_by_uitem):
        acc = 0
        for uid in self.uid_list:
            if len(self.test_positem_byuid[uid]) < self.top_N:
                continue
                #pass
            positem_test = list(self.test_positem_byuid[uid])

            if len(set(positem_test) & set(uid_to_recomm[uid])) != 0:
                acc += 1
        return acc / len(uid_to_recomm)

    def _base_recommend(self, prob_by_uitem, top_N):
        uid_to_recomm = {}
        for uid in self.uid_list:
            if len(self.test_positem_byuid[uid]) < self.top_N:
                continue
                #pass
            prob_row = prob_by_uitem[uid]
            prob_arr = list(zip(self.itemid_list, prob_row))
            prob_arr = sorted(prob_arr, key=lambda d: -d[1])
            cnt = 0
            uid_to_recomm[uid] = []
            for pair in prob_arr:
                itemid = pair[0]
                if itemid not in self.train_rateditem_byuid[uid]:
                    uid_to_recomm[uid].append(itemid)
                    cnt += 1
                    if cnt == top_N:
                        break
        return uid_to_recomm

    def _print_diversity(self, uid_to_recomm):
        avg_reward0 = 0.0
        avg_reward1 = 0.0
        agg_div = 0.0
        enp_div = 0.0

        cnt = 0
        for uid in uid_to_recomm:
            reward0 = 0.0
            reward1 = 0.0
            for itemid in uid_to_recomm[uid]:
                if (itemid in self.test_positem_byuid[uid]):
                    nov = self.novelty(uid, itemid)
                    if nov == np.inf or np == -np.inf:
                        nov = 0
                    if nov != 0:
                        nov0 = pow(nov, 0)
                        nov1 = pow(nov, 1)
                        cnt += 1
                    reward0 = max(reward0, nov0)
                    reward1 = max(reward1, nov1)
            avg_reward0 += reward0
            avg_reward1 += reward1

        if avg_reward0 != 0:
            avg_reward0 /= len(uid_to_recomm)
        if avg_reward1 != 0:
            avg_reward1 /= cnt

        recomm_set = set()
        for uid in uid_to_recomm:
            recomm_set = recomm_set | set(uid_to_recomm[uid])
        agg_div = len(recomm_set) / len(uid_to_recomm) / self.top_N

        itemid_to_recomuser = {}

        for uid in uid_to_recomm:
            for itemid in uid_to_recomm[uid]:
                if itemid not in itemid_to_recomuser:
                    itemid_to_recomuser[itemid] = 0
                itemid_to_recomuser[itemid] += 1

        s = 0
        for itemid in itemid_to_recomuser:
            s += itemid_to_recomuser[itemid]

        for itemid in itemid_to_recomuser:
            probb = itemid_to_recomuser[itemid] / s + pow(10, -9)
            enp_div += -(np.log2(probb) * probb)

        #print('over diver %f'%(time.time()-t1))
        print(
            'Diversity: accuracy=%.5f novelty=%.5f aggdiv=%.5f entropydiv=%.5f'
            % (avg_reward0, avg_reward1, agg_div, enp_div))
        return avg_reward0, avg_reward1, agg_div, enp_div

    def _train_a_batch(self, iter):
        
        loss_all = 0

        user_batch = []
        item_batch = []
        label_batch = []
        list_positemid = []
        uid_list = []
        list_label = []
        list_negitemid = []

        for i in range(self.batch_size):
            uid = 0
            while (True):
                uid = self.rng.randint(1, self.NUM_USERS)
                if ((uid in self.uid_list)
                        and len(self.train_positem_byuid[uid]) != 0
                        and len(self.train_negitem_byuid[uid]) != 0):
                    break
            uid_list.append(uid)
            
        for uid in uid_list:
            pos_itemid = self.rng.choice(
                self.train_positem_byuid[uid], p=self.pos_distr[uid])
            list_positemid.append(pos_itemid)
            list_label.append(1)
            user_batch.append(self._user_vectorize(uid))
            pos_itemvec = self._item_vectorize(pos_itemid)
            item_batch.append(pos_itemvec)

        prob_by_uitem = self._predict_mat(uid_list, list_positemid)

        neg_itemset = set()
        neg_index = {}
        for uid in uid_list:
            neg_itemset = neg_itemset | set(self.train_negitem_byuid[uid])
        for index, neg_item in enumerate(neg_itemset):
            neg_index[neg_item] = index
        neg_itemset = list(neg_itemset)
        neg_prob_by_uitem = self._predict_mat(uid_list, neg_itemset)

        violator_cnt = 0
        for i, uid in enumerate(uid_list):
            neg_itemid = -1
            pos_itemid = list_positemid[i]
            pos_prob = prob_by_uitem[i][i]
            for k in range(self.LIMIT):
                neg_itemid = self.rng.choice(
                    self.train_negitem_byuid[uid],
                    p=self.neg_distr[uid])
                neg_prob = neg_prob_by_uitem[i][neg_index[neg_itemid]]
                if neg_prob >= pos_prob and neg_prob != 0:
                    break
                else:
                    neg_itemid = -1

            if neg_itemid != -1:
                violator_cnt += 1
                list_label.append(-1)
                user_batch.append(self._user_vectorize(uid))
                neg_itemvec = self._item_vectorize(neg_itemid)
                item_batch.append(neg_itemvec)

        label_batch = [[1] * len(user_batch) for j in range(len(user_batch))]
        for i, label in enumerate(list_label):
            label_batch[i][i] = label

        #print(np.shape(user_batch),np.shape(item_batch))
        feed_dict = {
            self.user_input: user_batch,
            self.item_input: item_batch,
            self.label: label_batch
        }
        [_optimize, _loss] = self.sess.run(
            [self.optimize, self.loss], feed_dict=feed_dict)
        return _loss

    def _cal_val_loss(self):
        p=[]
        q=[]
        prob_by_uitem = self._predict_mat(self.uid_list, self.itemid_list)
        for uid in self.uid_list:
            for itemid in self.val_positem_byuid[uid]:
                prob=prob_by_uitem[uid][itemid]
                p.append(1)
                q.append(prob)
            for itemid in self.val_negitem_byuid[uid]:
                prob=prob_by_uitem[uid][itemid]
                p.append(0)
                q.append(prob)
        q=[x+1e-20 for x in q]
        return scipy.stats.entropy(p, q)

    def _es_train(self, es_stage):
        
        if es_stage==1:
            self.train_positem_byuid_buffer=self.train_positem_byuid.copy()
            self.train_negitem_byuid_buffer=self.train_negitem_byuid.copy()
            
            self.train_positem_byuid,self.val_positem_byuid=self._split_history(self.train_positem_byuid,5/7)
            self.train_negitem_byuid,self.val_negitem_byuid=self._split_history(self.train_negitem_byuid,5/7)
       
        elif es_stage==2:
            self.train_positem_byuid=self.train_positem_byuid
            self.train_negitem_byuid=self.train_negitem_byuid
            
    def _build(self):
        tf.set_random_seed(self.seed)
        with tf.name_scope("input"):
            self.user_input = tf.placeholder(
                tf.int32, shape=[None, self._USER_SIZE], name='user_info')
            self.item_input = tf.placeholder(
                tf.int32, shape=[None, self._ITEM_SIZE], name='item_info')
            self.label = tf.placeholder(
                tf.int32, shape=[None, None], name='label')

        # Variables
        # embedding for users

        with tf.name_scope("intercept"):
             b = tf.Variable(
                initial_value=tf.truncated_normal(
                    (self.embedding_size, 1),
                    stddev=1.0 / np.sqrt(self.embedding_size)))

        # select and sum the columns of W depending on the input

        with tf.name_scope("user_embedding"):
            W = tf.Variable(
                initial_value=tf.truncated_normal(
                    (self.embedding_size, self._USER_SIZE_BIN),
                    stddev=1.0 / np.sqrt(self.embedding_size)))
            # embedding for movies

            # intercept

            w_offsets = [0] + [
                sum(self._USER_SIZE_OF_MASK_FIELDS[:i + 1])
                for i, j in enumerate(self._USER_SIZE_OF_MASK_FIELDS[:-1])
            ]
            w_offsets = tf.matmul(
                tf.ones(
                    shape=(tf.shape(self.user_input)[0], 1), dtype=tf.int32),
                tf.convert_to_tensor([w_offsets]))
            w_columns = self.user_input[:, :-
                                        self._USER_SIZE_ONLY_NUM] + w_offsets  # last column is not an index
            w_selected = tf.gather(W, w_columns, axis=1)
        # age * corresponding column of W

            aux = tf.matmul(
                W[:, -self._USER_SIZE_ONLY_NUM:],
                tf.transpose(
                    tf.to_float(
                        (self.user_input[:, -self._USER_SIZE_ONLY_NUM:]))))
            batch_age = tf.reshape(
                aux,
                shape=(self.embedding_size, tf.shape(self.user_input)[0], 1))
            w_with_age = tf.concat([w_selected, batch_age], axis=2)
            w_result = tf.reduce_sum(w_with_age, axis=2)
        with tf.name_scope("item_embedding"):
            A = tf.Variable(
                initial_value=tf.truncated_normal(
                    (self.embedding_size, self._ITEM_SIZE_BIN),
                    stddev=1.0 / np.sqrt(self.embedding_size)))
            # select and sum the columns of A depending on the input
            a_offsets = [0] + [
                sum(self._ITEM_SIZE_OF_MASK_FIELDS[:i + 1])
                for i, j in enumerate(self._ITEM_SIZE_OF_MASK_FIELDS[:-1])
            ]
            a_offsets = tf.matmul(
                tf.ones(
                    shape=(tf.shape(self.item_input)[0], 1), dtype=tf.int32),
                tf.convert_to_tensor([a_offsets]))
            a_columns = self.item_input[:, :-
                                        self._ITEM_SIZE_ONLY_NUM] + a_offsets  # last two columns are not indices
            a_selected = tf.gather(A, a_columns, axis=1)
            # dates * corresponding last two columns of A
            aux = tf.matmul(
                A[:, -self._ITEM_SIZE_ONLY_NUM:],
                tf.transpose(
                    tf.to_float(
                        self.item_input[:, -self._ITEM_SIZE_ONLY_NUM:])))
            batch_dates = tf.reshape(
                aux,
                shape=(self.embedding_size, tf.shape(self.item_input)[0], 1))
            # ... and the intercept
            intercept = tf.gather(
                b,
                tf.zeros(
                    shape=(tf.shape(self.item_input)[0], 1), dtype=tf.int32),
                axis=1)
            a_with_dates = tf.concat(
                [a_selected, batch_dates, intercept], axis=2)
            a_result = tf.reduce_sum(a_with_dates, axis=2)

            # Definition of g (Eq. (14) in the paper g = <Wu, Vi> = u^T * W^T * V * i)
        with tf.name_scope("output"):

            g = tf.matmul(tf.transpose(w_result), a_result,name="score")

            x = tf.to_float(self.label) * g
            self.prob = tf.nn.sigmoid(x,name="prob")

            loss = tf.reduce_mean(tf.nn.softplus(tf.diag_part(-x)),name="loss")

            # Regularization
            reg = self.nu * (tf.nn.l2_loss(W) + tf.nn.l2_loss(A))
            # Loss function with regularization (what we want to minimize)
            loss_to_minimize = loss + reg

            self.loss= loss_to_minimize

            self.optimize = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(
                    loss=loss_to_minimize)

    def _train(self):
        self.pos_distr,self.neg_distr=self._load_distribution()
        gpu_options = tf.GPUOptions(visible_device_list='1')
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        if self.first_fit==1:
            self.sess=tf.Session(config=config)
        tf.set_random_seed(self.seed)
        if self.first_fit==1:
            self.sess.run(tf.global_variables_initializer())
            self.first_fit=0
        
        if early_stop_method==None:
            for e in range(self.epochs+1):
                print('epochs %d'%(e))
                if e!=0:
                    for iter in range(int(np.ceil(len(self.uid_list)/self.batch_size))):
                        train_loss = self._train_a_batch(iter)

                        print('Iteration', iter, 'Train_loss', train_loss)
                reward0, reward1, agg_div, entro_div = self.eval_performance(
                        )
        else:
            best_epoch=0
            self._es_train(1)
            self.val_hist=[]
            self.train_hist=[]
            best_epoch=0
            for e in range(self.epochs+1):
                best_epoch=e
                print('epochs %d'%(e))
                average_loss=0.0
                cnt=0
                
                if e!=0:
                    for iter in range(int(np.ceil(len(self.uid_list)/self.batch_size))):
                        train_loss = self._train_a_batch(iter)
                        average_loss+=train_loss
                        
                        print('Iteration', iter, 'Train_loss', train_loss)
                self.val_hist.append(self._cal_val_loss())
                cnt=cnt if cnt !=0 else 1
                self.train_hist.append(average_loss/cnt)
                flg,best_epoch=self.early_stop_method(e,self.train_hist,self.val_hist)
                if flg == 1 : 
                    break
                reward0, reward1, agg_div, entro_div = self.eval_performance()
            self._es_train(2)
            for e in range(best_epoch+1):
                best_epoch=e
                print('epochs %d'%(e))
                if e!=0:
                    for iter in range(int(np.ceil(len(self.uid_list)/self.batch_size))):
                        train_loss = self._train_a_batch(iter)
                        print('Iteration', iter, 'Train_loss', train_loss)
                reward0, reward1, agg_div, entro_div = self.eval_performance()
            

    
    def fit(self,
             novelty_importance=0.0,
             batch_size=128,learning_rate=0.006,nu=0.0001,
             embedding_size=600,epochs=0,
             topk=10, limit=100,early_stop_method=None):
    
        self.beta = novelty_importance
        self.rng=np.random.RandomState(SEED)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.nu = nu
        self.embedding_size = embedding_size
        self.epochs=epochs
        self.top_N = topk
        self.LIMIT = limit
        self.NUM_USERS=len(self.user_info)
        self.NUM_ITEMS=len(self.item_info)
        
        if self.first_fit==1:
            self._build()
            
        self._train()
        
    def recommend(self,user,top_N=10):
        tmp = self._predict_mat_by_queue([user], self.itemid_list)
        prob_arr = list(zip(self.itemid_list, tmp[0]))
        prob_arr = sorted(prob_arr, key=lambda d: -d[1])
        cnt = 0
        recommend_lst=[]
        for pair in prob_arr:
            itemid = pair[0]
            if itemid not in self.train_rateditem_byuid[user]:
                recommend_lst.append(itemid)
                cnt += 1
                if cnt == top_N:
                    break
        return recommend_lst
    def predict(self,predict_pair):
        user_list = []
        item_list = []
        
        user_index={}
        item_index={}
        for (idx,(user, item)) in enumerate(predict_pair):
            
            if user not in user_index:
                user_index[user]=idx
            if item not in item_index:
                item_index[item]=idx
                
            user_list.append(user)
            item_list.append(item)
        tmp = self._predict_mat_by_queue(user_list, item_list)
        result=[]
        for (user,item) in predict_pair:
            result.append(tmp[user_index[user]][item_index[item]])
        return result


# # In[44]:


# user_num_inds=[]
# item_num_inds=[]
# for idx,feat in enumerate(movielens.df_userinfo.columns):
#     if feat in movielens.user_numerical_attr:
#         user_num_inds.append(idx)
        
# for idx,feat in enumerate(movielens.df_iteminfo.columns):
#     if feat in movielens.item_numerical_attr:
#         item_num_inds.append(idx)
    


# # In[45]:


# resys=NovResysClassifier(0,0,
#                          movielens.df_userinfo.values,0,user_num_inds,
#                          movielens.df_iteminfo.values,0,item_num_inds,
#                         movielens.df_rating.values)


# # In[46]:


# resys.preprocess()


# # In[47]:


# resys.precalculate()


# # In[53]:


# resys.fit(epochs=100)


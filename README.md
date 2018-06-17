# AccuracyNovelty

This is the implementation of the paper "Trade-off Between Accuracy and Novelty in Recommender Systems".

## Contents

1. Environment
2. Example
3. Run the Model
4. API

## Environment

Following environment is required (in python3).

```
pip install scikit-learn

pip install numpy 

pip install pandas

pip install scipy

pip install tensorflow

pip install tqdm
```



## Example

An example code of movielens is in "movielens_example.py"

If you want to run the code, you should first download dataset of [Movielens 100K Dataset](https://grouplens.org/datasets/movielens/).



## Run the Model

Import class

```
from NovResysClassifier import NovResysClassifier
```

First, construct recommender system.

```
resys=NovResysClassifier(0,0,
                         user_info,0,user_num_inds,

                         item_info,0,item_num_inds,

                         ratings)
```

Second, make preperation for calculation of novelty.

```
resys.preprocess()
```

Third, calculate and save distance and novelty, make preperation for training of recommender system.

```
resys.precalculate()
```

Last, train the recommender system.

```
resys.fit(epochs=100)
```

Recommend items for some user.

```
resys.recommend(1)
```

Predict the preference possibility of every user-item pair. 

```
resys.predict([(1,2),(2,3),(1,5)])
```



## API

```
class NovResysClassifier(
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
                 random_state=2018)
usage:
constructor of accuracy-novelty recommender system

parameters:
1. NOVELTY_TYPE(int, 0 or 1)-- 0 for popularity-based novelty, 1 for distance-based novelty.
2. DISTANT_TYPE(int, 0 or 1)-- 0 for set distance, 1 for euclid distance.
3. user_info: (2-dim array)-- features of users. Every row represents one user's feature vector and every column represents one kind of user attribute. There must be one columns represents user's ID. 
4. uid_ind: (int)-- index of column which represent user's ID.
5. user_num_inds: (1-dim array)-- indexes of columns which represent user's numerical features.
5. item_info: (2-dim array)-- features of items. Every row represents one item's feature vector and every column represents one kind of item attribute. There must be one columns represents item's ID. 
6. itemid_ind: (int)-- index of column which represent item's ID.
7. item_num_inds: (1-dim array)-- indexes of columns which represent item's numerical features.
8. rating_info: (tuple array)-- ratings of user-item pairs. Every element of array is like (user id, itemid, rating).
9. rating_threshold: (int)-- a user likes an item if his rating> rating_threshold, else he dislikes.
10. test_size: (float, 0~1)-- the ratio of test dataset.
11. random_state: (int)-- random seed.
```



```
NovResysClassifier.preprocess()
usage:
preprocess rating information and feature information, make preperation for calculation of novelty. This function must be called after classifier is constructed.
```



```
NovResysClassifier.precalculate()
usage:calculate and save distance and novelty, make preperation for training of recommender system. This function must be called after preprocess() is called.
```



```
NovResysClassifier.fit(self,
             novelty_importance=0.0,
             batch_size=128,learning_rate=0.006,nu=0.0001,
             embedding_size=600,epochs=0,
             topk=10, limit=100,early_stop_method=None):

usage: train the recommender system.

parameters:
1. novelty_importance(float, 0~1): importance of novelty mentioned in paper.
2. batch_size(int): training batch size.
3. learning_rate(int, 0~1): training learning rate.
4. nu(int, 0~1): training regularization coefficient.
5. embedding_size(int, 0~1): latent factor size of matrix factorizatio.
6. epochs(int): training epochs.
7. topk(int): size of topK recommendation list.
8. limit(int): try how many times to find incorrect negtive sample while training.
9. early_stop_method(function): if you want to use early stopping method to find best epoch, you should custom stopping function.
   "early_stop_method" expects a callable with following function: func(train_loss_history,val_loss_history) 
   and Returns(stop_flg, best_epoch). 
   train_loss_history: list of average training batch loss of every epoch. 
   val_loss_history: list of validation dataset loss of every epoch. 
   stop_flg: whether or not stop training.
   best_epoch: best epochs.
```



```
NovResysClassifier.predict(self,predict_pair)
usage: predict the preference possibility of every user-item pair

parameters:
predict_pair (tuple array): Every element is like (user id,item id).
```



```
NovResysClassifier.recommend(self,user,top_N=10)
usage: recommend items for some user

parameters
user(int): user id.
top_N(int): size of topk recommendation list.
```


# AccuracyNovelty

This is the implementation of the paper "Trade-off Between Accuracy and Novelty in Recommender Systems".

## Environment

If you want to run the code, you should first download dataset of [Movielens 100K Dataset](https://grouplens.org/datasets/movielens/).

Then, following environment is required (in python3).

```
pip install scikit-learn

pip install pandas

pip install scipy

pip install tensorflow
```

## Config

You should set config in config.ini.

1. seed: random seed used while preprocessing
2. data_dir: path of dataset
3. distant_type: 0 represents using eq15 in paper to calculate distance, 1 represents using eq14
4. novelty_type: 0 represents using eq12 in paper to calculate novelty, 1 represents using eq13

## Model

To run the model,  for example:

```
python accuracy_novelty_trainer.py -beta 0.5
```

There is a required argument named "beta", which represents the importance of novelty in recommender system.



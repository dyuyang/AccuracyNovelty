SEED=2000

DATA_DIR='./ml-100k/'
MODEL_DIR='./'

DISTANT_TYPE=1#int(FLAGS.dist)
NOVELTY_TYPE=0#int(FLAGS.nov)
BASELINE=1#int(FLAGS.baseline)

assert(DISTANT_TYPE==0 or DISTANT_TYPE==1)
assert(NOVELTY_TYPE==0 or NOVELTY_TYPE==1)

MLOBJ_PATH = 'ml_obj_%d.pkl'%(SEED)
UTILOBJ_PATH='ml_util_%d_dis_%d.pkl'%(SEED,DISTANT_TYPE)

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
        self.df_iteminfo["itemid"]=self.df_iteminfo["itemid"]-1
        self.df_userinfo["uid"]=self.df_userinfo["uid"]-1
        self.df_rating["itemid"]=self.df_rating["itemid"]-1
        self.df_rating["uid"]=self.df_rating["uid"]-1
        self.feature_engineering()
        
        self.user_numerical_attr =  ["age"]
        self.item_numerical_attr = ["year", "month", "day"]


# In[4]:


movielens=MovieLens()


# In[43]:
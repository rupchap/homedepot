import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
# from sklearn import pipeline, model_selection
from sklearn import pipeline, grid_search
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
# from sklearn.feature_extraction import DictVectorizer
# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer
import random
from utils import *
from process import load_data

# developed from https://www.kaggle.com/hellozeyu/home-depot-product-search-relevance/test-script-1

use_preprocessed_data = False


def main():

    random.seed(240480)

    if use_preprocessed_data:
        print('load preprocessed data')
        df_train = pd.read_csv('data/train_processed.csv')
        df_test = pd.read_csv('data/test_processed.csv')
    else:
        df_train, df_test = load_data()

    print('configure data for training')
    id_test = df_test['id']
    y_train = df_train['relevance'].values
    X_train =df_train[:]
    X_test = df_test[:]

    print('construct model')
    rfr = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=240480, verbose=1)
    tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
    tsvd = TruncatedSVD(n_components=10, random_state=240480)

    clf = pipeline.Pipeline([
        ('union', FeatureUnion(
            transformer_list=[
                ('cst',  cust_regression_vals()),
                ('txt1', pipeline.Pipeline([('s1', cust_txt_col(key='search_term')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
                ('txt2', pipeline.Pipeline([('s2', cust_txt_col(key='product_title')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
                ('txt3', pipeline.Pipeline([('s3', cust_txt_col(key='product_description')), ('tfidf3', tfidf), ('tsvd3', tsvd)])),
                ('txt4', pipeline.Pipeline([('s4', cust_txt_col(key='brand')), ('tfidf4', tfidf), ('tsvd4', tsvd)]))
            ],
            transformer_weights={
                'cst': 1.0,
                'txt1': 0.5,
                'txt2': 0.25,
                'txt3': 0.0,
                'txt4': 0.5
            },
            n_jobs=-1
        )),
        ('rfr', rfr)])

    print('run grid search')
    param_grid = {'rfr__max_features': [10], 'rfr__max_depth': [20]}
    RMSE = make_scorer(fmean_squared_error, greater_is_better=False)
    model = grid_search.GridSearchCV(estimator=clf, param_grid=param_grid, cv=2, scoring=RMSE)
    model.fit(X_train, y_train)

    print("Best parameters found by grid search:")
    print(model.best_params_)
    print("Best CV score:")
    print(model.best_score_)

    print('run predictions')
    y_pred = model.predict(X_test)

    print('save submission file')
    pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv', index=False)

main()
from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data():
    print('import data')
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')
    df_pro_desc = pd.read_csv('data/product_descriptions.csv')
    df_attr = pd.read_csv('data/attributes.csv')

    print('extract brand attributes')
    df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})

    print('concatenate train and test data')
    df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

    print('merge product description and brand')
    df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
    df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')

    print('fix typos in search terms')
    df_all['search_term'] = df_all['search_term'].map(lambda x: fix_typos(str(x)))

    print('stem text fields')
    df_all['search_term'] = df_all['search_term'].map(lambda x: str_stem(str(x)))
    df_all['product_title'] = df_all['product_title'].map(lambda x: str_stem(str(x)))
    df_all['product_description'] = df_all['product_description'].map(lambda x: str_stem(str(x)))
    df_all['brand'] = df_all['brand'].map(lambda x: str_stem(str(x)))

    print('product title similarity')
    df_all['pt_similarity'] = tfidf_similarity(df_all['product_title'], df_all['search_term'])

    print('product description similarity')
    df_all['pd_similarity'] = tfidf_similarity(df_all['product_description'], df_all['search_term'])

    print('brand similarity')
    df_all['brand_similarity'] = tfidf_similarity(df_all['brand'], df_all['search_term'])

    print('save processed data')
    df_all.to_csv('data/all_processed.csv')

    print('create length of text features')
    df_all['len_of_query'] = df_all['search_term'].map(lambda x: len(str(x).split())).astype(np.int64)
    df_all['len_of_title'] = df_all['product_title'].map(lambda x: len(str(x).split())).astype(np.int64)
    df_all['len_of_description'] = df_all['product_description'].map(lambda x: len(str(x).split())).astype(np.int64)
    df_all['len_of_brand'] = df_all['brand'].map(lambda x: len(str(x).split())).astype(np.int64)

    print('create additional features')
    df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title'] + "\t" + df_all['product_description']
    df_all['query_in_title'] = df_all['product_info'].map(lambda x: str_whole_word(str(x).split('\t')[0],str(x).split('\t')[1],0))
    df_all['query_in_description'] = df_all['product_info'].map(lambda x: str_whole_word(str(x).split('\t')[0],str(x).split('\t')[2],0))
    df_all['word_in_title'] = df_all['product_info'].map(lambda x: str_common_word(str(x).split('\t')[0],str(x).split('\t')[1]))
    df_all['word_in_description'] = df_all['product_info'].map(lambda x: str_common_word(str(x).split('\t')[0],str(x).split('\t')[2]))
    df_all['ratio_title'] = df_all['word_in_title']/df_all['len_of_query']
    df_all['ratio_description'] = df_all['word_in_description']/df_all['len_of_query']
    df_all['attr'] = df_all['search_term']+"\t"+df_all['brand']
    df_all['word_in_brand'] = df_all['attr'].map(lambda x: str_common_word(str(x).split('\t')[0],str(x).split('\t')[1]))
    df_all['ratio_brand'] = df_all['word_in_brand']/df_all['len_of_brand']
    df_all['search_term_feature'] = df_all['search_term'].map(lambda x: len(str(x)))

    print('enumerate brands')
    df_brand = pd.unique(df_all.brand.ravel())
    d = {}
    i = 1
    for s in df_brand:
        d[s] = i
        i += 1
    df_all['brand_feature'] = df_all['brand'].map(lambda x: d[x])

    print('split training and test data')
    num_train = df_train.shape[0]
    df_train = df_all.iloc[:num_train]
    df_test = df_all.iloc[num_train:]

    print('save processed data')
    df_train.to_csv('data/train_processed.csv')
    df_test.to_csv('data/test_processed.csv')

    return df_train, df_test


def tfidf_similarity(comparator1, comparator2):

    num_records = comparator1.shape[0]

    tfidf = TfidfVectorizer(stop_words='english')
    comparator1_vec = tfidf.fit_transform(comparator1)
    comparator2_vec = tfidf.transform(comparator2)

    similarity_score = np.zeros(num_records)

    for i in range(num_records):
        similarity_score[i]=((comparator1_vec[i] * comparator2_vec[i].T))[0, 0]

    return similarity_score

if __name__ == "__main__":
    load_data()
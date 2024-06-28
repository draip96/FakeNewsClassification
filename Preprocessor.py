import pandas as pd
import os
import glob

def preprocess_data(news_path, labels_path):
    labels = get_labels(labels_path)
    articles = get_articles(news_path, labels)
    articles = clean_data(articles)
    fin = pd.merge(articles, labels)
    return fin

def clean_data(df):
    df["Sentence"]=df["content"].str.split()
    df["WordsCount"]=df["Sentence"].apply(lambda x: len(x))
    df=df[df["WordsCount"]>= 20]
    df=df.drop(['Sentence', 'WordsCount'], axis=1)

    df = df[df['content'] != '']
    df = df[df['title'] != '']
    df = df.reset_index()
    return df

def get_labels(labels_path):
    labels = pd.read_csv(labels_path)
    labels = labels[['source', 'label']]

    fake_sources = labels.loc[labels['label'] == 1]
    reliable_sources = labels.loc[labels['label'] == 0]
    usable_sources = pd.concat([fake_sources, reliable_sources], ignore_index=True)
    return usable_sources

def get_articles(article_path, usable_sources):
    json_pattern = os.path.join(article_path,'*.json')
    file_list = glob.glob(json_pattern)

    dfs = []
    count = 0
    for file in file_list:
        data = pd.read_json(file)
        data = data[['id', 'source', 'title', 'content', 'published_utc']]
        if data.loc[0].source not in usable_sources.source.unique():
            continue
        if len(data.index) < 5:
            continue
        dfs.append(data)
        
    print("Collating Dataframe")
    articles = pd.concat(dfs, ignore_index=True)
    return articles


def main(NEWS_PATH = './datasets/newsdata/',
    LABELS_PATH = '/datasets/label',
    SAVE_PATH = ''):
    print("Preprocessing")

    if SAVE_PATH == '':
        SAVE_PATH = os.path.dirname(LABELS_PATH) + "/articles.pkl"
    print(SAVE_PATH)

    ARTICLE_CACHE = SAVE_PATH
    if os.path.isfile(ARTICLE_CACHE):
        print('Loading Dataframe from cache')
        df = pd.read_pickle(ARTICLE_CACHE)
        print('Complete')
        return SAVE_PATH

    print('Creating Dataframe')
    articles = preprocess_data(NEWS_PATH, LABELS_PATH)
    print('Saving Dataframe')
    articles.to_pickle(SAVE_PATH)

    print('Dataframe saved')

    return SAVE_PATH

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('news_path', default='./datasets/newsdata/')
    parser.add_argument('labels_path', default='./datasets/labels.csv')
    parser.add_argument('-s', '--save_path', default='')
    args = parser.parse_args()

    main(NEWS_PATH=args.news_path, LABELS_PATH=args.labels_path, SAVE_PATH=args.save_path)

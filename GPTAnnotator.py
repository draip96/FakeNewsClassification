import argparse
import pandas as pd
import os
import openai
import tiktoken



def truncate_article(article, enc):  
  tkns = enc.encode(article)
  return enc.decode(tkns[:475])

def get_response(prompt):
  response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
          {"role": "system", "content": "You are a helpful assistant who classifies news articles as either real or fake."},
          {"role": "user", "content": prompt},
      ],
      temperature=0.2,
      max_tokens=10
  )

  return response

def get_label(resp):
  
  # Basic Response Parser. Gets GPT label based on if the response contains the words real ( label 0) or fake (label 1).
  # Considers sentences containing misleading, and false as fake
  # Considers both real and fake appear in the sentence to be uncertain (label -1)
  # Considers "difficult to assess as real" or "unclear" to be uncertain (label -3)
  # Considers "Real but difficult to tell" as real
  # Any halted communication with chatgpt is labeled as -2

  resp = resp.lower()
  fake = resp.find('fake')
  false = resp.find('false')
  mis = resp.find('misleading')
  real = resp.find('real')
  diff = resp.find('difficult')
  uncl = resp.find('unclear')
  if diff > -1 or uncl > -1:
    if real < diff and real > -1:
      return 0
    return -3
  if real == -1:
    if fake > -1:
      return 1
  if fake == -1:
    if real > -1:
      return 0
  if fake == -1 and real == -1:
    return -1
  if real < fake:
    return 0
  elif fake < real:
    return 1
  if false > -1 or mis > -1:
    return 1
  return -4



def annotate_data(df, api_key, org=''):
  
  # The Prompt is key to achieving accurate GPT responses.

  openai.organization = org
  openai.api_key = api_key

  prompt_template = "Here is a news article: \n \n \
  --- \n \n \
  {} \
  --- \
  \n \n Is this article real or fake? \n \n"

  return classify_articles(df, prompt_template)



def classify_articles(df, prompt_template):
  enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
  labels = []
  arr = []
  for i in range(len(df)):
    article = df.loc[i].content
    article = truncate_article(article, enc)
    prompt = prompt_template.format(article)
    try:
      response = get_response(prompt)
      arr.append((i, response.choices[0].message.content))
      predicted_label = get_label(response.choices[0].message.content)
      print(predicted_label)
      labels.append(predicted_label)
    except:
      print('halted')
      arr.append((i, 'halted'))
      predicted_label = -2
      labels.append(predicted_label)

  df['gpt label'] = labels
  return df, arr



def main(API_KEY = "",
         ORG="",
         SAMPLE_SIZE = 5000,
         NEWS_PATH = './datasets/articles.pkl',
         SAVE_PATH = '',
         ANNOTATIONS_PATH = './datasets/gpt_annotations.txt'):

    if SAVE_PATH == '':
        SAVE_PATH = os.path.dirname(NEWS_PATH) + '/gpt_articles.csv'

    print(NEWS_PATH, API_KEY, SAVE_PATH, ANNOTATIONS_PATH)

    GPT_CACHE = SAVE_PATH
    if os.path.isfile(GPT_CACHE):
        print('Loading Dataframe from cache')
        df = pd.read_csv(GPT_CACHE)
        print('Complete')
        return SAVE_PATH

    articles = pd.read_pickle(NEWS_PATH)

    df = articles.sample(SAMPLE_SIZE).reset_index()
    
    print("Annotating Data")
    res, arr = annotate_data(df, API_KEY, ORG)

    res.to_csv(SAVE_PATH)

    with open(ANNOTATIONS_PATH, 'w') as f:
        for line in arr:
            f.write(f"{line}\n")

    print(pd.crosstab(res['label'], res['gpt label']))
    
    return SAVE_PATH

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('news_path', default='./datasets/articles.pkl')
    parser.add_argument('-a', '--api_key', required=True)
    parser.add_argument('-o', '--org', default='')
    parser.add_argument('-s', '--sample_size', default=5000, type=int)
    parser.add_argument('-sp', '--save_path', default='')
    parser.add_argument('-ap', '--annotations_path', default='')
    args = parser.parse_args()

    main(API_KEY = args.api_key,
         ORG=args.org,
         SAMPLE_SIZE = args.sample_size,
         NEWS_PATH = args.news_path,
         SAVE_PATH = args.save_path,
         ANNOTATIONS_PATH = args.annotations_path)
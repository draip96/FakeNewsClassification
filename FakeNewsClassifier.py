import argparse
import pandas as pd
import NELAPreprocessor
import NELAClassifier
import GPTAnnotator
import GPTClassifier

# Args are:
# 'newsdata' directory path,
# 'labels.csv' file path,
# -a OpenAI API Key,
# -o OpenAI Org ID (optional),
# -m transformer model (optional),
#       ['distilbert-base-uncased', 'roberta-base', 'ArthurZ/opt-350m-dummy-sc', 'microsoft/deberta-base'] 
#       arg can also be int index of this list
# -s sample size int (optional),
# -e number of epochs,
# -b batch size

def main():


    parser = argparse.ArgumentParser()
    parser.add_argument('news_path', default='./datasets/newsdata/')
    parser.add_argument('labels_path', default='./datasets/labels.csv')
    parser.add_argument('-a', '--api_key', required=True)
    parser.add_argument('-o', '--org', default='')
    parser.add_argument('-m', '--model_name', default='roberta-base')
    parser.add_argument('-s', '--sample_size', default=25000, type=int)
    parser.add_argument('-e', '--epochs', default=10, type=int)
    parser.add_argument('-b', '--batch_size', default=-1, type=int)
    args = parser.parse_args()

    print( args.news_path, args.model_name, args.epochs)
    nela_path= NELAPreprocessor.main(NEWS_PATH=args.news_path,
                    LABELS_PATH=args.labels_path)
    NELAClassifier.main(MODEL_NAME = args.model_name,
                   SAMPLE_SIZE = args.sample_size,
                   NUM_EPOCHS = args.epochs,
                   NEWS_PATH=nela_path,
                   BATCH_SIZE=args.batch_size)

    gpt_path = GPTAnnotator.main(API_KEY=args.api_key,
                       ORG=args.org,
                       NEWS_PATH=nela_path,
                       SAMPLE_SIZE=int(args.sample_size/5))
    GPTClassifier.main(MODEL_NAME = args.model_name,
                   SAMPLE_SIZE = int(args.sample_size/5),
                   NEWS_PATH = gpt_path,
                   NUM_EPOCHS = int(args.epochs/2))
    return

if __name__ == "__main__":
    main()

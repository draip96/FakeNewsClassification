import argparse
import pandas as pd
import torch
import transformers
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from datasets import load_metric
from scipy.special import softmax
import warnings
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()


def get_split(df, train_split):
      length = len(df)
      split = int(train_split * length)
      train = df.sample(split)
      test = df[~df.index.isin(train.index)]
      return train, test

def get_data_lists(df):
  return df.content.tolist()

def get_label_list(df):
  return df['gpt label'].tolist()

def tokenize_data(content, tokenizer):
  return tokenizer(content, padding=True, truncation=True, max_length=512, return_tensors='pt')

def preprocess_data(train_df, test_df, tokenizer):

  train_content = get_data_lists(train_df)
  test_content = get_data_lists(test_df)

  train_labels = get_label_list(train_df)
  test_labels = get_label_list(test_df)

  train_encodings = tokenize_data(train_content, tokenizer)
  test_encodings = tokenize_data(test_content, tokenizer)

  return train_encodings, train_labels, test_encodings, test_labels

class BuildDataset(torch.utils.data.Dataset):
      def __init__(self, encodings, labels):
          self.encodings = encodings
          self.labels = labels

      def __getitem__(self, idx):
          item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
          item['labels'] = torch.tensor(self.labels[idx])
          return item

      def __len__(self):
          return len(self.labels)

def main(NEWS_PATH = './datasets/gpt_articles.csv',
      MODEL_NAME = 'roberta-base',
      SAMPLE_SIZE = 2000,
      TRAIN_SPLIT = 0.75,
      NUM_EPOCHS = 5,
      BATCH_SIZE = -1):
  
  print(NEWS_PATH, MODEL_NAME, SAMPLE_SIZE, TRAIN_SPLIT, NUM_EPOCHS)

  models = ['distilbert-base-uncased', 'roberta-base', 'ArthurZ/opt-350m-dummy-sc', 'microsoft/deberta-base']

  chosen_model = MODEL_NAME

  if chosen_model in models:
    model_name = chosen_model
  else:
    try:
      chosen_model = int(chosen_model)
      if chosen_model in range(0, 4):
        model_name = models[chosen_model]
      else:
        print("Model not supported.")
        return 0
    except :
      print("Model not supported.")
      return 0

  learning_rates = [5e-5, 2e-5, 2e-5, 2e-5]
  learning_dict = dict(zip(models, learning_rates))

  batch_sizes = [128, 64, 32, 32]
  batch_dict = dict(zip(models, batch_sizes))

  if BATCH_SIZE == -1:
     BATCH_SIZE = batch_dict[model_name]

  df = pd.read_csv(NEWS_PATH)
  df = df[df['gpt label'] > -1]

  print('    Getting Model    ')
  print('=====================')

  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForSequenceClassification.from_pretrained(model_name)

  print("Using GPU: ", torch.cuda.is_available(), '\n')

  print(model, '\n')

  train_df, test_df = get_split(df, TRAIN_SPLIT)

  print(' Getting Encodings ')
  print('=====================')  

  preprocessed_data = preprocess_data(train_df, test_df, tokenizer)
  train_encodings, train_labels, test_encodings, test_labels = preprocessed_data

  print('Complete \n')

  print(' Building Dataloader ')
  print('=====================')   

  train_dataset = BuildDataset(train_encodings, train_labels)
  test_dataset = BuildDataset(test_encodings, test_labels)

  print('Complete \n')

  acc_metric = load_metric("accuracy")
  auc_metric = load_metric("roc_auc")
  f1_metric = load_metric("f1")

  def compute_metrics(eval_pred):
      logits, labels = eval_pred
      predictions = np.argmax(logits, axis=-1)

      pred_scores = softmax(predictions.astype('float32'), axis=-1)
      return {
          'acc' : acc_metric.compute(predictions=predictions, references=labels),
          'auc' : auc_metric.compute(prediction_scores=pred_scores, references=labels),
          'f1' : f1_metric.compute(predictions=predictions, references=labels)
      }

  model_output_dir = './training'

  print('    Training Model   ')
  print('=====================')

  training_args = TrainingArguments(
      output_dir=model_output_dir,
      learning_rate=learning_dict[model_name],
      num_train_epochs=NUM_EPOCHS,              # total number of training epochs
      per_device_train_batch_size=BATCH_SIZE,  # batch size per device during training
      per_device_eval_batch_size=BATCH_SIZE,   # batch size for evaluation
      warmup_steps=int((len(train_dataset)/BATCH_SIZE) * NUM_EPOCHS * 0.2),
      weight_decay=0.01,               # strength of weight decay
      logging_strategy='epoch',
      logging_steps=0.05,
      evaluation_strategy='epoch',
      save_strategy="epoch"
  )

  trainer = Trainer(
      model=model,                         # the instantiated 🤗 Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=train_dataset,         # training dataset
      eval_dataset=test_dataset,           # evaluation dataset
      compute_metrics=compute_metrics
  )

  trainer.train()

  trainer.evaluate(test_dataset)

  trainer.save_model("./models/gpt/" + model_name)


# if __name__ == "__main__":
#   main()

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('news_path', default='./datasets/articles.pkl')
  parser.add_argument('-m', '--model_name', default='roberta-base')
  parser.add_argument('-s', '--sample_size', default=25000, type=int)
  parser.add_argument('-t', '--train_split', default=0.75, type=float)
  parser.add_argument('-e', '--epochs', default=10, type=int)
  parser.add_argument('-b', '--batch_size', default=-1, type=int)

  args = parser.parse_args()

  main(NEWS_PATH =  args.news_path,
      MODEL_NAME = args.model_name,
      SAMPLE_SIZE = args.sample_size,
      TRAIN_SPLIT = args.train_split,
      NUM_EPOCHS = args.epochs,
      BATCH_SIZE = args.batch_size)

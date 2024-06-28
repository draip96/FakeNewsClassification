# This package demonstrates the usage of ChatGPT to augment weak labels for more accurate Fake News classification

### Description

This package demonstrates the fine-tuning process used to train the Fake News Classifier that can be found [here](https://huggingface.co/newsmediabias/FakeNews-Classifier-NELA-GT_GPT-Labels).

The process begins by fine tuning and saving a transformer model on a large sample of the dataset with it's weak labels. A second, smaller sample of the dataset is then sampled and sent through OpenAI's API to ChatGPT to be re-labeled. The previously saved model is then further fine-tuned on these new labels, improving the accuracy of its classifications. 

Because the model is fine-tuned first on the weak labels, only a fraction of the dataset needs to be annotated by ChatGPT to improve performance.

### Design principles

In order of importance, here are the qualities we want this package to have:

- **Accurate.** Many fake news classifiers rely on source-level labels from news article datasets, as article-level labels are time-consuming to collate. Instead, this package uses chatGPT to augment the a subset of the labels to article-level, giving the final classifier better accuracy.
- **Low Cost.** Paid LLMs such as GPT-3.5 or Clause produce good results, but can be expensive for individuals or small organizations, as well as large workloads. This package, on the other hand, will allow for comparable accuracy to GPT-3.5 for a small up-front cost.
- **Performant.** While not as important as the first two factors, we do want to make our package reasonably fast and not too resource-hungry.

### Usage
Choose a model and use any dataset to train a Fake News Classifier using FakeNewsClassifier.py.

    FakeNewsClassifier.py news_path labels_path -a API_KEY [-o ORG] 
    [-m MODEL_NAME] [-s SAMPLE_SIZE] [-e EPOCHS] [-b BATCH_SIZE] [-h] 

    Required arguments:
        news_path (Positional)
        labels_path (Positional)
        -a API_KEY, --api_key API_KEY

    Optional Arguments:
        -h, --help            show this help message and exit
        -o ORG, --org ORG
        -m MODEL_NAME, --model_name MODEL_NAME, STR or INT, (Default RoBERTa)
            *From list or index:*
            ['distilbert-base-uncased', 'roberta-base', 
            'ArthurZ/opt-350m-dummy-sc', 'microsoft/deberta-base'] 

        -s SAMPLE_SIZE, --sample_size SAMPLE_SIZE, (Default 25000)
        -e EPOCHS, --epochs EPOCHS, (Default 10)
        -b BATCH_SIZE, --batch_size BATCH_SIZE, (Defaults A100 batch sizes)



**Steps:**

1. Clone:  https://github.com/VectorInstitute/NewsMediaBias/

2. Download Dependencies:

        cd NewsMediaBias/fakenewsclassifier
        
        pip install -r requirements.txt

3. Download Your Dataset:
        Recommended Folder structure:
        
        NewsMediaBias/
        ├─ debiaser/
        │  ├─ ...
        ├─ fakenewsclassifier/
        │  ├─ *datasets/*
        │  │  ├─ *labels.csv*
        │  │  ├─ *newsdata/*
        │  │  │  ├─ NewsOutlet.json
        │  │  │  ├─ ...
        │  ├─ FakeNewsClassifer.py
        │  ├─ README.md
        │  ├─ ...

3. Train the Model:

        Example Usage:
    
        python FakeNewsClassifier.py datasets/newsdata/ datasets/labels.csv\
        --api_key=sk-z2956PNO****************************************\
        --org=org-7cyuVnN*****************\
        --model_name='microsoft/deberta'\
        --epochs=10\
        --batch_size=32



#### Custom Datasets

To build a sequence classifier using your own data, save a data frame as a pickle, and use the individual models to classify and annotate the data. 

    Classifier.py  news_path [-m MODEL_NAME] [-s SAMPLE_SIZE] 
    [-t TRAIN_SPLIT] [-e EPOCHS] [-b BATCH_SIZE] [-h]
                        
    Optional Arguments:
        news_path (Positional)
        -h, --help            show this help message and exit
        -m MODEL_NAME, --model_name MODEL_NAME, STR or INT, (Default RoBERTa)
            *From list or index:*
            ['distilbert-base-uncased', 'roberta-base', 
            'ArthurZ/opt-350m-dummy-sc', 'microsoft/deberta-base'] 
        -s SAMPLE_SIZE, --sample_size SAMPLE_SIZE, (Default 25000)
        -t TRAIN_SPLIT, --train_split TRAIN_SPLIT, (Default 0.75)
        -e EPOCHS, --epochs EPOCHS, (Default 10)
        -b BATCH_SIZE, --batch_size BATCH_SIZE, (Defaults A100 batch sizes)
---
    GPTClassifier.py  news_path [-m MODEL_NAME] [-s SAMPLE_SIZE] 
    [-t TRAIN_SPLIT] [-e EPOCHS] [-b BATCH_SIZE] [-h]
                    
    Optional Arguments:
        news_path (Positional)
        -h, --help            show this help message and exit
        -m MODEL_NAME, --model_name MODEL_NAME, STR or INT, (Default RoBERTa)
            *From list or index:*
            ['distilbert-base-uncased', 'roberta-base', 
            'ArthurZ/opt-350m-dummy-sc', 'microsoft/deberta-base'] 
        -s SAMPLE_SIZE, --sample_size SAMPLE_SIZE, (Default 25000)
        -t TRAIN_SPLIT, --train_split TRAIN_SPLIT, (Default 0.75)
        -e EPOCHS, --epochs EPOCHS, (Default 10)
        -b BATCH_SIZE, --batch_size BATCH_SIZE, (Defaults A100 batch sizes)
---
    GPTAnnotator.py news_path -a API_KEY [-o ORG] [-s SAMPLE_SIZE] 
    [-sp SAVE_PATH] [-ap ANNOTATIONS_PATH][-h]

    Optional Arguments:
        news_path (Positional)
        -h, --help            show this help message and exit
        -a API_KEY, --api_key API_KEY (*Required*),
            Open AI API Key, this requires an OpenAI account with billing enabled
        -o ORG, --org ORG
        -s SAMPLE_SIZE, --sample_size SAMPLE_SIZE, (Default 5000)
        -sp SAVE_PATH, --save_path SAVE_PATH
        -ap ANNOTATIONS_PATH, --annotations_path ANNOTATIONS_PATH,

**Steps:**

1. Clone:  https://github.com/VectorInstitute/NewsMediaBias/

2. Download Dependencies:

        cd NewsMediaBias/fakenewsclassifier
        
        pip install -r requirements.txt

3. Save your dataset
   
    Use pandas to save a DataFrame of your data to a pkl file.
    (Note: Ensure the DataFrame contains a title, content, and label column)

        Recommended Folder structure:
        
        NewsMediaBias/
        ├─ debiaser/
        │  ├─ ...
        ├─ fakenewsclassifier/
        │  ├─ *datasets/*
        │  │  ├─ *articles.pkl*
        │  ├─ FakeNewsClassifer.py
        │  ├─ README.md
        │  ├─ ...


3. Train the preliminary model:

        python Classifier.py --options

4. Annotate the labels:

        python GPT Annotate.py --options

5. Train the final classifer:
    
        python GPTClassifier.py --options
    
The final model is built on top of the preliminary model. This is done to reduce the amount of GPT annotated labels needed to create an accurate classifier that represents the data. This model is saved in to the /models/gpt/{model_name} directory and can be loaded to make predictions.

### Implementation

This package leverages OpenAI's ChatGPT model to augment weak labels. The models are trained on fake news datasets for the purpose of sequence calssification. The DistilBERT, RoBERTa, DeBERTa and OPT models are available for testing. 


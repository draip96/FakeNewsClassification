import torch
import pandas as pd
import time
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, 
    TrainingArguments, EarlyStoppingCallback
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Constants
SEED = 200
MAX_SEQ_LENGTH = 2048
torch_dtype = torch.float16
attn_implementation = "eager"

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
OUTPUT_DIRECTORY = "/results"
ADAPTER_PATH = "/adapter"
SAVE_TO = "/results/final-model"


# Function to prepare the dataset for the chat model format
def chat_function(row, tokenizer):
    row_json = [
        {
            "role": "user",
            "content": (
                "You are a helpful news fact-checking bot trained to assess the accuracy of information. "
                "Your task is to analyze the given article and determine whether it is 'Factually Correct' or 'Factually Incorrect'. "
                "Fact-checking is the methodical process of verifying claims in public discourse or media reports. "
                "It is vital for countering misinformation and disinformation, thereby enhancing public knowledge and trust. "
                "Consider the following in your evaluation:\n"
                "Misinformation: Incorrect or misleading information shared without intent to harm.\n"
                "Disinformation: Information that is knowingly false, often prejudiced, and disseminated with the intent to mislead.\n"
                "Your analysis should include:\n"
                "Verification of key claims against multiple reliable sources.\n"
                "Identification of logical fallacies or statements that may mislead readers.\n"
                "Assessment of the context in which the information was presented, including the source’s history and potential motivations.\n"
                "Evaluation for any presence of hate speech, linguistic harm, or intent to spread prejudice.\n"
                "Provide your assessment in the following format:\n"
                "Classification: [Factually Correct/Factually Incorrect]\n"
                "Explanation: Provide a concise, evidence-based explanation for your classification. Reference specific examples from the article and contradicting evidence from trusted sources, if applicable.\n"
                "Ensure to remain objective, basing your assessment strictly on facts and evidence rather than personal opinions or biases."
            )
        },
        {"role": "assistant", "content": f"Label: {row['text_label']}"}
    ]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

# Load and prepare dataset
def load_and_prepare_data(train_path, val_path, tokenizer):
    train_df = pd.read_csv(train_path).sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = pd.read_csv(val_path)
    print(f"Training records: {len(train_df)}, and validation records: {len(val_df)}")
    
    train_ds = Dataset.from_pandas(train_df).map(lambda row: chat_function(row, tokenizer), num_proc=4)
    val_ds = Dataset.from_pandas(val_df).map(lambda row: chat_function(row, tokenizer), num_proc=4)

    return train_ds, val_ds

# Model and tokenizer setup
def get_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        quantization_config=bnb_config,
        attn_implementation=attn_implementation,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer

# Training arguments setup
def get_training_arguments():
    return TrainingArguments(
        OUTPUT_DIRECTORY=OUTPUT_DIRECTORY,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        optim="paged_adamw_32bit",
        evaluation_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        logging_steps=1000,
        learning_rate=2e-5,
        weight_decay=0.01,
        adam_beta2=0.999,
        fp16=False,
        bf16=False,
        seed=SEED,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine"
    )

# Main function to start training
def main():
    train_path = "train.csv"
    val_path = "val.csv"
    
    model, tokenizer = get_tokenizer()
    training_arguments = get_training_arguments()
    train_ds, val_ds = load_and_prepare_data(train_path, val_path, tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_arguments
    )

    start_time = time.time()
    trainer.train()
    end_time = time.time()
    
    print('Training is successful')
    hours, minutes = divmod((end_time - start_time) // 60, 60)
    print(f"Total training time taken: {hours:.0f} hours and {minutes:.0f} minutes")

if __name__ == "__main__":
    main()

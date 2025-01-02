import json
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainingLogger")

# Load your JSON data
with open('question_answers.json', 'r') as f:
    json_data = json.load(f)

# Flatten JSON data into question-answer pairs
data = [
    {"prompt": item["question"], "completion": item["answer"]}
    for section in json_data.values()
    for item in section
]

# Split data into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Convert to DatasetDict
dataset = DatasetDict({
    "train": Dataset.from_dict({
        "prompt": [d["prompt"] for d in train_data],
        "completion": [d["completion"] for d in train_data]
    }),
    "validation": Dataset.from_dict({
        "prompt": [d["prompt"] for d in val_data],
        "completion": [d["completion"] for d in val_data]
    })
})

# Load tokenizer and model
model_name = "EleutherAI/gpt-neo-125M"  # GPT-Neo example
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

model = GPTNeoForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

# Tokenize dataset
def tokenize_function(examples):
    inputs = tokenizer(examples['prompt'], padding="max_length", truncation=True, max_length=128)
    outputs = tokenizer(examples['completion'], padding="max_length", truncation=True, max_length=128)
    inputs['labels'] = outputs['input_ids']
    return inputs

logger.info("Tokenizing datasets...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=1e-5,  # Adjusted learning rate
    per_device_train_batch_size=8,  # Adjusted batch size
    per_device_eval_batch_size=8,
    num_train_epochs=3,  # Reduced epochs for quicker runs
    weight_decay=0.01,
    warmup_steps=200,
    logging_dir='./logs',
    logging_steps=50,
    gradient_accumulation_steps=4,
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    fp16=False,
    report_to="none"
)

# Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Train the model
logger.info("Starting model training...")
trainer.train()

# Save the fine-tuned model and tokenizer
logger.info("Saving the fine-tuned model...")
model.save_pretrained("./fine-tuned-model")
tokenizer.save_pretrained("./fine-tuned-model")

# Test the model with a text generation pipeline
from transformers import pipeline

logger.info("Testing the fine-tuned model...")
generator = pipeline("text-generation", model="./fine-tuned-model", tokenizer="./fine-tuned-model")
response = generator("What is your name?", max_length=50, num_return_sequences=1)
logger.info(f"Generated response: {response[0]['generated_text']}")
print(response[0]['generated_text'])

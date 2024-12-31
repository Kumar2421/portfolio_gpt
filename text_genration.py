import json
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset

# Load your JSON data
with open('question_answers.json', 'r') as f:
    json_data = json.load(f)

# Flatten your JSON data into a list of question-answer pairs
data = [
    {"prompt": item["question"], "completion": item["answer"]}
    for section in json_data.values()
    for item in section
]

# Convert the list into a Hugging Face Dataset
dataset = Dataset.from_dict({
    "prompt": [d["prompt"] for d in data],
    "completion": [d["completion"] for d in data]
})

# Load the tokenizer and model
model_name = "EleutherAI/gpt-neo-125M"  # Example with GPT-Neo
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Add a padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

model = GPTNeoForCausalLM.from_pretrained(model_name)

# Tokenize the dataset
def tokenize_function(examples):
    inputs = tokenizer(examples['prompt'], padding="max_length", truncation=True, max_length=128)
    outputs = tokenizer(examples['completion'], padding="max_length", truncation=True, max_length=128)
    inputs['labels'] = outputs['input_ids']
    return inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,  # Ideally, use a separate validation set
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./fine-tuned-model")
tokenizer.save_pretrained("./fine-tuned-model")

# Test the model
from transformers import pipeline

# Load the fine-tuned model
model = GPTNeoForCausalLM.from_pretrained("./fine-tuned-model")
tokenizer = GPT2Tokenizer.from_pretrained("./fine-tuned-model")

# Create a text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Test the model
response = generator("What is your name?", max_length=50)
print(response[0]['generated_text'])
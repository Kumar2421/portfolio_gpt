from transformers import GPTNeoForCausalLM, GPT2Tokenizer, pipeline

# Load the fine-tuned model and tokenizer
model = GPTNeoForCausalLM.from_pretrained("./fine-tuned-model")
tokenizer = GPT2Tokenizer.from_pretrained("./fine-tuned-model")

# Create a text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Function to ask questions
def ask_question(question, max_length=50):
    response = generator(
        question,
        max_length=max_length,
        num_return_sequences=1,
        truncation=True  # Explicitly set truncation
    )
    return response[0]['generated_text']

# Example questions
questions = [
    "What is your name?",
    "What is your current job title?",
    "Can you briefly describe your professional profile?",
    "Where did you study and what did you major in?",
    "Can you tell me about your internship experience?"
]

# Ask the model each question
for question in questions:
    print(f"Question: {question}")
    answer = ask_question(question)
    print(f"Answer: {answer}\n")
import nltk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "t5-base-medium-title-generation/checkpoint-2000"
model_dir = f"drive/MyDrive/Models/{model_name}"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

max_input_length = 512

text = "Jeg vil gerne have en pizza med pepperoni og champignon. Den skal være med dressing. "

inputs = ["summarize: " + text]

inputs = tokenizer(
    inputs, max_length=max_input_length, truncation=True, return_tensors="pt"
)
output = model.generate(
    **inputs, num_beams=8, do_sample=True, min_length=10, max_length=64
)
decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]

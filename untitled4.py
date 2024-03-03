from transformers import AutoTokenizer, TFBertForQuestionAnswering
import tensorflow as tf
from tika import parser # pip install tika

tokenizer = AutoTokenizer.from_pretrained("ESGBERT/GovernanceBERT-governance")
model = TFBertForQuestionAnswering.from_pretrained("ESGBERT/GovernanceBERT-governance", from_pt=True)

raw=parser.from_file('example.pdf')
text=raw['content']

#question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

inputs = tokenizer("What is ESG", text, return_tensors="tf")
outputs = model(**inputs)

answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])

predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
print(tokenizer.decode(predict_answer_tokens))
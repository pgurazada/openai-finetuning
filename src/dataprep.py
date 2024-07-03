import json
from datasets import load_dataset

# Prompt templates
system_message = """
You are a helpful assistant that creates a concise summary of an input dialogue.
"""

user_message_template = """
### Dialogue:
{dialogue}
"""

assistant_message_template = """
### Summary:
{summary}
"""

# Data
dataset = load_dataset("knkarthick/dialogsum")
train_size, validation_size = 100, 20
training_dataset = dataset['train'].shuffle(seed=42).select(range(train_size))
validation_dataset = dataset['validation'].shuffle(seed=42).select(range(validation_size))

def convert_to_msg(row):
    return { 
        'messages': [
            {
                'role': 'system', 
                'content': system_message
            },
            {
                'role': 'user', 
                'content': user_message_template.format(dialogue=row['dialogue'])
            },
            {
                'role': 'assistant',
                'content': assistant_message_template.format(summary=row['summary'])
            }           
        ]
    }

f = open('data/train.jsonl', 'w')
for row in training_dataset:
    json.dump(convert_to_msg(row), f)
    f.write('\n')
f.close()

f = open('data/valid.jsonl', 'w')
for row in validation_dataset:
    json.dump(convert_to_msg(row), f)
    f.write('\n')
f.close()
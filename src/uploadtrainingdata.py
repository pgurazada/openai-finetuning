import os
import pickle

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.environ["OPENAI_KEY"]

client = OpenAI(api_key=openai_api_key)

train_file = 'data/train.jsonl'
valid_file = 'data/valid.jsonl'

training_response = client.files.create(
    file=open('data/train.jsonl','rb'),
    purpose="fine-tune",
)

training_file_id = training_response.id

validation_response = client.files.create(
    file=open('data/valid.jsonl','rb'),
    purpose="fine-tune",
)

validation_file_id = validation_response.id

openai_files_data = {
    'training_file_id': training_file_id,
    'validation_file_id': validation_file_id
}

with open('data/openai-file-data.pkl', 'wb') as output_file:
    pickle.dump(openai_files_data, output_file)
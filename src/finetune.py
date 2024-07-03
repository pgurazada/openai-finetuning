import os
import pickle

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.environ["OPENAI_KEY"]

client = OpenAI(api_key=openai_api_key)

with open('data/openai-file-data.pkl', 'rb') as f:
    openai_file_data = pickle.load(f)

finetuning_response = client.fine_tuning.jobs.create(
    training_file=openai_file_data['training_file_id'],
    validation_file=openai_file_data['validation_file_id'],
    model="gpt-3.5-turbo",
    suffix='dialogsum',
    hyperparameters={
        "n_epochs": 2
    },
    seed=42
)

# List currently running fine-tuning jobs

client.fine_tuning.jobs.list(limit=10)

finetuning_job = client.fine_tuning.jobs.retrieve("ftjob-Um7AMWzmreBZxwpjBgYbngOd")
finetuning_job.status
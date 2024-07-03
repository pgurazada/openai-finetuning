import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.environ["OPENAI_KEY"]

client = OpenAI(api_key=openai_api_key)

finetuning_job_id = "ftjob-Um7AMWzmreBZxwpjBgYbngOd"
finetuning_job = client.fine_tuning.jobs.retrieve("ftjob-Um7AMWzmreBZxwpjBgYbngOd")
finetuned_model_name = finetuning_job.fine_tuned_model

system_message = """
You are a helpful assistant that creates a concise summary of an input dialogue.
"""

gold_dialogue = """
#Person1#: do you have any plans for dinner tonight?
#Person2#: no, I was thinking of putting a frozen pizza in the oven or something. How about you? 
#Person1#: I was thinking maybe we could make dinner together tonight. What do you think? 
#Person2#: I'm absolutely useless at cooking! 
#Person1#: I could teach you how to cook something healthy. Frozen pizza are so bad for you! 
#Person2#: I know they aren't good for me, but they are cheap, convenient, and fairly tasty. 
#Person1#: I recently saw a piece for spicy chicken curry in a magadize. Maybe we could try that? 
#Person2#: yeah, why not. Do you have all the ingredients? 
#Person1#: I bought all the ingredients this morning, so let's start! 
#Person2#: what do we do first? 
#Person1#: first, you need to wash the vegetables and then chop them into little pieces. 
#Person2#: ok. Should I heat the wok? 
#Person1#: yes. Once it gets hot, put a little oil in it, add the vegetables and stir-fry them for a few minutes. 
#Person2#: what about the chicken? 
#Person1#: that needs to be cut into thin strips about 3 cm long and then it can be stir-fried on its own until its cooked through. 
#Person2#: how about the rice? 
#Person1#: I'll prepare it. Do you prefer white rice or brown rice? 
#Person2#: white rice, please. None of that healthy brown stuff for me!
"""

completion = client.chat.completions.create(
  model=finetuned_model_name,
  messages=[
    {
        "role": "system", 
        "content": system_message
    },
    {
        "role": "user", 
        "content": gold_dialogue
    }
  ]
)

print(completion.choices[0].message)
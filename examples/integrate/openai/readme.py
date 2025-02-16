import os
import time

from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def response_text(openai_resp):
    return openai_resp['choices'][0]['message']['content']


# Before running this case, make sure the OPENAI_API_KEY environment variable is set

question = 'what‘s chatgpt'

# OpenAI API original usage
start_time = time.time()
response = client.chat.completions.create(model='gpt-3.5-turbo',
messages=[
  {
      'role': 'user',
      'content': question
  }
])
print(f'Question: {question}')
print('Time consuming: {:.2f}s'.format(time.time() - start_time))
print(f'Answer: {response_text(response)}\n')

# GPTCache exact matching usage
print('GPTCache exact matching example.....')
print('Cache loading.....')

# To use GPTCache, that's all you need
# -------------------------------------------------
from gptcache import cache
from gptcache.adapter import openai

cache.init()
cache.set_openai_key()
# -------------------------------------------------

question = 'what is github'
for _ in range(2):
    start_time = time.time()
    response = client.chat.completions.create(model='gpt-3.5-turbo',
    messages=[
      {
          'role': 'user',
          'content': question
      }
    ])
    print(f'Question: {question}')
    print('Time consuming: {:.2f}s'.format(time.time() - start_time))
    print(f'Answer: {response_text(response)}\n')

# GPTCache similar search usage
print('GPTCache similar search example.....')
print('Cache loading.....')

from gptcache import cache
from gptcache.adapter import openai
from gptcache.embedding import Onnx
from gptcache.manager import get_data_manager, VectorBase
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

onnx = Onnx()
vector_base = VectorBase('faiss', dimension=onnx.dimension)
data_manager = get_data_manager('sqlite', vector_base)
cache.init(
    embedding_func=onnx.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
    )
cache.set_openai_key()

questions = [
    'what is github',
    'can you explain what GitHub is',
    'can you tell me more about GitHub'
    'what is the purpose of GitHub'
]

for question in questions:
    for _ in range(2):
        start_time = time.time()
        response = client.chat.completions.create(model='gpt-3.5-turbo',
        messages=[
            {
                'role': 'user',
                'content': question
            }
        ])
        print(f'Question: {question}')
        print('Time consuming: {:.2f}s'.format(time.time() - start_time))
        print(f'Answer: {response_text(response)}\n')

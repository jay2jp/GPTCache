import time


def response_text(openai_resp):
    return openai_resp.choices[0].message.content #'choices'][0]['message']['content']

from gptcache import cache
from gptcache.adapter import openai
from gptcache.embedding import Onnx
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

print("Cache loading.....")

onnx = Onnx()
data_manager = get_data_manager(CacheBase("sqlite"), VectorBase("faiss", dimension=onnx.dimension))
cache.init(
    embedding_func=onnx.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
    )
cache.set_openai_key()

questions = [
    "what's github",
    "can you explain what GitHub is",
    "can you tell me more about GitHub",
    "what is the purpose of GitHub"
]

for question in questions:
    start_time = time.time()
    response = openai.ChatCompletion.create(
        model='gpt-4o',
        stream=True,
        messages=[
            {
                'role': 'system',
                'content': 'You are a helpful assistant.'
            },
            {
                'role': 'user',
                'content': "what is cheese"
            },
            {
                'role': 'assistant',
                'content': "cheese is a food"
            },
            {
                'role': 'user',
                'content': question
            }
        ],
    )
    first_chunk_received = False
    collected_text = ""
    print(f'\nQuestion: {question}')
    print("Time consuming: {:.2f}s".format(time.time() - start_time))
    
    for chunk in response:
        if not first_chunk_received:
            print(f'Time to first chunk: {time.time() - start_time:.2f}s')
            first_chunk_received = True
        
        if "content" in chunk.choices[0].delta:#['choices'][0]['delta']:
            content = chunk.choices[0].delta.content#['choices'][0]['delta']['content']
            collected_text += content
    
    print(f'Complete response:\n{collected_text}\n')

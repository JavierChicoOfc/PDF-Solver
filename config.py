OPENAI_API_KEY = "ENTER YOUR OPENAI API KEY HERE"

PINECONE_API_KEY = "ae6ec0cd-baba-43cc-b4b2-f81d53efd409"
PINECONE_ENVIRONMENT = "asia-southeast1-gcp-free"
PINECONE_INDEX_NAME = "tfg-vectordb"

MODEL_TYPE = "gpt-3.5-turbo" # ChatGPT model as default
CHAIN_TYPE = "stuff"
NUM_SIMILAR_DOCS = 4
EVALUATION = False

AI_TASK_TEMPLATE = """
You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
In a different paragraph, include verbatim quote and a comment where to find it in the text (page number).
After the quote write a step by step explanation.Use bullet points.
If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.

{context}

Question: {question}

Helpful answer in markdown:
"""

AI_EVAL_TEMPLATE = """
You are a helpful AI assistant. Evaluate the following response to the question at the end, that makes use of the context below.
The evaluation must follow the following format: 1-5, where 1 is the worst and 5 is the best.
If the response is not related to the question and dont include anything of this: verbatim quote, a comment where to find it in the text (page number), a step by step explanation. You must evaluate it with a 1.
If the response is related somehow to the question but dont include correctly: verbatim quote, a comment where to find it in the text (page number), a step by step explanation. You must evaluate it with a 2.
If the response is related somehow to the question and include correctly: verbatim quote, or a comment where to find it in the text (page number), or a step by step explanation. You must evaluate it with a 3.
If the response answer the question but dont include include correctly: verbatim quote or a comment where to find it in the text (page number) or a step by step explanation. You must evaluate it with a 4.
If the response answer the question and include correctly: verbatim quote, a comment where to find it in the text (page number), a step by step explanation. You must evaluate it with a 5.

{context}

Question: {question}

Response: {response}

Helpful answer in markdown:
"""
import numpy as np
import openai

client = openai.OpenAI(api_key='YOUR_API_KEY')

def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def cosine_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

def verify_fact(fact, transcription, threshold=0.8):
    embedding_fact = get_embedding(fact)
    embedding_transcript = get_embedding(transcription)

    similarity = cosine_similarity(embedding_fact, embedding_transcript)
    verified = similarity >= threshold

    return similarity, verified

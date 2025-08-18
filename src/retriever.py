from numpy import dot
from numpy.linalg import norm
from openai import OpenAI
from utils.config import get_openai_key

class Retriever:
    def __init__(self, top_k):
        self.top_k = top_k
    
    def cos_sim(self, a, b): 
        return dot(a,b)/(norm(a)*norm(b))  
    
    def embedding_openai(self, query):
        client = OpenAI(api_key=get_openai_key())
        response = client.embeddings.create(
            input = query,
            model = 'text-embedding-3-small',
        )
        return response
    
    def retrieve_nodes(self, query, memory):
        query_embedding = self.embedding_openai(query).data[0].embedding
        memory_score = []
        for key in memory:
            memory_embedding = memory[key]['embedding']
            score = self.cos_sim(query_embedding, memory_embedding)
            memory_score.append((key, score))
        memory_score = sorted(memory_score, key=lambda x:x[1], reverse=True)
        return memory_score[:self.top_k]

    def memory_to_embedding(self, memory):
        memory_key = []
        memory_val = []
        memory_embedding = {}
        for key, val in memory.items():
            memory_key.append(key)
            memory_val.append(val)
        response = self.embedding_openai(memory_val)
        for i in range(len(memory_key)):
            memory_embedding[memory_key[i]] = {'text': memory_val[i], 'embedding': response.data[i].embedding}
        return memory_embedding

    def get_cur_memory(self, session_num, memory_embedding):
        memory_cur = {}
        for key in list(memory_embedding.keys()):
            if int(key[1])==session_num:
                memory_cur[key] = {'text': memory_embedding[key]['text'], 'embedding': memory_embedding[key]['embedding']}
        return memory_cur

    def get_past_memory(self, session_num, memory_embedding):
        memory_past = {}
        for key in list(memory_embedding.keys()):
            if int(key[1])<session_num:
                memory_past[key] = {'text': memory_embedding[key]['text'], 'embedding': memory_embedding[key]['embedding']}
        return memory_past
    
    def retrieve_for_linking(self, session_num, target_key, memory_embedding):
        retrieved_nodes = []
        memory_cur = self.get_cur_memory(session_num, memory_embedding)
        memory_past = self.get_past_memory(session_num, memory_embedding)
        
        for key in memory_past:
            score = self.cos_sim(memory_cur[target_key]['embedding'], memory_past[key]['embedding'])
            retrieved_nodes.append((key, score))

        retrieved_nodes = sorted(retrieved_nodes, key=lambda x:x[1], reverse=True)
        retrieved_nodes =  retrieved_nodes[:self.top_k]
     
        retrieved_memory = []
        for retrieved_node, score in retrieved_nodes:
            retrieved_memory.append({retrieved_node: memory_past[retrieved_node]['text']})
        return retrieved_memory


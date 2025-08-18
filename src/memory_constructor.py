import json
import pprint
import argparse
import os

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback

from src.retriever import Retriever
from utils.config import get_openai_key
from utils.path import (
    load_prompt,
    get_save_path,
    load_episode,
    load_summary
)
from utils.utils import (
    get_dialogue,
    to_dic
)

class MemoryConstructor:
    def __init__(self, prompt_name, model_name, temperature, data_name, summary_path, result_path):
        self.llm = ChatOpenAI(
            temperature=temperature,
            max_tokens=2048,
            model_name=model_name,
            api_key=get_openai_key()
        )
        self.retriever = Retriever(3)
        self.template = load_prompt(prompt_name)
        self.episode = load_episode(data_name)
        self.summary = load_summary(summary_path)
        self.result_path = result_path
        self.init_memory(self.summary)
        
    def init_memory(self, memory):
        self.linked_memory = {}
        for key in memory:
            self.linked_memory[key] = {}
        return
        
    def generate_gpt_response(self, text, input_variables):
        with get_openai_callback() as cb:
            prompt = PromptTemplate(template=self.template, input_variables=input_variables)
            llm_chain = LLMChain(prompt=prompt, llm=self.llm)
            result = llm_chain.apply(text)
            cost = cb.total_cost
        return result[0]['text'], cost

    def extract_relations(self, sentence1, sentence2, dialogue1, dialogue2):
        input = [{
            'dialogue1': dialogue1, 'dialogue2': dialogue2, 
            'sentence1': sentence1, 'sentence2': sentence2
        }]
        input_variables = list(input[0].keys())
        return self.generate_gpt_response(input, input_variables)
    
    def find_links(self, target_key, target_text, retrieved_memory):
        total_cost = 0
        raw_output = []
        retrieved_memory, key_lst = to_dic(retrieved_memory)
        key_lst.sort(key=lambda x: int(x[1]), reverse=True)
        while len(key_lst)>0:
            print(f"linking {target_key} with {key_lst[0]} ...")
            sentence1 = retrieved_memory[key_lst[0]]
            sentence2 = target_text['text']
            dialogue1 = get_dialogue(self.episode, int(key_lst[0][1]))
            dialogue2 = get_dialogue(self.episode, int(target_key[1]))
            result, cost = self.extract_relations(sentence1, sentence2, dialogue1, dialogue2)
            total_cost += cost
            relation = result.split('- Relation:')[1].strip()
            raw_output.append({
                'sentence1_id': key_lst[0], 
                'sentence2_id': target_key, 
                'sentence1': sentence1, 
                'sentence2': sentence2, 
                'relation': relation, 
                'raw_output': result, 
                'dialogue1': dialogue1, 
                'dialogue2': dialogue2
            })
            
            if "None" not in relation:
                self.linked_memory[target_key][key_lst[0]] = relation
                self.linked_memory[key_lst[0]][target_key] = relation
            key_lst.remove(key_lst[0])
        return raw_output, total_cost
    
    def find_all_links(self, session_num, memory_embedding):
        total_cost = 0
        output = {}
        memory_cur = self.retriever.get_cur_memory(session_num, memory_embedding)
                
        for target_key, target_text in memory_cur.items():
            retrieved_memory = self.retriever.retrieve_for_linking(session_num, target_key, memory_embedding)
            raw_output, cost = self.find_links(target_key, target_text, retrieved_memory)
            output[target_key] = raw_output
            total_cost += cost
            print(f"{target_key} done, cost: {round(total_cost, 4)}")
        return output, total_cost

    def linking(self):
        total_output = {}
        total_cost = 0
        memory_embedding = self.retriever.memory_to_embedding(self.summary)
        
        for session_num in range(2,5): 
            output, cost = self.find_all_links(session_num, memory_embedding)
            total_output[f"s{session_num}"] = output
            total_cost += cost
            print(f"## Session {session_num} done, cost: {total_cost} ##")
   
    def save(self):
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        with open(f'{self.result_path}/linked_memory.json', 'w', encoding="UTF-8") as f : 
            json.dump(self.linked_memory, f, indent=4, ensure_ascii=False)
        return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_name', type=str, default='relation-extraction.txt')
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--data_name', type=str, default='sample_dialogue.json')
    parser.add_argument('--summary_path', type=str, default='summary.json')
    parser.add_argument('--result_path', type=str, default='results/memory')
    args = parser.parse_args()

    memory_constructor = MemoryConstructor(
        prompt_name=args.prompt_name,
        model_name=args.model_name,
        temperature=args.temperature,
        data_name=args.data_name,
        summary_path=args.summary_path,
        result_path=args.result_path
    )
    memory_constructor.linking()
    memory_constructor.save()


if __name__ == "__main__":
    main()
import json
import random
import os
import argparse

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback
from src.retriever import Retriever
from utils.config import get_openai_key
from utils.utils import get_dialogue_
from utils.path import (
    load_prompt,
    load_episode,
    load_summary,
    load_memory,
    get_save_path
)

class Theanine:
    def __init__(self, prompt_refine, prompt_rg, model_name, temperature, data_name, summary_path, linked_memory_path):
        self.llm = ChatOpenAI(
            temperature=temperature,
            max_tokens=2048,
            model_name=model_name,
            api_key=get_openai_key()
        )
        self.retriever = Retriever(3)
        self.refine_template = load_prompt(prompt_refine)
        self.response_template = load_prompt(prompt_rg)
        self.episode = load_episode(data_name)
        self.summary = load_summary(summary_path)
        self.linked_memory = load_memory(linked_memory_path)

    def generate_gpt_response(self, input, template, input_variables):
        with get_openai_callback() as cb:
            prompt = PromptTemplate(template=template, input_variables=input_variables)
            llm_chain = LLMChain(prompt = prompt, llm = self.llm)
            result = llm_chain.apply(input)
            cost = cb.total_cost
        return result[0]['text'], cost

    def generate_response(self, input_memory, current_dialogue, speaker):
        memory_text = ""
        for i, memory in enumerate(input_memory):
            memory_text += f"{i+1}: {memory}\n"
        input = [{
            "memory_text": memory_text.strip(), 
            "current_dialogue": current_dialogue, 
            "speaker": speaker
        }]
        input_variables = list(input[0].keys())
        result, cost = self.generate_gpt_response(input, self.response_template, input_variables)
        return result, cost

    def get_path_text(self, path):
        text = ""
        for i, element in enumerate(path):
            if i % 2 == 0:
                text += f"[{self.summary[element]}] - "
            else:
                text += f"({element}) - "
        return text[:-2]

    def get_all_path(self, search_node, session_num):
        future_paths = []
        future_search = []
        past_paths = []
        past_search = []
        all_path = []

        ### initialize ###
        memory_past = {}
        memory_future = {}
        for key in self.summary:
            memory_past[key] = {}
            memory_future[key] = {}
        for node in self.linked_memory:
            if int(node[1])>=session_num:
                continue
            for sub_node in self.linked_memory[node]:
                if int(sub_node[1])>=session_num:
                    continue
                if int(node[1])>int(sub_node[1]):
                    memory_past[node][sub_node] = self.linked_memory[node][sub_node]
                else:
                    memory_future[node][sub_node] = self.linked_memory[node][sub_node]
                    
        for head in list(memory_past[search_node].keys()):
            past_search.append((head, memory_past[search_node][head], search_node))
        while len(past_search)>0:
            search_path = past_search[0]
            first_head = search_path[0]
            next_heads = list(memory_past[first_head].keys())
            if len(next_heads)==0:
                past_paths.append(search_path)
                past_search = past_search[1:]
            else:
                for next_head in next_heads:
                    new_path = (next_head, memory_past[first_head][next_head], ) + search_path
                    past_search.append(new_path)
                past_search = past_search[1:]
                
        for tail in list(memory_future[search_node].keys()):
            future_search.append((search_node, memory_future[search_node][tail], tail))
        while len(future_search)>0:
            search_path = future_search[0]
            last_tail = search_path[-1]
            next_tails = list(memory_future[last_tail].keys())
            if len(next_tails)==0:
                future_paths.append(search_path)
                future_search = future_search[1:]
            else:
                for next_tail in next_tails:
                    new_path = search_path + (memory_future[last_tail][next_tail], next_tail, )
                    future_search.append(new_path)
                future_search = future_search[1:]

        if future_paths==[] and past_paths==[]:
            all_path.append([search_node])
        elif len(future_paths)==0:
            all_path += past_paths
        elif len(past_paths)==0:
            all_path += future_paths
        else:
            for past_path in past_paths:
                for future_path in future_paths:
                    all_path.append(past_path[:-1]+future_path)
        return all_path
    
    def retrieve_timeline(self, query, memory, session_num):
        retrieved_nodes = self.retriever.retrieve_nodes(query, memory)
        retrieved_nodes.sort(key = lambda x:int(x[0][1]), reverse=True)
        timeline = []
        use_timeline = []
        all_timeline = {}
        for search_node, _ in retrieved_nodes:
            all_path = self.get_all_path(search_node, session_num)
            timeline.append({"retrieved_node": search_node,  "all_timeline": all_path})
            nothing_to_add = False
            all_path_ = all_path.copy()
            while nothing_to_add == False:
                choosed_path = random.sample(all_path_, k=1)
                all_path_.remove(choosed_path[0])
                if choosed_path[0] in use_timeline:
                    nothing_to_add = False
                else:
                    use_timeline.append(choosed_path[0])
                    nothing_to_add = True
                if len(all_path_)==0:
                    nothing_to_add = True
        all_timeline["retrieved_nodes"] = retrieved_nodes
        all_timeline["use_timeline"] = use_timeline
        all_timeline["timeline"] = timeline
        return all_timeline
    
    def link_refinement(self, current_dialogue, input_path):
        input = [{'current_dialogue': current_dialogue, 'input_path': input_path}]
        input_variables = list(input[0].keys())
        result, cost = self.generate_gpt_response(input, self.refine_template, input_variables)
        return result, cost

    def theanine_all(self, session_num):
        total_cost = 0
        cost = 0
        result_dict = {}
        current_dialogue_lst = []
        current_dialogue = ""
        
        dialogue, speakers = get_dialogue_(self.episode, session_num)
        memory_embedding = self.retriever.memory_to_embedding(self.summary)

        for turn_num in range(1, len(dialogue)):
            line = f"{speakers[turn_num-1]}: {dialogue[turn_num-1]}"
            current_dialogue_lst.append(line)
            current_dialogue += f"{line}\n"
            speaker = speakers[turn_num]
            
            ### retrieve timeline ###
            timelines = self.retrieve_timeline(
                query=current_dialogue, 
                memory=memory_embedding, 
                session_num=session_num
            )
            
            ### timeline refinement ###
            input_memory = []
            input_path_lst = []
            for timeline in timelines["use_timeline"]:
                input_path = self.get_path_text(timeline)
                input_path_lst.append(input_path)
                result, cost = self.link_refinement(current_dialogue, input_path)
                input_memory.append(result)
            total_cost += cost
            print(f"blending cost: {total_cost}")

            ### response generation ###
            result, cost = self.generate_response(input_memory, current_dialogue.strip(), speaker)
            response = result.split("[Response]:")[-1].strip()
            total_cost += cost
            result_dict[f"s{session_num}-t{turn_num}"] = {
                "current_dialogue": current_dialogue.strip().split("\n"), 
                "response": response, 
                "input_memory_num": len(input_memory),
                "before_refinement": input_path_lst,
                "after_refinement": input_memory, 
                "raw_response_output": result, 
                # "timelines": timelines
            }
            print(f"[Response]: {response}")
            print(f"t{turn_num} done, cost: {total_cost}")
        return result_dict, total_cost

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session_num', type=int, default=5)
    parser.add_argument('--result_path', type=str, default='results/theanine')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--data_name', type=str, default='sample_dialogue.json')
    parser.add_argument('--summary_path', type=str, default='summary.json')
    parser.add_argument('--linked_memory_path', type=str, default='linked_memory.json')
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--prompt_refine', type=str, default='timeline-refinement.txt')
    parser.add_argument('--prompt_rg', type=str, default='response-generation.txt')
    args = parser.parse_args()

    theanine = Theanine(
        prompt_refine=args.prompt_refine,
        prompt_rg=args.prompt_rg,
        temperature=args.temperature,
        data_name=args.data_name,
        summary_path=args.summary_path,
        linked_memory_path=args.linked_memory_path,
        model_name=args.model_name)
    result, cost = theanine.theanine_all(session_num=args.session_num)
    print(f"Total_cost: {cost}")
    
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    with open(f'{args.result_path}/response_s{args.session_num}.json', 'w', encoding="UTF-8") as f : 
        json.dump(result, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
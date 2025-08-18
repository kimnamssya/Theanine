import json
import re
import argparse
import os

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback

from utils.utils import get_dialogue
from utils.config import get_openai_key
from utils.path import (
    load_prompt,
    load_episode,
    get_save_path
)

class Summarizer:
    def __init__(self, prompt_name, model_name, temperature, data_name, result_path):
        self.llm = ChatOpenAI(
            temperature=temperature,
            model_name=model_name,
            api_key=get_openai_key()
        )
        self.template = load_prompt(prompt_name)
        self.episode = load_episode(data_name)
        self.result_path = result_path
        
    def process_text(self, result):
        refined_summary = []
        self.pattern = re.compile(r"[0-9]+\.\s")
        result = result.split("\n")
        for text in result:
            refined_text = self.pattern.sub("", text)
            refined_summary.append(refined_text)
        return refined_summary
    
    def create_node(self, summary, session_num):
        node = {}
        for idx, summary_sample in enumerate(summary):
            key = f"s{session_num}-m{idx+1}"
            node[key] = summary_sample
        return node

    def generate_gpt_response(self, input, input_variables):
        with get_openai_callback() as cb:
            prompt = PromptTemplate(template=self.template, input_variables=input_variables)
            llm_chain = LLMChain(prompt=prompt, llm=self.llm)
            result = llm_chain.apply(input)
            cost = cb.total_cost
        return result[0]['text'], cost

    def summarize(self, dialogue, session_num):
        summary_, _ = self.generate_gpt_response(
            input=[{'dialogue': dialogue}], 
            input_variables=['dialogue']
        )
        summary = self.process_text(summary_)
        return self.create_node(summary, session_num)
    
    def summarize_all_session(self):
        summary_all = {}
        ### TODO: Dynamic session number
        for session_num in range(1, 5):
            dialogue = get_dialogue(self.episode, session_num)
            summary = self.summarize(dialogue, session_num)
            summary_all.update(summary)
        return summary_all
    
    def save(self, summary):
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        with open(f'{self.result_path}/summary.json', 'w', encoding="UTF-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_name', type=str, default='dialogue-summarization.txt')
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--data_name', type=str, default='sample_dialogue.json')
    parser.add_argument('--result_path', type=str, default='results/memory')
    args = parser.parse_args()

    summarizer = Summarizer(
        prompt_name=args.prompt_name,
        model_name=args.model_name,
        temperature=args.temperature,
        data_name=args.data_name,
        result_path=args.result_path
    )
    summary = summarizer.summarize_all_session()
    summarizer.save(summary)

if __name__ == "__main__":
    main()
import os
import json


def get_dialogue_(dialogue, session_num):
    session_names = ["first", "second", "third", "fourth", "fifth"]
    dialogue_lst = dialogue[f"{session_names[session_num-1]}_session_dialogue"]
    speakers = dialogue[f"{session_names[session_num-1]}_session_speakers"]
    return dialogue_lst, speakers


def get_dialogue(episode, session_num) -> str:
    session_names = ["first", "second", "third", "fourth", "fifth"]
    dialogue = episode[f"{session_names[session_num-1]}_session_dialogue"]
    speakers = episode[f"{session_names[session_num-1]}_session_speakers"]
    dialogue_input = ""
    for idx, sentence in enumerate(dialogue):
        dialogue_input += f"{speakers[idx]}: {sentence}\n"
    return dialogue_input.strip()


def to_dic(retrieved_memory):
    retrieved_memory_dic = {}
    key_lst = []
    for d in retrieved_memory:
        key = list(d.keys())[0]
        retrieved_memory_dic[key] = d[key]
        key_lst.append(key)
    return retrieved_memory_dic, key_lst

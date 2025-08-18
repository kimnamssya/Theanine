import json
import os

from pathlib import Path


def get_project_path():
    return Path(__file__).parents[1]


def get_episode_path(episode_name: str) -> Path:
    return get_project_path() / 'resources' / 'data' / episode_name


def get_summary_path(summary_name: str) -> Path:
    return get_project_path() / 'results' / 'memory' / summary_name


def get_memory_path(memory_name: str) -> Path:
    return get_project_path() / 'results' / 'memory' / memory_name


def get_save_path() -> Path:
    save_path = get_project_path() / 'results' / 'memory'
    if not save_path.exists():
        os.makedirs(save_path)
    return save_path


def get_output_path(dataset_type, experiment_type) -> Path:
    output_path = get_project_path() / 'results' / dataset_type / experiment_type
    if not output_path.exists():
        os.makedirs(output_path)
    return output_path


def read_text_file(file_path):
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if file_path.exists() and file_path.is_file():
        with file_path.open(mode='r', encoding='utf-8') as f:
            content = f.read()
        return content
    else:
        print(f"File '{file_path}' does not exist or is not a regular file.")
        return None


def load_prompt(template_name: str):
    prompt_path = get_project_path() / 'resources' / 'prompts' / template_name
    return read_text_file(prompt_path)


def load_episode(episode_name: str):
    episode_path = get_episode_path(episode_name)
    with open(episode_path, 'r') as f:
        episode = json.load(f)
    return episode[0]


def load_summary(summary_name: str):
    summary_path = get_summary_path(summary_name)
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    return summary


def load_memory(memory_name: str):
    memory_path = get_memory_path(memory_name)
    with open(memory_path, 'r') as f:
        memory = json.load(f)
    return memory





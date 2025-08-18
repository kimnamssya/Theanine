#!/bin/bash

# Run the summarization script
python src/summarize.py \
    --prompt_name "dialogue-summarization.txt" \
    --model_name "gpt-3.5-turbo" \
    --temperature 0.7 \
    --data_name "sample_dialogue.json" \
    --result_path "results/memory"

# Run the relation-aware memory linking script
python src/memory_constructor.py \
    --prompt_name "relation-extraction.txt" \
    --model_name "gpt-3.5-turbo" \
    --temperature 0.5 \
    --data_name "sample_dialogue.json" \
    --summary_path "summary.json" \
    --result_path "results/memory"

# Run the context-aware timeline refinement and response generation scripts
python src/theanine.py \
    --session_num 5 \
    --model_name "gpt-3.5-turbo" \
    --temperature 0.7 \
    --prompt_refine "timeline-refinement.txt" \
    --prompt_rg "response-generation.txt" \
    --data_name "sample_dialogue.json" \
    --summary_path "summary.json" \
    --linked_memory_path "linked_memory.json" \
    --result_path "results/theanine"
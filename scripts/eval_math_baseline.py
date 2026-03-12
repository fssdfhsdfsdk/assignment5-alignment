from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from typing import Callable, List, Tuple
import pandas as pd
import re
from collections import Counter
import json

r1_zero_prompt_path= r"../cs336_alignment/prompts/r1_zero.prompt"
gsm8k_train_path = r"../data/gsm8k/train.jsonl"
gsm8k_test_path = r"../data/gsm8k/test.jsonl"
MODEL_NAME = "../models/Qwen2.5-Math-1.5B"

def load_prompt(path: str)->str:
    with open(path, 'r') as f:
        return f.read()

r1_zero_prompt = load_prompt(r1_zero_prompt_path)

def load_jsonl_data_to_prompts(path: str, promp_templat: str) -> Tuple[List[str], List[str]]:
    df = pd.read_json(path, lines=True)

    prompts = []
    answers_ground_truth = []
    for row in df.itertuples():
        prompts.append(promp_templat.format(question=row.question))
        answers_ground_truth.append(re.split(r"\W####\W", row.answer)[1])
    
    return prompts, answers_ground_truth

def get_answer_types(ans: dict):
        """
        {
            "format_reward": 0.0,
            "answer_reward": 0.0,
            "reward": 0.0
        }
        return: 
          0: format and answer ok
          1: format ok, answer error
          2: both error
        """
        if ans['reward'] == '1.0':
             return 0
        elif ans['format_reward'] == '1.0':
             return 1
        else:
             return 2

def evaluate_llm(vllm_model: LLM, 
                 reward_fn: Callable[[str, str], dict[str, float]],
                 prompts: List[str],
                 answers: List[str],
                 eval_samp_params: SamplingParams):
    """
    Evaluate a language model on a list of prompts.
    compute evaluation metrics, and serialize results to disk.
    """

    outputs = llm.generate(prompts, eval_samp_params)

    type_stat = Counter()
    type_list = []
    for idx, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        reward_dc = reward_fn(generated_text, answers[idx])
        tp = get_answer_types(reward_dc)
        type_stat[tp] += 1
        type_list.append(tp)

    print("Answer Type Statistics:")
    # 打印占比
    total = sum(type_stat.values())
    for tp, count in type_stat.items():
        print(f"Type {tp}: {count} ({count/total:.2%})")

    # write output and type to josonl
    with open("eval_results.jsonl", "w") as f:
        for idx, output in enumerate(outputs):
            tp = type_list[idx]
            generated_text = output.outputs[0].text
            f.write(json.dumps({
                "generated_text": generated_text,
                "type": tp
            }) + "\n")

if __name__ == "__main__":
    prompts, answers = load_jsonl_data_to_prompts(gsm8k_test_path, r1_zero_prompt)
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, 
                                     max_tokens=1024, stop=["</answer>"])
    sampling_params.include_stop_str_in_output = True
    """
    ValueError: Bfloat16 is only supported on GPUs with compute capability of at least 8.0. 
    Your Tesla V100-SXM2-32GB GPU has compute capability 7.0. 
    You can use float16 instead by explicitly setting the`dtype` flag in CLI, 
    for example: --dtype=half.
    """
    llm = LLM(model=MODEL_NAME, dtype="half")

    
    evaluate_llm(llm, r1_zero_reward_fn, prompts, answers, sampling_params)

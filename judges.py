from fastchat.model import (
    get_conversation_template
)
import re
from system_prompts import get_judge_system_prompt

from language_models import GPT

from utils import get_model_path_and_template
from language_models import HuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_judge(args):
    if "gpt" in args.judge_model:
        return GPTJudge(args)
    # 开源模型的judge
    elif args.judge_model in ["vicuna", "llama-2"]:
        return LocalJudge(args)
    elif args.judge_model == "no-judge":
        return NoJudge(args)
    else:
        raise NotImplementedError(f"Judge model {args.judge_model} not supported.")

class JudgeBase:
    def __init__(self, args):
        self.max_n_tokens = args.judge_max_n_tokens
        self.temperature = args.judge_temperature
        self.judge_name = args.judge_model
        self.goal = args.goal
        self.target_str = args.target_str
        self.system_prompt = get_judge_system_prompt(args.goal, args.target_str)

    def get_judge_prompt(self, attack_prompt, target_response):
        return f"[PROMPT]:{attack_prompt}\n[ASSISTANT'S RESPONSE]: {target_response}"

    def score(self, prompt_list, response_list):
        raise NotImplementedError

    def process_output(self, raw_output):
        pattern = r'\[\[(\d+)\]\]'
        match = re.search(pattern, raw_output)
        output = int(match.group(1)) if match else None
        if output is None:
            print(f"Error in processing judge output: {raw_output}" )
            output = 1
        return output
               
class NoJudge(JudgeBase):
    def __init__(self, args):
        super(NoJudge, self).__init__(args)
        self.judge_model = None

    def score(self, prompt_list, response_list):
        return [1 for _ in prompt_list]

class GPTJudge(JudgeBase):
    def __init__(self, args):
        super(GPTJudge, self).__init__(args)
        self.judge_model = GPT(model_name = self.judge_name)

    def create_conv(self, full_prompt):
        conv = get_conversation_template(self.judge_name)
        conv.set_system_message(self.system_prompt)
        conv.append_message(conv.roles[0], full_prompt)
        return conv.to_openai_api_messages()

    def score(self, attack_prompt_list, target_response_list):
        convs_list = [self.create_conv(self.get_judge_prompt(prompt, response)) for prompt, response in zip(attack_prompt_list, target_response_list)]
        raw_outputs = self.judge_model.batched_generate(convs_list, 
                                                        max_n_tokens = self.max_n_tokens,
                                                        temperature = self.temperature)
        outputs = [self.process_output(raw_output) for raw_output in raw_outputs]
        return outputs

class LocalJudge(JudgeBase):
    def __init__(self, args):
        super(LocalJudge, self).__init__(args)
        model_path, _ = get_model_path_and_template(self.judge_name)
        self.device = torch.device("cuda:1")  # 固定单卡
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(self.device).eval()

        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        self.judge_model = HuggingFace(self.judge_name, model, tokenizer)

    def score(self, attack_prompt_list, target_response_list):
        prompts = [
            self.get_judge_prompt(prompt, response)
            for prompt, response in zip(attack_prompt_list, target_response_list)
        ]
        full_prompts = [
            f"{self.system_prompt}\n{p}" for p in prompts
        ]
        raw_outputs = self.judge_model.batched_generate(
            full_prompts,
            max_n_tokens=self.max_n_tokens,
            temperature=self.temperature,
            top_p=1.0
        )
        outputs = [self.process_output(raw_output) for raw_output in raw_outputs]
        return outputs

#from vllm import LLM, SamplingParams
from openai import AsyncOpenAI
from collections import defaultdict
import json 
from colorama import Fore, Style
import numpy as np 

SYSTEM_PROMPT = """You are Interpretzel, an expert free-text classifier. Whenever you receive biological sample metada and a 
category you carefully analyze whether the metadata fits the category and reply with a simple YES or NO. 
You focus on where the sample is taken from and ignore the rest regardless of whether it fits a category.
You are very objective, you do not make assumptions about the data, neither do you have any bias. 
However, you do realize you are standardizing free text fields that might have tiny spelling mistakes which you try to interpret.
"""


class LLMPred:


    def __init__(self, model_name = "meta-llama/Meta-Llama-3-8B-Instruct", verbose = False, max_tokens = 2, base_url ="http://localhost:8000/v1"):
        self.model_name = model_name
        self.client = AsyncOpenAI(
            api_key="EMPTY",
            base_url= base_url,
        )
        self.verbose = verbose 
        self.max_tokens = max_tokens
        print(f"Verbose: {self.verbose}")
    
    def _read_queries(self, file_path):
        unique_queries = set()
        total = 0
        with open(file_path, "r") as in_file:
            for line in in_file:
                unique_queries.add(line.strip())
                total += 1
        if total != len(unique_queries):
            print('[WARNING] We only kept unique items in your query file')
        return unique_queries

    def _read_json_descriptions(self, desc_json):
        with open(desc_json) as in_file:
            desc = json.load(in_file)
            print(f'[INFO] Loaded {len(desc)} descriptons')
            return desc 
        
    async def _get_prompt(self, query, cat):
        return f"""You are an expert in categorizing biological samples. Given the following metadata description, determine if the sample belongs to the category specified. Respond only with "YES" or "NO".

Metadata: "{query}"
Category: "{cat}"

The metadata is clearly a child of the category: [YES or NO]?
""" 

    async def process_reply(self, reply):
        # Extract answer with prob
        answers = dict()
        for top in reply.choices[0].logprobs.top_logprobs:
            for answer, prob in top.items():
                # Log prob to linear prob
                answers[answer.lower()] = np.round(np.exp(prob)*100,2)   
        if self.verbose:
            print(answers)
        # Check how much  more likely one is over the other
        yes_prob =  answers.get("yes", 0)
        no_prob = answers.get("no", 0)
        return {"yes":yes_prob, "no":no_prob}

    async def _prompt_llm(self, prompt):
        completion = await self.client.chat.completions.create(
            model = self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content":prompt},
                ],
            logprobs=True, 
            top_logprobs=2,
            max_tokens=2
        )
        return await self.process_reply(completion)

    def _write_output(self, preds, output_file):
        with open(output_file, "w") as out:
            for query, pred_list in preds.items():
                class_str = ", ".join(pred_list)
                out.write(f"{query}\t{class_str}\n")
        
    async def predict(self, query_file, class_file, output_file, cut_off=99.9):
        predictions = defaultdict(list)
        queries = self._read_queries(query_file)
        class_json = self._read_json_descriptions(class_file)
        total = len(queries)
        for (i, query) in enumerate(queries):
             if self.verbose: 
                 print(Fore.MAGENTA + f"({i}/{total}): " + query + Style.RESET_ALL) 
             for class_name, desc in class_json.items():
                class_name = class_name.strip().lstrip("#")
                prompt = await self._get_prompt(query, class_name)
                yes_no_prob = await self._prompt_llm(prompt)
                if yes_no_prob["yes"] > cut_off:
                    if self.verbose: print(Fore.GREEN + f"\t{class_name}: {yes_no_prob['yes']}%" + Style.RESET_ALL)
                    predictions[query].append(class_name)
        self._write_output(predictions, output_file)
    
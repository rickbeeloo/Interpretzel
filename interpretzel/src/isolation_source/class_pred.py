
#from vllm import LLM, SamplingParams
from openai import AsyncOpenAI
from collections import defaultdict
import json 
from colorama import Fore, Style
import numpy as np 
import time
import csv
from transformers import AutoTokenizer

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
        if self.verbose.count("v") >=1:
            print(f"Model: {self.model_name}")
            print(f"Max tokens: {self.max_tokens}")
            print(f"Client url: {base_url}")

    def _get_meaningless_token_bias(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        bias_dict = dict()
        for exclude in [" ", ".", "\n", "\n\n"]:
            token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(exclude))
            bias_dict[token_id] = -100
        if self.verbose.count("v") >= 2:
            print("Bias tokens added")
            for token, bias in bias_dict:
                print(f"{token} => {bias}")
        return bias_dict

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

    def _read_class_info(self, file_path):
        class_info = dict()
        with open(file_path, "r") as in_file:
            reader = csv.DictReader(in_file, delimiter = "\t")
            for row in reader:
                class_info[row["class_name"]] = row.get("desc", "") # Description can be empty
        return class_info
        
    async def _get_prompt(self, query, cat, desc):
        return f"""You are an expert in categorizing biological samples. Given the following metadata description, determine if the sample belongs to the category specified. Respond only with "YES" or "NO".

Metadata: "{query}"
Category: "{cat}"
Category description: "{desc}"


The metadata is clearly a child of the category?
""" 

    async def process_reply(self, reply):
        # Extract answer with prob
        answers = dict()
        for top in reply.choices[0].logprobs.top_logprobs:
            for answer, prob in top.items():
                # Log prob to linear prob
                answers[answer.lower()] = np.round(np.exp(prob)*100,2)   
        # Check how much  more likely one is over the other
        yes_prob =  answers.get("yes", 0)
        no_prob = answers.get("no", 0)
        if self.verbose.count('v') >=2:
            print(answers)
        return {"yes":yes_prob, "no":no_prob}

    async def _prompt_llm(self, prompt, bias):
        completion = await self.client.chat.completions.create(
            model = self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content":prompt},
                ],
            logprobs=True, 
            top_logprobs=2,
            max_tokens=2, 
            logit_bias=bias
           # logit_bias={220:-100, 198:-100, 271:-100}
        )
        return await self.process_reply(completion)

    def _write_output(self, preds, output_file):
        with open(output_file, "w") as out:
            for query, pred_list in preds.items():
                query = query.replace("\t", " ") # Make sure to not mess us tab delim
                class_str = ", ".join(pred_list)
                out.write(f"{query}\t{class_str}\n")

    async def single_prediction(self, query, class_name, desc, bias, cut_off):
        class_name = class_name.strip().lstrip("#")
        prompt = await self._get_prompt(query, class_name, desc)
        yes_no_prob = await self._prompt_llm(prompt, bias)

        if self.verbose.count("v") >= 2:
            print(f"{class_name}: {yes_no_prob}")

        if yes_no_prob["yes"] > cut_off:    
            if self.verbose.count("v") >=1: 
                print(Fore.GREEN + f"\t{class_name}: {yes_no_prob['yes']}%" + Style.RESET_ALL)
            else:
                print(Fore.RED + f"\t{class_name}: {yes_no_prob['no']}%" + Style.RESET_ALL)
            return True, yes_no_prob['yes']
        return False, 0
            

    async def run_predictions(self, queries, class_json, bias, cut_off):
        total = len(quit)
        predictions = defaultdict(list)
        for (i, query) in enumerate(queries):
            s = time.time()
            
            if self.verbose.count("v") >=1 :
                print(Fore.MAGENTA + f"({i}/{total}): " + query + Style.RESET_ALL) 

            for class_name, desc in class_json.items():
                passed, prob = self.single_prediction(query, class_name, desc, bias, cut_off)
                if passed:
                    predictions[query].append(class_name + f"[{prob}]")

            took = round(time.time() - s, 2)
            if self.verbose.count("v") >=1: print(f"Took: {took} seconds.")
        return predictions

        
    async def predict(self, query_file, class_file, output_file, cut_off=99.9):
        queries = self._read_queries(query_file)
        class_json = self._read_class_info(class_file)
        bias = self._get_meaningless_token_bias()
        predictions = await self.run_predictions(queries, class_json, bias, cut_off)
        self._write_output(predictions, output_file)
    
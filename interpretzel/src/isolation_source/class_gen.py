
from vllm import LLM, SamplingParams
import csv
import json 

class InvalidExtension(Exception): pass
class InvalidColumns(Exception): pass

class DescGen:

    def __init__(self, model_name = "meta-llama/Meta-Llama-3-8B-Instruct", max_tokens = 100): 
        self.model_name = model_name
        self.llm = LLM(
            model= model_name, 
            enforce_eager=True,  # Don't compile the graph
            gpu_memory_utilization=0.99,
            max_model_len=1024,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.max_tokens = max_tokens
    

    def _read_example_file(self, file_path: str):
        print("[INFO] Reading input")
        data = dict()
        file_ext = file_path.split(".")[-1]
        if file_ext not in ["csv", "tsv"]: raise InvalidExtension
        delim = ',' if file_ext == "csv" else "\t"
        with open(file_path, "r") as in_file:
            reader = csv.DictReader(in_file, delimiter = delim)
            columns = list(reader.fieldnames)
            if "class_name" not in columns or "examples" not in columns:
                raise InvalidColumns
            for row in reader:
                data[row["class_name"]] = row["examples"]
        return data
                    
    def _get_prompt(self, data):
        print("[INFO] Generating prompts")
        query_mapping = dict()
        prompts = []
        i = 0
        for class_label, examples in data.items():
            q = f"Describe the category {class_label} in one sentence, provide two examples. Examples of samples wtihint this category are: {examples}"
            chat = [{'role': 'user', 'content': q}]    
            tokenized_chat = self.tokenizer.apply_chat_template(chat, tokenize=False)
            prompts.append(tokenized_chat)
            query_mapping[i] = class_label
            i += 1
        return prompts, query_mapping

    def _prompt_llm(self, prompts):
        print("[INFO] Querying LLM")
        outputs = self.llm.generate(
            prompts,
            SamplingParams(
                temperature=0,
                top_p=0.9,
                max_tokens=self.max_tokens, # An average sentence is 15-20, long is 30
                stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
            ),
            use_tqdm=False,
        )
        return outputs

    def _parse_output(self, outputs, request_mapping):
        parsed = dict()
        for output in outputs:
            req_id = int(output.request_id)
            class_name = request_mapping[req_id]
            desc = output.outputs[0].text.split(">")[-1].strip() # I case header tags are propagated
            parsed[class_name] = desc 
        return parsed
        
    def _write_output(self, data, output_file):
        with open(output_file, "w") as out: 
            json.dump(data, out)

    def process(self, example_file, output_file):
        data = self._read_example_file(example_file)
        prompts, request_mapping = self._get_prompt(data)
        outputs = self._prompt_llm(prompts)
        parsed = self._parse_output(outputs, request_mapping)
        self._write_output(parsed, output_file)


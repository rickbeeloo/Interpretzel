
from vllm import LLM, SamplingParams
from collections import defaultdict
import json 

class LLMPred:

    def __init__(self, model_name = "meta-llama/Meta-Llama-3-8B-Instruct", verbose = True, max_tokens = 10):
        self.model_name = model_name
        self.llm = LLM(
            model= model_name, 
            enforce_eager=True,  # Don't compile the graph
            gpu_memory_utilization=0.99,
            max_model_len=1024
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.verbose = verbose 
        self.na_fallback = "no"
        self.unclear = 0
        self.max_tokens = max_tokens
    
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
        
    def _get_templates(self, class_json):
        templates = dict()
        for class_name, desc in class_json.items():
            q = f'Given the description for the category "{class_name}" being: {desc}.\n\nIs a sample with the metadata "WILDCARD" clearly part of {class_name}? Reply with JUST YES or NO, no explanation.'
            q = q.replace("WILDCARD", "{}")
            templates[class_name] = q
        return templates # dict

    def _fill_templates(self, queries, templates):
        prompts = []
        req_mapping = dict()
        class_mapping = dict()
        i = 0
        for query in queries:
            for class_name, templ in templates.items():
                chat = [{"role":"user", "content":templ.format(query)}]
                tokenized_chat = self.tokenizer.apply_chat_template(chat, tokenize=False)
                prompts.append(tokenized_chat)
                req_mapping[i] = query
                class_mapping[i] = class_name
                i += 1
        return prompts, req_mapping, class_mapping
    
    def _prompt_llm(self, prompts):
        outputs = self.llm.generate(
            prompts,
            SamplingParams(
                temperature=1,
                top_p=1,
                repetition_penalty=1,
                top_k = 50,
                max_tokens=self.max_tokens, # 3 tokens should be more than enough, but to be safe this can be set higher
                stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
            ),
            use_tqdm=True,
        )
        return outputs

    def _check_answer(self, output):
        answer = output.outputs[0].text.split(">")[-1].strip().lower() # In case header tags are propagated
        if "yes" in answer:
            return "yes"
        if "no" in answer:
            return "no"
        else:
            self.unclear += 1
            return self.na_fallback
    
    def _process_outputs(self, outputs, request_mapping, class_mapping):
        predictions = defaultdict(list)
        for output in outputs:
            req_id = int(output.request_id)
            query = request_mapping[int(req_id)]
            class_name = class_mapping[int(req_id)]
            raw = output.outputs[0].text
            answer = self._check_answer(output)
            if self.verbose:
                emoji = "❌" if answer == "no" else "✅"
                print(f"Q: {query}\nP:{output.prompt}\nC:{class_name}\nA:{raw}\nVerdict:{emoji}\n")
            if answer == "yes":
                predictions[query].append(class_name)
        return predictions

    def _write_output(self, preds, output_file):
        with open(output_file, "w") as out:
            for query, pred_list in preds.items():
                class_str = ", ".join(pred_list)
                out.write(f"{query}\t{class_str}\n")
        print(f"Unclear predictions to fallback: {self.unclear}")

    def predict(self, query_file, class_file, output_file):
        queries = self._read_queries(query_file)
        class_json = self._read_json_descriptions(class_file)
        templates = self._get_templates(class_json)
        prompts, req_mapping, class_mapping = self._fill_templates(queries, templates)
        outputs = self._prompt_llm(prompts)
        predictions = self._process_outputs(outputs, req_mapping, class_mapping)
        self._write_output(predictions, output_file)


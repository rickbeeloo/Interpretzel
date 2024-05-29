from vllm import LLM, SamplingParams
import csv 
import re

class CountryPred:

    def __init__(self, model_name = "mistralai/Mistral-7B-Instruct-v0.2", iso3_file = "../../data/country_names/iso3.csv"):
        self.model_name = model_name
        self.llm = LLM(
            model= model_name, 
            enforce_eager=True,  # Don't compile the graph
            gpu_memory_utilization=0.99,
            max_model_len=1024,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.by_name, self.by_iso = self._read_iso3_file(iso3_file)

    def _removearticles(text):
        re.sub('\s+(a|an|and|the)(\s+)', '\2', text)

    def _read_iso3_file(self, iso3_file):
        by_name_map = dict()
        by_iso_map = dict()
        with open(iso3_file, "r") as in_file:
            reader = csv.DictReader(in_file, delimiter=",")
            for row in reader:
                by_name_map[row["name"]] = row 
                by_iso_map[row["alpha-3"]] = row
        return by_name_map, by_iso_map
    
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

    def _exact_match_pass(self, data, by="name"):
        assert by in ["name", "iso3"], "Invalid by option"
        mapped = dict()
        unmapped = []
        if by == "name":
            for q in data:
                clean_q = q.lower().strip()
                res = self.by_name.get(clean_q)
                if res: 
                    mapped[]
                



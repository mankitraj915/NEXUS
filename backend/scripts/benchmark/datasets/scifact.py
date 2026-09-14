import os
import json
import urllib.request
import zipfile
import ssl
from typing import Generator, Dict
from scripts.benchmark.datasets.base import DatasetProvider, CorpusItem, QueryItem

class SciFactDataset(DatasetProvider):
    def __init__(self, cache_dir: str = ".cache/scifact"):
        self.cache_dir = cache_dir
        self.url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
        
    def get_name(self) -> str:
        return "scifact"
        
    def get_version(self) -> str:
        return "1.0.0"
        
    def prepare(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        zip_path = os.path.join(self.cache_dir, "scifact.zip")
        data_dir = os.path.join(self.cache_dir, "scifact")
        
        if not os.path.exists(data_dir):
            print("Downloading SciFact dataset...")
            ssl._create_default_https_context = ssl._create_unverified_context
            urllib.request.urlretrieve(self.url, zip_path)
            print("Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.cache_dir)
                
    def iter_corpus(self) -> Generator[CorpusItem, None, None]:
        corpus_path = os.path.join(self.cache_dir, "scifact", "corpus.jsonl")
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                yield {
                    "id": str(data["_id"]),
                    "title": data.get("title", ""),
                    "text": data.get("text", ""),
                    "metadata": data.get("metadata", {})
                }
                
    def iter_queries(self) -> Generator[QueryItem, None, None]:
        queries_path = os.path.join(self.cache_dir, "scifact", "queries.jsonl")
        with open(queries_path, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                yield {
                    "id": str(data["_id"]),
                    "text": data.get("text", "")
                }
                
    def get_qrels(self) -> Dict[str, Dict[str, int]]:
        qrels_path = os.path.join(self.cache_dir, "scifact", "qrels", "test.tsv")
        qrels = {}
        with open(qrels_path, encoding="utf-8") as f:
            header = f.readline() # skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    continue
                qid, docid, score = parts
                qid = str(qid)
                docid = str(docid)
                score = int(score)
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][docid] = score
        return qrels

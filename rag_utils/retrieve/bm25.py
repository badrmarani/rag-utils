from dataclasses import dataclass, asdict
import pickle
from typing import Callable, List, Optional
import os
import json

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from .base import Vectorstore
from ..text_utils import preprocess_text


@dataclass
class BM25Config:
    path: Optional[str] = None
    b: float = 0.75
    k1: float = 1.2


def load_config(path: str) -> BM25Config:
    with open(os.path.join(path, "config.json"), "r") as f:
        config = BM25Config(**json.load(f))
    return config


def save_config(path: str, config: BM25Config):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(asdict(config), f)


class BM25(Vectorstore):
    def __init__(
        self,
        config: Optional[BM25Config] = None,
        preprocess_func: Callable[[str], List[str]] = preprocess_text,
        path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.preprocess_func = preprocess_func

        if config is not None:
            save_config(path=config.path, config=config)
        elif path is not None:
            config = load_config(path)
        else:
            raise ValueError()
        self.config = config

    def transform(self, documents: List[Document]):
        bm25_retriever = BM25Retriever.from_documents(
            documents=documents,
            preprocess_func=self.preprocess_func,
            bm25_params={"b": self.config.b, "k1": self.config.k1},
            k=20,
        )

        save_path = os.path.join(self.config.path, "db.index")
        with open(save_path, "wb") as f:
            pickle.dump(bm25_retriever, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        load_path = os.path.join(self.config.path, "db.index")
        with open(load_path, "rb") as f:
            bm25_retriever = pickle.load(f)
        return bm25_retriever

from dataclasses import asdict, dataclass
import json
import os
from typing import List, Literal, Optional

from langchain_milvus import Milvus
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from .base import Vectorstore
from ..torch_utils import torch_device


@dataclass
class MilvusConfig:
    embedding_model_name: str
    path: Optional[str] = None
    metric_type: Literal["COSINE"] = "COSINE"
    index_type: Literal["HNSW"] = "HNSW"
    M: int = 4
    ef: int = 500
    efConstruction: int = 400


def load_config(path: str) -> MilvusConfig:
    with open(os.path.join(path, "config.json"), "r") as f:
        config = MilvusConfig(**json.load(f))
    return config


def save_config(path: str, config: MilvusConfig):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(asdict(config), f)


class MilvusVectorstore(Vectorstore):
    def __init__(self, config: Optional[MilvusConfig] = None, path: Optional[str] = None) -> None:
        super().__init__()
        if config is not None:
            save_config(path=config.path, config=config)
        elif path is not None:
            config = load_config(path)
        else:
            raise ValueError()
        self.config = config

        self.embedding = HuggingFaceEmbeddings(
            model_name=config.embedding_model_name,
            model_kwargs={"device": torch_device(), "trust_remote_code": True},
        )
        self.index_params = {
            "metric_type": config.metric_type,
            "index_type": config.index_type,
            "params": {"M": config.M, "efConstruction": config.efConstruction},
        }
        self.search_params = {
            "metric_type": config.metric_type,
            "params": {"ef": config.ef},
        }

    def transform(self, documents: List[Document]):
        return Milvus.from_documents(
            documents=documents,
            embedding=self.embedding,
            collection_name="vectorstore",
            connection_args={"uri": os.path.join(self.config.path, "index.db")},
            index_params=self.index_params,
            search_params=self.search_params,
        )

    def load(self):
        return Milvus(
            embedding_function=self.embedding,
            collection_name="vectorstore",
            connection_args={"uri": os.path.join(self.config.path, "index.db")},
            index_params=self.index_params,
            search_params=self.search_params,
        )

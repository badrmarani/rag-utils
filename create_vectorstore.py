import argparse

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
import tiktoken

from rag_utils.retrieve.milvus import MilvusConfig, MilvusVectorstore

encoder = tiktoken.get_encoding("cl100k_base")


def num_tokens_text(text: str) -> int:
    return len(encoder.encode(text))


def main(
    dataframe_path: str,
    embedding_model_name: str,
    path: str,
    metric_type: str,
    index_type: str,
    M: int,
    ef: int,
    efConstruction: int,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    data = pd.read_csv(dataframe_path)
    data = data[data["content"].notna()]

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=num_tokens_text,
    )
    loader = DataFrameLoader(data, page_content_column="content")
    documents = loader.load_and_split(text_splitter=text_splitter)

    config = MilvusConfig(
        embedding_model_name=embedding_model_name,
        path=path,
        metric_type=metric_type,
        index_type=index_type,
        M=M,
        ef=ef,
        efConstruction=efConstruction,
    )
    vs = MilvusVectorstore(config=config)
    vs.transform(documents=documents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataframe-path", type=str, default="")
    parser.add_argument("--embedding-model-name", type=str)
    parser.add_argument("--path", type=str, default="results/vectorstores/")
    parser.add_argument("--metric-type", type=str, default="COSINE")
    parser.add_argument("--index-type", type=str, default="HNSW")
    parser.add_argument("--M", type=int, default=4)
    parser.add_argument("--ef", type=int, default=500)
    parser.add_argument("--efConstruction", type=int, default=400)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--chunk-overlap", type=int, default=20)
    config = parser.parse_args()

    main(
        dataframe_path=config.dataframe_path,
        embedding_model_name=config.embedding_model_name,
        path=config.path,
        metric_type=config.metric_type,
        index_type=config.index_type,
        M=config.M,
        ef=config.ef,
        efConstruction=config.efConstruction,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

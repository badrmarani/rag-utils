import ast
import random
from typing import Any, Optional

import pandas as pd
import tiktoken
from dotenv import load_dotenv
from langchain_community.document_loaders import DataFrameLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..llm_utils import load_openai_llm

load_dotenv()
encoder = tiktoken.get_encoding("cl100k_base")


def num_tokens_text(text: str) -> int:
    return len(encoder.encode(text))


def load_prompt(path: str, **kwargs: Any) -> str:
    with open(path, "r") as f:
        prompt = f.read()
    return prompt.format(**kwargs)


def generate(
    dataframe_path: str,
    system_prompt_filepath: str,
    num_samples: int,
    chunk_size: int,
    chunk_overlap: int,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
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
    documents = random.choices(documents, k=num_samples)
    contexts = [document.page_content for document in documents]

    llm = load_openai_llm("chat", temperature=0.0)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", load_prompt(system_prompt_filepath)),
            ("human", "{context}"),
        ]
    )

    chain = prompt | llm

    responses = chain.batch(inputs=contexts)
    responses = list(map(lambda x: ast.literal_eval(x.content), responses))
    output = pd.DataFrame(
        [
            (question, context)
            for context in contexts
            for response in responses
            for question in response
        ],
        columns=["questions", "contexts"],
    )

    if save_path is not None:
        output.to_csv(save_path, index=False)
    return output

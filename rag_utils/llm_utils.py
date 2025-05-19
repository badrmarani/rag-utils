import os
from typing import Any, List, Literal

from dotenv import load_dotenv
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAI,
    AzureOpenAIEmbeddings,
)

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", None)
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", None)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", None)


def capture_value_error(*args: List[Any]) -> None:
    for var_name, value in args:
        if value is None:
            raise ValueError(f"{var_name} environment variable is missing or empty.")


def load_openai_llm(ask_or_chat: Literal["chat", "ask"], **llm_kwargs: Any) -> BaseLanguageModel:
    capture_value_error(
        ("AZURE_OPENAI_API_KEY", AZURE_OPENAI_API_KEY),
        ("OPENAI_API_VERSION", OPENAI_API_VERSION),
        ("AZURE_OPENAI_ENDPOINT", AZURE_OPENAI_ENDPOINT),
    )
    if ask_or_chat == "ask":
        return AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            deployment_name="davinci",
            **llm_kwargs,
        )
    elif ask_or_chat == "chat":
        return AzureChatOpenAI(
            openai_api_version=OPENAI_API_VERSION,
            openai_api_key=AZURE_OPENAI_API_KEY,
            deployment_name="chat",
            **llm_kwargs,
        )
    else:
        raise ValueError("Invalid value for 'ask_or_chat'. Expected 'ask' or 'chat'.")


def load_openai_embedding(**kwargs: Any):
    return AzureOpenAIEmbeddings(
        openai_api_version=OPENAI_API_VERSION,
        openai_api_key=AZURE_OPENAI_API_KEY,
        azure_deployment="text-embedding-ada-002",
        **kwargs,
    )

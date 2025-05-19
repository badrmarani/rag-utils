from typing import Dict, Union
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import pandas as pd


def evaluate(dataframe_or_path: Union[str, pd.DataFrame], embedding_name: str) -> Dict[str, float]:
    if isinstance(dataframe_or_path, str):
        data = pd.read_csv(dataframe_or_path)
    elif not isinstance(dataframe_or_path, pd.DataFrame):
        raise ValueError()

    questions = dict(enumerate((data["questions"].tolist())))
    unique_contexts = data["contexts"].unique()
    contexts = dict(enumerate(unique_contexts.tolist()))

    context_to_doc_id = {context: doc_id for doc_id, context in enumerate(unique_contexts)}

    document_to_questions = {}
    for idx, row in data.iterrows():
        question_id = idx
        context = row["contexts"]
        document_id = context_to_doc_id[context]

        if document_id not in document_to_questions:
            document_to_questions[document_id] = set()

        document_to_questions[document_id].add(question_id)

    model = SentenceTransformer(embedding_name)

    ir_evaluator = InformationRetrievalEvaluator(
        queries=contexts,
        corpus=questions,
        relevant_docs=document_to_questions,
    )
    results = ir_evaluator(model)
    return results

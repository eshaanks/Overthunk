from __future__ import annotations


def example_to_text(example) -> str:
    """
    Convert a ProntoQA example into a canonical text string for embedding.

    We embed the problem itself, not the answer and not the reasoning trace.
    So we include:
    - the factual context
    - the query to be solved

    Returns
    -------
    str
        A stable text representation of the problem.
    """
    question = example.test_example.question.strip()
    query = example.test_example.query.strip()

    return f"Facts:\n{question}\n\nQuery:\n{query}"
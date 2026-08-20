"""
Prompt templates for the anime recommendation RAG chain.

Summary
-------
This module owns the single user-facing prompt used by the retrieval chain. The
retriever pulls candidate anime documents out of the vector store (each one a
flattened `combined_info` record: title, English title, type, score, genres and
synopsis) and this template turns them into a grounded recommendation.

The prompt enforces four behaviours, in priority order:

1. **Groundedness** — recommendations must come from the retrieved context. If
   the context does not support an answer, the model says so instead of
   inventing titles, scores or plot details.
2. **Count control** — three recommendations by default, but the model honours
   an explicit count in the question ("just one film", "give me five").
3. **Scope handling** — non-anime questions get a short, plain answer rather
   than a forced recommendation.
4. **Clarification** — genuinely underspecified requests get one follow-up
   question instead of a guess.

All instructions live *above* the final answer cue. Anything placed after
`Your response:` is read by the model as the start of its own output rather
than as a rule, which is a common and silent source of instruction drift.

Usage
-----
    from prompt import get_anime_prompt

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | get_anime_prompt()
        | llm
        | StrOutputParser()
    )
"""

from langchain_core.prompts import PromptTemplate

ANIME_RECOMMENDER_TEMPLATE = """You are an expert anime recommender. You help users find anime they will love, using only the catalogue entries provided to you.

CONTEXT
Each entry below is a catalogue record containing the title, English title, type, MAL score, genres and synopsis.

{context}

USER'S QUESTION
{question}

HOW TO RESPOND

If the question asks for anime recommendations:
- Recommend {num_recommendations} titles by default. If the user asks for a specific number ("just one", "give me five"), give exactly that many instead.
- Use only titles that appear in the context above. Never invent a title, score, episode count or plot detail.
- Present the recommendations as a numbered list. For each one include:
  1. The title, with the English title in parentheses if it differs.
  2. A concise plot summary of two to three sentences.
  3. A clear explanation of why it matches what the user asked for, referring to their stated preferences.
- If the user set a hard constraint (episode count, type, year, rating, studio), only recommend titles in the context that actually meet it. Recommend fewer titles rather than breaking a constraint, and say why the list is short.
- If the user asked for something family-friendly or free of adult content, exclude anything rated R+, Rx, or tagged Hentai, Erotica or Ecchi.

If the context does not contain anything that fits:
- Say plainly that you could not find a match in the catalogue. Do not substitute a title you happen to know but that is not in the context.

If the question is too vague to act on (for example "recommend something good"):
- Ask one short clarifying question about genre, mood or length instead of guessing.

If the question is not about anime:
- Answer it briefly and naturally in a sentence or two, without a numbered list, and without pretending the anime catalogue is relevant.

If the question asks for something you should not provide (piracy links, personal information about real people, reproduced scripts or lyrics, adult content):
- Decline in one sentence and offer a reasonable alternative.

Your response:"""


def get_anime_prompt(num_recommendations: int = 3) -> PromptTemplate:
    """Build the prompt template for the anime recommendation chain.

    Produces a grounded recommendation prompt that consumes retrieved catalogue
    documents and a user question, and steers the model toward a numbered list
    of titles drawn strictly from those documents.

    The template covers the non-recommendation paths as well: empty or
    irrelevant retrieval, vague questions, off-topic questions, and requests
    that should be refused. Those branches exist so that the chain degrades into
    an honest answer rather than a confident hallucination when retrieval
    returns nothing useful.

    Args:
        num_recommendations: Default number of titles to recommend when the user
            does not specify a count. Baked into the template at build time, so
            call this once per chain rather than per request. An explicit count
            in the user's question still overrides it.

    Returns:
        A ``PromptTemplate`` with input variables ``context`` and ``question``.
        ``context`` should be the retrieved documents joined into a single
        string (typically ``"\\n\\n".join(doc.page_content for doc in docs)``);
        ``question`` is the raw user query.

    Example:
        >>> prompt = get_anime_prompt()
        >>> prompt.input_variables
        ['context', 'question']
        >>> text = prompt.format(context="Title : Mononoke | ...", question="scary anime?")
    """
    return PromptTemplate(
        template=ANIME_RECOMMENDER_TEMPLATE.replace(
            "{num_recommendations}", str(num_recommendations)
        ),
        input_variables=["context", "question"],
    )
    
    
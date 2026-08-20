"""
Serving-side pipeline for the anime recommendation system.

Summary
-------
This is the runtime counterpart to the offline build. Where ``build_pipeline``
creates the vector store, this module loads it and answers queries against it.
Nothing here writes to disk or re-embeds the corpus — if the persisted index is
missing or stale, that is a build problem, not a serving one.

``AnimeRecommendationPipeline`` wraps the two runtime pieces behind a single
``recommend(query)`` call:

1. **Retriever** — ``VectorStoreBuilder`` loads the persisted Chroma index and
   exposes it as a LangChain retriever. ``csv_path`` is intentionally empty:
   the builder is being used in load mode, not build mode.
2. **Recommender** — ``AnimeRecommender`` owns the LLM and the prompt, and turns
   retrieved catalogue documents into a grounded, formatted recommendation.

Construction is deliberately expensive and query handling deliberately cheap.
Instantiate this once at application startup — module scope, a FastAPI
lifespan handler, or ``@st.cache_resource`` under Streamlit — and reuse it
across requests. Building one per request reloads the index and re-establishes
the model client every time.

Failure policy: both construction and querying wrap failures in
``CustomException``. An initialization failure means the app cannot serve at
all and should surface at startup rather than on the first user request; a
recommendation failure is per-query and leaves the pipeline usable.

Usage
-----
    pipeline = AnimeRecommendationPipeline()          # once, at startup
    answer = pipeline.recommend("something like Cowboy Bebop")
"""

from Config.config import MODEL_NAME
from Src.recommender import AnimeRecommender
from Src.vector_store import VectorStoreBuilder
from Utils.custom_exception import CustomException
from Utils.logger import get_logger

logger = get_logger(__name__)


class AnimeRecommendationPipeline:
    """Query-time entrypoint for anime recommendations.

    Holds a retriever backed by the persisted vector store and an
    ``AnimeRecommender`` that turns retrieved documents into a written answer.
    The object is stateless across calls — no conversation history is kept, so
    each query is answered independently of the ones before it.

    Attributes:
        recommender: The configured ``AnimeRecommender`` handling retrieval and
            generation for every query.
    """

    def __init__(self, persist_dir: str = "chroma_db") -> None:
        """Load the persisted vector store and wire up the recommender.

        Args:
            persist_dir: Directory holding the Chroma index written by the
                offline build. Must match the path the build pipeline wrote to,
                or the retriever will come up empty and every query will return
                a "no matches found" answer rather than an error.

        Raises:
            CustomException: If the vector store cannot be loaded or the
                recommender cannot be constructed, with the original exception
                chained. Common causes are a missing or empty ``persist_dir``
                (the build never ran) and absent model credentials.
        """
        try:
            logger.info("Initializing Recommendation Pipeline")

            # csv_path is empty by design — load mode, not build mode.
            vector_builder = VectorStoreBuilder(csv_path="", persist_dir=persist_dir)
            retriever = vector_builder.load_vector_store().as_retriever()

            self.recommender = AnimeRecommender(
                retriever=retriever,
                model_name=MODEL_NAME,
            )
            logger.info("Pipeline initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize pipeline: %s", str(e))
            raise CustomException("Error during pipeline initialization", e) from e

    def recommend(self, query: str) -> str:
        """Answer a single user query against the anime catalogue.

        Retrieves the most relevant catalogue entries for the query and passes
        them to the LLM, which grounds its recommendations in those entries.
        Off-topic, underspecified, and unsupported queries are handled by the
        prompt rather than raised here — they return an honest answer, not an
        exception.

        Args:
            query: The user's request in natural language, e.g. "short horror
                anime under 13 episodes".

        Returns:
            The formatted recommendation text, ready to display.

        Raises:
            CustomException: If retrieval or generation fails, with the original
                exception chained. This signals an infrastructure failure
                (index unreachable, model call errored), not an unanswerable
                question.
        """
        try:
            logger.info("Received query: %s", query)
            return self.recommender.get_recommendation(query)

        except Exception as e:
            logger.error("Recommendation failed: %s", str(e))
            raise CustomException("Error during recommendation", e) from e
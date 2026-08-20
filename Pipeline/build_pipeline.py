"""
Offline build pipeline for the anime recommendation system.

Summary
-------
This is the entrypoint that turns raw MyAnimeList exports into a queryable
vector store. It runs once (or whenever the source data changes), separate from
the serving path — nothing here is called at request time.

Two stages, in order:

1. **Load and process** — ``AnimeDataLoader`` reads the raw CSVs, joins them,
   and flattens each anime into a single ``combined_info`` text column
   (title, English title, type, score, genres, synopsis). The result is written
   back to disk as a processed CSV so the flattening is reproducible and
   inspectable without re-running the whole pipeline.
2. **Embed and persist** — ``VectorStoreBuilder`` embeds that column and saves
   the index to disk for the retrieval chain to load.

Failure policy: any exception is logged with context and re-raised as a
``CustomException``, which appends the originating file and line number. A
partial build is treated as a failed build — the caller should not proceed to
serving on a non-zero exit.

Row-order contract
------------------
The evaluation set keys ground truth on ``doc_index``, the row position in the
processed CSV. Anything in ``load_and_process`` that drops, dedupes, filters or
shuffles rows will silently shift those positions and invalidate every gold
label. If the processing step ever changes shape, either keep ``anime_id`` in
the processed output as a stable join key, or regenerate the eval set against
the new file.

Usage
-----
    python build_pipeline.py
"""

from dotenv import load_dotenv

from Src.data_loader import AnimeDataLoader
from Src.vector_store import VectorStoreBuilder
from Utils.custom_exception import CustomException
from Utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

RAW_SYNOPSIS_CSV = "Data/anime-dataset-2023.csv"
RAW_METADATA_CSV = "Data/processed_anime.csv"


def main() -> None:
    """Run the full offline build: raw CSVs in, persisted vector store out.

    Executes the two build stages in sequence, logging progress at each
    boundary so a failure can be attributed to loading versus embedding. The
    stages are ordered and dependent: the vector store is built from the
    processed CSV that the loader returns, so a loader failure short-circuits
    the run before any embedding cost is incurred.

    Side effects:
        Writes the processed CSV and the persisted vector index to disk, at the
        paths configured in ``AnimeDataLoader`` and ``VectorStoreBuilder``.
        Both are overwritten if they already exist. Embedding calls may consume
        API quota, depending on the configured embedding backend.

    Raises:
        CustomException: Wraps any failure in either stage, with the original
            exception chained so the underlying traceback is preserved.
    """
    try:
        logger.info("Starting to build pipeline...")

        loader = AnimeDataLoader(RAW_SYNOPSIS_CSV, RAW_METADATA_CSV)
        processed_csv = loader.load_and_process()
        logger.info("Data loaded and processed: %s", processed_csv)

        vector_builder = VectorStoreBuilder(processed_csv)
        vector_builder.build_and_save_vectorstore()
        logger.info("Vector store built successfully.")

        logger.info("Pipeline built successfully.")

    except Exception as e:
        logger.error("Failed to execute pipeline: %s", str(e))
        raise CustomException("Error during pipeline", e) from e


if __name__ == "__main__":
    print("WORKING")
    main()
    
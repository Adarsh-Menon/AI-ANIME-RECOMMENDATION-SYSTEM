"""Vector store construction and loading for the anime recommender.

Wraps the two halves of the retrieval layer: a one-off build step that embeds
the processed CSV into a persisted Chroma collection, and a cheap load step
that reopens that collection for querying without re-embedding.

Typical use:
    builder = VectorStoreBuilder("data/processed_anime.csv")
    builder.build_and_save_vectorstore()   # run once, or when the data changes
    store = builder.load_vector_store()    # run at app start
"""

from pathlib import Path                    # OS-independent path handling
from langchain_text_splitters import RecursiveCharacterTextSplitter  # length-based chunking
from langchain_chroma import Chroma         # persistent local vector database
from langchain_community.document_loaders.csv_loader import CSVLoader  # CSV -> one Document per row
from langchain_huggingface import HuggingFaceEmbeddings  # local sentence-transformer embeddings


class VectorStoreBuilder:
    """Builds and reopens the Chroma collection backing the recommender.

    Attributes:
        csv_path (Path): Absolute path to the processed CSV to embed.
        persist_dir (Path): Absolute path to the on-disk Chroma directory.
        embedding (HuggingFaceEmbeddings): Shared embedding function. The same
            model must be used for building and querying, or the query vectors
            won't be comparable to the stored ones.
    """

    def __init__(self, csv_path: str, persist_dir: str = "chroma_db") -> None:
        """Resolve paths, create the persist directory, and load the model.

        The embedding model is downloaded (first run, ~90MB) or loaded from the
        local HF cache here, so constructing this class is not free — build it
        once and reuse the instance.

        Args:
            csv_path (str): Path to the processed CSV produced by
                AnimeDataLoader. Resolved to an absolute path.
            persist_dir (str): Directory Chroma writes its database into.
                Created if absent. Defaults to "chroma_db".
        """
        # Path: resolved so behaviour doesn't depend on the working directory.
        self.csv_path: Path = Path(csv_path).resolve()
        self.persist_dir: Path = Path(persist_dir).resolve()

        # parents=True creates intermediate dirs; exist_ok=True makes reruns a no-op.
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # HuggingFaceEmbeddings: all-MiniLM-L6-v2 — 384-dim output, runs on CPU.
        self.embedding: HuggingFaceEmbeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

    def build_and_save_vectorstore(self) -> Chroma:
        """Embed the CSV into a persisted Chroma collection.

        Expensive — every chunk is passed through the embedding model. Run this
        when the source data changes, not on every app start; use
        `load_vector_store` for that.

        Returns:
            Chroma: The populated store, so a build can flow straight into a
            query without reloading from disk.
        """
        # CSVLoader: yields one Document per row, page_content rendered as
        # "column_name: value" lines.
        loader = CSVLoader(
            file_path=str(self.csv_path),
            encoding="utf-8"
        )

        # list[Document]: one entry per CSV row.
        documents = loader.load()

        # RecursiveCharacterTextSplitter: splits on paragraph, then sentence,
        # then word boundaries, falling back to raw characters. Documents under
        # chunk_size pass through untouched.
        #   chunk_size    -> max characters per chunk
        #   chunk_overlap -> characters repeated between neighbours (0 = none)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        # list[Document]: chunks ready for embedding; >= len(documents).
        split_docs = splitter.split_documents(documents)

        # Chroma.from_documents: embeds every chunk and writes to persist_dir.
        # langchain-chroma persists automatically — no .persist() call needed.
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embedding,
            persist_directory=str(self.persist_dir)
        )
        return vectorstore  # optional but useful

    def load_vector_store(self) -> Chroma:
        """Reopen the already-persisted collection for querying.

        Cheap — reads the existing database rather than re-embedding. The
        embedding function is still required, since query strings have to be
        embedded with the same model used at build time.

        Returns:
            Chroma: The store, ready for `.similarity_search()` or
            `.as_retriever()`.
        """
        return Chroma(
            persist_directory=str(self.persist_dir),
            embedding_function=self.embedding
        )
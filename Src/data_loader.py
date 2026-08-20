"""Loading and preprocessing for the anime dataset.

Reads the raw ANIME CSV, validates that the columns the pipeline
depends on are present, and flattens the descriptive fields into a single
natural-language string suitable for embedding into a vector store. The
identifying fields are carried through unchanged so they can be attached as
retrieval metadata later.

Typical use:
    loader = AnimeDataLoader("data/anime.csv", "data/processed.csv")
    path = loader.load_and_process()
"""

import pandas as pd    # DataFrame I/O and column-wise string operations

# Column names as they appear in the source CSV, pulled out as constants so a
# rename in the dataset is a one-line fix rather than a search across the file.

COL_NAME: str = "Name"
COL_ENGLISH_NAME: str = "English name"
COL_SCORE: str = "Score"
COL_GENRES: str = "Genres"
COL_SYNOPSIS: str = "Synopsis"
COL_TYPE: str = "Type"

# list[str]: every column read from the CSV. Used for validation, for scoping
# the null-drop, and to select what gets written out.
REQUIRED_COLS: list[str] = [
    COL_NAME,
    COL_ENGLISH_NAME,
    COL_SCORE,
    COL_GENRES,
    COL_SYNOPSIS,
    COL_TYPE,
]


class AnimeDataLoader:
    """Turns a raw anime CSV into an embeddable text column plus metadata.

    Attributes:
        original_csv (str): Path to the raw input CSV.
        processed_csv (str): Path the processed CSV is written to.
    """

    def __init__(self, original_csv: str, processed_csv: str) -> None:
        """Store the input and output paths. No I/O happens here.

        Args:
            original_csv (str): Path to the raw dataset on disk.
            processed_csv (str): Destination path for the processed output.
                Its parent directory must already exist.
        """
        self.original_csv: str = original_csv    # read from in load_and_process
        self.processed_csv: str = processed_csv  # written to in load_and_process

    def load_and_process(self) -> str:
        """Read, validate, and flatten the dataset, then write it to disk.

        Returns:
            str: The path written to (`self.processed_csv`), so callers can
            chain straight into a loader/splitter step.

        Raises:
            ValueError: If any required column is absent from the CSV. The
                message names both the missing and the available columns.
            FileNotFoundError: If `original_csv` does not exist.
        """
        # DataFrame: raw rows. on_bad_lines='skip' silently discards malformed
        # records rather than aborting — synopsis fields often contain stray
        # quotes and commas that break the parser.
        df: pd.DataFrame = pd.read_csv(
            self.original_csv,
            encoding="utf-8",
            on_bad_lines="skip",
        )

        # Validate BEFORE any column access, so a schema mismatch surfaces as a
        # readable error instead of a bare KeyError deeper in the method.
        missing: list[str] = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing columns in {self.original_csv}: {missing}. "
                f"Found: {list(df.columns)}"
            )

        # Drop nulls only within the columns actually used. A blanket .dropna()
        # would discard rows for blanks in unrelated fields (Ranked, Aired,
        # Studios...), which are missing often enough to gut the dataset.
        df = df.dropna(subset=REQUIRED_COLS)

        # Cast to str before concatenating: Score reads as float64, and adding a
        # numeric Series to a string Series raises TypeError.
        # Series[str]: one flattened description per anime, for embedding.
        df["combined_info"] = (
            "Title : " + df[COL_NAME].astype(str)
            + " | English Title : " + df[COL_ENGLISH_NAME].astype(str)
            + " | Type : " + df[COL_TYPE].astype(str)
            + " | Score : " + df[COL_SCORE].astype(str)
            + " | Genres : " + df[COL_GENRES].astype(str)
            + " | Overview : " + df[COL_SYNOPSIS].astype(str)
        )

        # Keep the source columns alongside the derived text so the retriever can
        # surface a clean title or filter by type/score without re-parsing
        # combined_info. index=False keeps row numbers out of the file.
        df[["combined_info"]].to_csv(
            self.processed_csv,
            index=False,
            encoding="utf-8",
        )

        return self.processed_csv
    
if __name__ == "__main__":
    # Quick manual check — run this file directly to verify the paths and
    # schema before wiring the loader into the pipeline.
    loader = AnimeDataLoader(
        original_csv="Data/anime-dataset-2023.csv",
        processed_csv="Data/processed_anime.csv",
    )
    out_path: str = loader.load_and_process()
    print(f"Wrote: {out_path}")

    # Show what actually landed in the file.
    check: pd.DataFrame = pd.read_csv(out_path, encoding="utf-8")
    print(f"Rows: {len(check)}")
    print(f"Columns: {list(check.columns)}")
    print("\nFirst combined_info entry:\n")
    print(check["combined_info"].iloc[0])   
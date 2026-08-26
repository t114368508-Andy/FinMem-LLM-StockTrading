import os
import httpx
import numpy as np
from typing import List, Union
from langchain_community.embeddings import OpenAIEmbeddings
from .http_retry import request_with_retry, gemini_pacer


class OpenAILongerThanContextEmb:
    """
    Embedding function with openai as embedding backend.
    If the input is larger than the context size, the input is split into chunks of size `chunk_size` and embedded separately.
    The final embedding is the average of the embeddings of the chunks.
    Details see: https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
    """

    def __init__(
        self,
        openai_api_key: Union[str, None] = None,
        embedding_model: str = "text-embedding-ada-002",
        chunk_size: int = 5000,
        verbose: bool = False,
    ) -> None:
        """
        Initializes the Embedding object.

        Args:
            openai_api_key (str): The API key for OpenAI.
            embedding_model (str, optional): The model to use for embedding. Defaults to "text-embedding-ada-002".
            chunk_size (int, optional): The maximum number of token to send to openai embedding model at one time. Defaults to 5000.
            verbose (bool, optional): Whether to show progress bar during embedding. Defaults to False.

        Returns:
            None
        """
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.emb_model = OpenAIEmbeddings(
            model=embedding_model,
            api_key=openai_api_key or os.environ.get("OPENAI_API_KEY"),
            chunk_size=chunk_size,
            show_progress_bar=verbose,
        )

    def _emb(self, text: Union[List[str], str]) -> List[List[float]]:
        """
        Asynchronously performs embedding on a list of text.

        This method calls the `aembed_documents` method of the `emb_model` object to embed the input text.

        Args:
            self: The instance of the class.
            text (List[str]): A list of text to be embedded.

        Returns:
            List[List[float]]: The embeddings of the input text as a list of lists of floats.

        """
        if isinstance(text, str):
            text = [text]
        return self.emb_model.embed_documents(texts=text, chunk_size=None)

    def __call__(self, text: Union[List[str], str]) -> np.ndarray:
        """
        Performs embedding on a list of text.

        This method calls the `_emb` method to asynchronously embed the input text using the `emb_model` object.

        Args:
            self: The instance of the class.
            text (List[str]): A list of text to be embedded.

        Returns:
            np.array: The embedding of the input text as a NumPy array.

        """
        return np.array(self._emb(text)).astype("float32")

    def get_embedding_dimension(self):
        """
        Returns the dimension of the embedding.

        This method checks the value of `self.emb_model.model` and returns the corresponding embedding dimension. If the model is not implemented, a `NotImplementedError` is raised.

        Args:
            self: The instance of the class.

        Returns:
            int: The dimension of the embedding.

        Raises:
            NotImplementedError: Raised when the embedding dimension for the specified model is not implemented.

        """
        match self.emb_model.model:
            case "text-embedding-ada-002":
                return 1536
            case _:
                raise NotImplementedError(
                    f"Embedding dimension for model {self.emb_model.model} not implemented"
                )


class GeminiLongerThanContextEmb:
    """
    Embedding function with Gemini (Google AI Studio) as embedding backend.
    Calls the REST embedding endpoint directly with httpx, the same low-level
    approach `puppy/chat.py` uses for the chat model, instead of adding the
    google-genai SDK as a new dependency.

    If the input is larger than the model's context window, the input is
    split into chunks of `chunk_char_size` characters and embedded
    separately. The final embedding is the average of the embeddings of the
    chunks, mirroring `OpenAILongerThanContextEmb`'s behavior.
    """

    _VALID_DIMENSIONS = {768, 1536, 3072}

    def __init__(
        self,
        gemini_api_key: Union[str, None] = None,
        embedding_model: str = "gemini-embedding-001",
        output_dimensionality: int = 768,
        chunk_char_size: int = 6000,
        max_retries: int = 8,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            gemini_api_key (str): The API key for Gemini (Google AI Studio).
                Defaults to the `GEMINI_API_KEY` environment variable.
            embedding_model (str, optional): The Gemini embedding model to
                use. Defaults to "gemini-embedding-001".
            output_dimensionality (int, optional): The output embedding
                dimension. Must be one of 768, 1536 or 3072 (Google's
                recommended truncation points via Matryoshka Representation
                Learning). Defaults to 768.
            chunk_char_size (int, optional): The maximum number of
                characters to send to the Gemini embedding model at one
                time. Defaults to 6000.
            max_retries (int, optional): Number of retries on rate limits,
                transient server errors, and network failures, with
                exponential backoff. Defaults to 8.
            verbose (bool, optional): Whether to print retry/progress
                messages. Defaults to False.
        """
        if output_dimensionality not in self._VALID_DIMENSIONS:
            raise ValueError(
                f"output_dimensionality must be one of {self._VALID_DIMENSIONS}, "
                f"got {output_dimensionality}"
            )
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY", "-")
        self.embedding_model = embedding_model
        self.output_dimensionality = output_dimensionality
        self.chunk_char_size = chunk_char_size
        self.max_retries = max_retries
        self.verbose = verbose
        self.end_point = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{embedding_model}:batchEmbedContents"
        )

    def _chunk_text(self, text: str) -> List[str]:
        if len(text) <= self.chunk_char_size:
            return [text]
        return [
            text[i : i + self.chunk_char_size]
            for i in range(0, len(text), self.chunk_char_size)
        ]

    # Gemini's batchEmbedContents endpoint rejects more than 100 requests in
    # a single call ("at most 100 requests can be in one batch") -- observed
    # directly on a news-heavy day (117 articles on a 10-K filing date), so
    # a single day's memory ingestion has to be split into sub-batches.
    _MAX_BATCH_SIZE = 100

    def _embed_one_batch_call(self, texts: List[str]) -> List[List[float]]:
        payload = {
            "requests": [
                {
                    "model": f"models/{self.embedding_model}",
                    "content": {"parts": [{"text": t}]},
                    "outputDimensionality": self.output_dimensionality,
                }
                for t in texts
            ]
        }
        headers = {
            "x-goog-api-key": self.gemini_api_key,
            "Content-Type": "application/json",
        }

        gemini_pacer.wait()
        response = request_with_retry(
            "POST",
            self.end_point,
            headers=headers,
            json=payload,
            timeout=60.0,
            max_retries=self.max_retries,
        )
        response.raise_for_status()
        values = [item["values"] for item in response.json()["embeddings"]]
        for v in values:
            if len(v) != self.output_dimensionality:
                raise ValueError(
                    f"Gemini returned a {len(v)}-dim embedding but "
                    f"output_dimensionality={self.output_dimensionality} was requested "
                    "(the API has been observed to silently ignore this parameter "
                    "depending on where it's placed in the request). Fix the request "
                    "before this reaches faiss.IndexFlatIP, which is sized to "
                    "output_dimensionality and will fail confusingly otherwise."
                )
        return values

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        values: List[List[float]] = []
        for i in range(0, len(texts), self._MAX_BATCH_SIZE):
            values.extend(self._embed_one_batch_call(texts[i : i + self._MAX_BATCH_SIZE]))
        return values

    def _emb(self, text: Union[List[str], str]) -> List[List[float]]:
        """
        Performs embedding on a list of text via Gemini's batchEmbedContents
        endpoint, one HTTP call per up-to-100 chunks (the endpoint's own
        per-request cap), so a single `__call__` may issue several calls on
        a heavy day.

        Args:
            self: The instance of the class.
            text (List[str]): A list of text to be embedded.

        Returns:
            List[List[float]]: The embeddings of the input text as a list of lists of floats.
        """
        if isinstance(text, str):
            text = [text]

        # flatten (doc index -> chunks) so every chunk across every doc is
        # sent in a single batch request
        flat_chunks: List[str] = []
        chunk_owner: List[int] = []
        for doc_idx, doc in enumerate(text):
            chunks = self._chunk_text(doc)
            flat_chunks.extend(chunks)
            chunk_owner.extend([doc_idx] * len(chunks))

        chunk_embeddings = self._embed_batch(flat_chunks)

        result: List[List[float]] = []
        for doc_idx in range(len(text)):
            vectors = [
                chunk_embeddings[i]
                for i, owner in enumerate(chunk_owner)
                if owner == doc_idx
            ]
            result.append(np.mean(vectors, axis=0).tolist())
        return result

    def __call__(self, text: Union[List[str], str]) -> np.ndarray:
        """
        Performs embedding on a list of text.

        Args:
            self: The instance of the class.
            text (List[str]): A list of text to be embedded.

        Returns:
            np.array: The embedding of the input text as a NumPy array.
        """
        return np.array(self._emb(text)).astype("float32")

    def get_embedding_dimension(self):
        """
        Returns the dimension of the embedding, matching the
        `output_dimensionality` this instance was configured with, so
        `puppy/memorydb.py` can size the FAISS index correctly.
        """
        return self.output_dimensionality

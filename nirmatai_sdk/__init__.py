"""Module."""

__doc__ = """
NirmatAI SDK - a SDK for NirmatAI RAG system for requirements auditing
----------------------------------------------------------------------

**NirmatAI SDK**, as the name suggest, is a Python package providing a SDK to the
NirmatAI Retrieval-Augmented Generation (RAG) system for requirements auditing.

The RAG system is built on top of PrivateGPT and uses extensively Ollama for LLM serving
and Qdrant for embeddings storage.

Main Features
^^^^^^^^^^^^^
Here are just a few of the things that NirmatAI does well:

    - **Ingestion**: Ingest requirements and documents in various formats.
    - **Processing**: Process requirements and documents to generate results.
    - **Scoring**: Score the results using a variety of metrics.
    - **Reporting**: Generate reports and save results in various formats.
"""

from nirmatai_sdk.core import NirmatAI
from nirmatai_sdk.telemetry import Scorer

__all__ = ["NirmatAI", "Scorer"]

"""NirmatAI SDK Package."""

doc = """
NirmatAI SDK - A Comprehensive SDK for NirmatAI RAG System
----------------------------------------------------------

NirmatAI SDK is a Python package that provides a robust SDK for integrating with the
NirmatAI Retrieval-Augmented Generation (RAG) system for effective requirements auditing.

The RAG system leverages PrivateGPT for its core processing, Ollama for advanced language 
model serving, and Qdrant for efficient embeddings storage and retrieval.

Key Features
^^^^^^^^^^^^
NirmatAI SDK offers a variety of functionalities to enhance your requirements auditing process:

    - Ingestion: Seamlessly ingest requirements and documents in multiple formats.
    - Processing: Process and analyze documents to extract meaningful insights.
    - Scoring: Utilize various metrics to score and evaluate results.
    - Reporting: Generate comprehensive reports and export results in diverse formats.
"""

from .core import NirmatAI  # Main class for interacting with NirmatAI RAG system
from .telemetry import Scorer  # Class for scoring and evaluating results

all = ['NirmatAI', 'Scorer']

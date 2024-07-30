# NirmatAI Software Development Kit

**NirmatAI SDK** is a Python package providing an SDK to the NirmatAI Retrieval-Augmented Generation (RAG) system for requirements auditing. The RAG system is built on top of PrivateGPT and extensively uses Ollama for LLM serving and Qdrant for embeddings storage.

## Main Features

Here are just a few of the things that NirmatAI does well:

- **Ingestion**: Ingest requirements and documents in various formats.
- **Processing**: Process requirements and documents to generate results.
- **Scoring**: Score the results using a variety of metrics.
- **Reporting**: Generate reports and save results in various formats.

## Installation

You can install the NirmatAI SDK using pip:

```bash
pip install nirmatai_sdk
```

## Project Structure

```plaintext
NirmatAI_SDK/
├── .github/
│   ├── dependabot.yml
│   ├── workflows/
│   │   ├── actions/
│   │   │   ├── install_dependencies/
│   │   │   │   ├── action.yml
│   │   ├── handling-issues.yml
│   │   ├── static_tests.yml
├── docs/
│   ├── source/
│   │   ├── conf.py
│   │   ├── dev.rst
│   │   ├── index.rst
├── nirmatai_sdk/
│   ├── __init__.py
│   ├── core.py
│   ├── telemetry.py
│   ├── tests/
│   │   ├── test_core_integration.py
│   │   ├── test_core_unit.py
│   │   ├── test_telemetry_unit.py
├── .pre-commit-config.yaml
├── Dockerfile.client
├── Dockerfile.dev
├── check_functions.yml
├── LICENSE
├── Makefile
├── pyproject.toml
├── README.md
```

## Usage

Here's a basic example of how to use the NirmatAI SDK:

```python
from nirmatai_sdk.core import NirmatAI
from nirmatai_sdk.telemetry import Scorer

# Initialize the NirmatAI instance
nirmat_ai = NirmatAI()

# Example method usage
nirmat_ai.some_method()

# Initialize the Scorer
scorer = Scorer()

# Example scoring
score = scorer.calculate_score()
```

## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or concerns, please contact CertX at [ilker.gul@certx.com](mailto:ilker.gul@certx.com).

## Acknowledgements

- **PrivateGPT**: The underlying technology for the RAG system.
- **Ollama**: For LLM serving.
- **Qdrant**: For embeddings storage.

### Summary:
- **Introduction**: Describes the SDK and its main features.
- **Installation**: Provides instructions on how to install the SDK.
- **Project Structure**: Lists and explains the directory structure of the project.
- **Usage**: Gives a basic example of how to use the SDK.
- **License**: States the license under which the project is distributed.
- **Contact**: Provides contact information for further inquiries.
- **Acknowledgements**: Credits the technologies used in the SDK.

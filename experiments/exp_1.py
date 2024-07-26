"""Hello World example of an experiment.

The script is a demonstration of how to use the NirmataAI class to process
requirements and documents.
"""

import argparse
import os

import mlflow

from nirmatai_sdk.core import NirmatAI
from nirmatai_sdk.telemetry import Scorer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run experiment with internal parameters."
    )
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text")
    args = parser.parse_args()

    mlflow.set_experiment("Large dataset experiment #33.")
    with mlflow.start_run():
        mlflow.log_param("PGPT_OLLAMA_LLM_MODEL", os.environ["PGPT_OLLAMA_LLM_MODEL"])
        mlflow.log_param(
            "PGPT_OLLAMA_EMBEDDING_MODEL", os.environ["PGPT_OLLAMA_EMBEDDING_MODEL"]
        )
        mlflow.log_param(
            "PGPT_RAG_SIMILARITY_TOP_K", os.environ["PGPT_RAG_SIMILARITY_TOP_K"]
        )

        POD_NAME = os.environ["POD_NAME"]
        mlflow.log_param("POD_NAME", POD_NAME)

        # Log internal parameters
        mlflow.log_param("prompt", args.prompt)

        # Initialize the NirmataDemo instance
        demo = NirmatAI(
            system_prompt=args.prompt,
            base_url=f"http://{POD_NAME}_privategpt:8080/",
            timeout=60 * 60,
            verbose=3,
            prompt="""
            The requirement to be evaluated is: {req_item}
            The means of compliance is: {moc_item}
            """,
        )

        # Ingest files
        demo.ingest(directory="data/docs/")

        # Load requirements and get the results
        demo.load_requirements(reqs_file="data/req/requirements_17021.xlsx")

        # Process the requirements
        demo.process_requirements()

        # Score the results
        score = Scorer(demo.y_true, demo.y_pred)
        cm, cm_path, M_MAE, k = score.run_scores()
        # Log the scores
        mlflow.log_metric("Macro-averaged MAE", M_MAE)
        mlflow.log_metric("Kappa", k)

        # Delete all ingested documents
        demo.delete_all_documents()


if __name__ == "__main__":
    main()

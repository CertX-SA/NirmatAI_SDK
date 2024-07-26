# type: ignore
"""Integration tests for the core module of NirmatAI."""


import pandas as pd
from pgpt_python.types import HealthResponse

from nirmatai_sdk.core import NirmatAI


def test_health_check_ok():
    """Test the health_check method of the NirmatAI class."""
    # Create a NirmatAI instance
    nirmatai = NirmatAI(verbose=3)

    # Check the health
    health = nirmatai.health_check()

    assert health == HealthResponse(status="ok")


def test_health_check_ko():
    """Test the health_check method of the NirmatAI class."""
    # Create a NirmatAI instance
    nirmatai = NirmatAI(verbose=3, base_url="http://localhost:9000")

    # Check the health
    health = nirmatai.health_check()

    assert health == HealthResponse(status="ko")


def test_ingest(tmp_path):
    """Test the ingest method of the NirmatAI class."""
    # Create a NirmatAI instance
    nirmatai = NirmatAI(verbose=3)

    # Create a directory
    directory = tmp_path / "test_dir"
    directory.mkdir()

    # Create a txt file containing "Hello, World!"
    file = directory / "test_file.txt"
    file.write_text("Hello, World!")

    # Ingest the file
    nirmatai.ingest(directory)

    # Get the files
    files = nirmatai.files

    # Delete documents
    nirmatai.delete_all_documents()

    assert len(files.keys()) == 1


def test__get_completion():
    """Test the _get_completion method of the NirmatAI class."""
    # Create a NirmatAI instance
    nirmatai = NirmatAI(verbose=3)

    message, sources = nirmatai._NirmatAI__get_completion(
        "Hello World!", "Hello World!"
    )

    assert isinstance(message, str)


def test_process_requirements(tmp_path):
    """Test the process_requirements method of the NirmatAI class."""
    # Create a NirmatAI instance
    nirmatai = NirmatAI(verbose=3)

    # Create a directory
    directory = tmp_path / "test_dir"
    directory.mkdir()

    # Create a txt file containing "Hello, World!"
    file = directory / "doc1.txt"
    file.write_text("Hello, World!")

    # Ingest the file
    nirmatai.ingest(directory)

    # Create a xlsx file containing Requirements, Potential Means of Compliance
    file = directory / "req.xlsx"
    df = pd.DataFrame(
        {
            "Requirement": ["Hello, World!"],
            "Potential Means of Compliance": ["Hello, World!"],
        }
    )
    df.to_excel(file, index=False)
    nirmatai.load_requirements(file)

    # Process the requirements
    result = nirmatai.process_requirements()

    # Delete documents
    nirmatai.delete_all_documents()

    # Assert that the result is a dataframe with one row
    assert result.shape[0] == 1

    # Assert that it has 3 coulumns
    assert result.shape[1] == 3

    # Assert that nirmatai.y_pred has one element
    assert len(nirmatai.y_pred) == 1


def test_process_requirements_with_moc_na(tmp_path):
    """Test the process_requirements method of the NirmatAI class."""
    # Create a NirmatAI instance
    nirmatai = NirmatAI(verbose=3)

    # Create a directory
    directory = tmp_path / "test_dir"
    directory.mkdir()

    # Create a txt file containing "Hello, World!"
    file = directory / "doc1.txt"
    file.write_text("Hello, World!")

    # Ingest the file
    nirmatai.ingest(directory)

    # Create a xlsx file containing Requirements, Potential Means of Compliance
    file = directory / "req.xlsx"
    df = pd.DataFrame(
        {
            "Requirement": ["Hello, World!"],
            "Potential Means of Compliance": pd.NA,
        }
    )
    df.to_excel(file, index=False)
    nirmatai.load_requirements(file)

    # Process the requirements
    result = nirmatai.process_requirements()

    # Delete documents
    nirmatai.delete_all_documents()

    # Assert that the result is a dataframe with one row
    assert result.shape[0] == 1

    # Assert that it has 3 coulumns
    assert result.shape[1] == 3

    # Assert that nirmatai.y_pred has one element
    assert len(nirmatai.y_pred) == 1

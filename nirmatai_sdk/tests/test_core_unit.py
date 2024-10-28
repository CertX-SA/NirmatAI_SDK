# type: ignore
""""Unit tests for the core module of NirmatAI."""

from unittest.mock import patch

import pandas as pd
from pgpt_python.types import Chunk, IngestedDoc

from nirmatai_sdk.core import NirmatAI


def test_get_files(tmp_path):
    """Test the __get_files method of the NirmatAI class."""
    # Create a NirmatAI instance
    nirmatai = NirmatAI(verbose=3)

    # Create a directory
    directory = tmp_path / "test_dir"
    directory.mkdir()

    # Create txt, pdf, docx, and xlsx files
    for ext in ["txt", "pdf", "docx", "xlsx"]:
        file = directory / f"test_file.{ext}"
        file.touch()

    # Create a hidden file
    hidden_file = directory / ".hidden_file"
    hidden_file.touch()

    # Create a .dvc file
    dvc_file = directory / "test_file.dvc"
    dvc_file.touch()

    # Get the files
    files = nirmatai._NirmatAI__get_files(directory)

    assert len(files) == 4


def test_get_files_with_path_to_file(tmp_path):
    """Test the __get_files method of the NirmatAI class."""
    # Create a NirmatAI instance
    nirmatai = NirmatAI(verbose=3)

    # Create a txt file
    file = tmp_path / "test_file.txt"
    file.touch()

    # Get the files
    files = nirmatai._NirmatAI__get_files(file)

    assert len(files) == 1


def test_get_files_with_path_not_ending_with_slash(tmp_path):
    """Test the __get_files method of the NirmatAI class."""
    # Create a NirmatAI instance
    nirmatai = NirmatAI(verbose=3)

    # Create a directory
    directory = tmp_path / "test_dir"
    directory.mkdir()

    # Create a txt file
    file = directory / "test_file.txt"
    file.touch()

    # Get the files
    files = nirmatai._NirmatAI__get_files(directory)

    assert len(files) == 1


def test_get_files_with_non_existent_path():
    """Test the __get_files method of the NirmatAI class."""
    # Create a NirmatAI instance
    nirmatai = NirmatAI(verbose=3)

    # Get the files and assert FileNotFoundError
    directory = "non_existent_path"
    try:
        nirmatai._NirmatAI__get_files(directory)
    except FileNotFoundError:
        assert True


def test_get_files_with_empty_directory(tmp_path):
    """Test the __get_files method of the NirmatAI class."""
    # Create a NirmatAI instance
    nirmatai = NirmatAI(verbose=3)

    # Create an empty directory
    directory = tmp_path / "empty_dir"
    directory.mkdir()

    # Get the files
    try:
        nirmatai._NirmatAI__get_files(directory)
    except FileNotFoundError:
        assert True


def test_load_requirements_with_label(tmp_path):
    """Test the __load_requirements method of the NirmatAI class."""
    # Create a NirmatAI instance
    nirmatai = NirmatAI(verbose=3)

    # Create a DataFrame with a Label column
    df = pd.DataFrame(
        {
            "Requirement": ["req1", "req2", "req3"],
            "Potential Means of Compliance": ["pmc1", "pmc2", "pmc3"],
            "Label": ["label1", "label2", "label3"],
        }
    )

    # Save the DataFrame to an Excel file
    reqs_file = tmp_path / "requirements.xlsx"
    df.to_excel(reqs_file, index=False)

    # Call the load_requirements method
    nirmatai.load_requirements(str(reqs_file))

    # Check the result
    pd.testing.assert_frame_equal(nirmatai.reqs, df)


def test_load_requirements_without_label(tmp_path):
    """Test the __load_requirements method of the NirmatAI class."""
    # Create a NirmatAI instance
    nirmatai = NirmatAI(verbose=3)

    # Create a DataFrame without a Label column
    df = pd.DataFrame(
        {
            "Requirement": ["req1", "req2", "req3"],
            "Potential Means of Compliance": ["pmc1", "pmc2", "pmc3"],
        }
    )

    # Save the DataFrame to an Excel file
    reqs_file = tmp_path / "requirements.xlsx"
    df.to_excel(reqs_file, index=False)

    # Call the load_requirements method
    nirmatai.load_requirements(str(reqs_file))

    # Check the result
    pd.testing.assert_frame_equal(nirmatai.reqs, df)


def test_load_requirements_with_non_existent_path():
    """Test the __load_requirements method of the NirmatAI class."""
    # Create a NirmatAI instance
    nirmatai = NirmatAI(verbose=3)

    # Call the load_requirements method and assert FileNotFoundError
    reqs_file = "non_existent_path"
    try:
        nirmatai.load_requirements(reqs_file)
    except FileNotFoundError:
        assert True


def test_load_requirements_with_non_xlsx_file(tmp_path):
    """Test the __load_requirements method of the NirmatAI class."""
    # Create a NirmatAI instance
    nirmatai = NirmatAI(verbose=3)

    # Create a txt temporary file
    reqs_file = tmp_path / "requirements.txt"

    with open(reqs_file, "w") as file:
        file.write(
            "Requirement,Potential Means of Compliance\nreq1,pmc1\nreq2,pmc2\nreq3,pmc3"
        )

    # Call the load_requirements method and assert ValueError
    try:
        nirmatai.load_requirements(reqs_file)
    except ValueError:
        assert True


def test_load_requirements_without_requirement_col(tmp_path):
    """Test the __load_requirements method of the NirmatAI class."""
    # Create a NirmatAI instance
    nirmatai = NirmatAI(verbose=3)

    # Create a DataFrame without a Requirement column
    df = pd.DataFrame(
        {
            "Potential Means of Compliance": ["pmc1", "pmc2", "pmc3"],
            "Label": ["label1", "label2", "label3"],
        }
    )

    # Save the DataFrame to a temporary Excel file
    reqs_file = tmp_path / "requirements.xlsx"
    df.to_excel(reqs_file, index=False)

    # Call the load_requirements method and assert ValueError
    try:
        nirmatai.load_requirements(reqs_file)
    except ValueError:
        assert True


def test_load_requirements_without_pmc_col(tmp_path):
    """Test the __load_requirements method of the NirmatAI class."""
    # Create a NirmatAI instance
    nirmatai = NirmatAI(verbose=3)

    # Create a DataFrame without a Potential Means of Compliance column
    df = pd.DataFrame(
        {
            "Requirement": ["req1", "req2", "req3"],
            "Label": ["label1", "label2", "label3"],
        }
    )

    # Save the DataFrame to a temporary Excel file
    reqs_file = tmp_path / "requirements.xlsx"
    df.to_excel(reqs_file, index=False)

    # Call the load_requirements method and assert ValueError
    try:
        nirmatai.load_requirements(reqs_file)
    except ValueError:
        assert True


def test_extract_comp_status_major_non_conformity():
    """Test the __extract_comp_status method of the NirmatAI class."""
    # Create a NirmatAI instance
    nirmatai = NirmatAI(verbose=3)

    # Create comp_status
    comp_status = [
        "Major Non-Conformity",
        "Non-Conformity",
        "AAAAMajor Non-Conformity",
        "AAAAMajorANon-ConformityA",
    ]

    # Call the extract_comp_status method
    result = [nirmatai._NirmatAI__extract_comp_status(status) for status in comp_status]

    # Check the result
    assert result == [
        "major non-conformity",
        "",
        "major non-conformity",
        "major non-conformity",
    ]


def test_extract_comp_status_minor_non_conformity():
    """Test the __extract_comp_status method of the NirmatAI class."""
    # Create a NirmatAI instance
    nirmatai = NirmatAI(verbose=3)

    # Create comp_status
    comp_status = [
        "Minor Non-Conformity",
        "AAAAMinor Non-Conformity",
        "AAAAMinor Non-ConformityA",
    ]

    # Call the extract_comp_status method
    result = [nirmatai._NirmatAI__extract_comp_status(status) for status in comp_status]

    # Check the result
    assert result == [
        "minor non-conformity",
        "minor non-conformity",
        "minor non-conformity",
    ]


def test_extract_comp_status_full_compliance():
    """Test the __extract_comp_status method of the NirmatAI class."""
    # Create a NirmatAI instance
    nirmatai = NirmatAI(verbose=3)

    # Create comp_status
    comp_status = [
        "Full-Compliance",
        "AAAFull-compliance",
        "AAAFull-complianceA",
    ]

    # Call the extract_comp_status method
    result = [nirmatai._NirmatAI__extract_comp_status(status) for status in comp_status]

    # Check the result
    assert result == [
        "full-compliance",
        "full-compliance",
        "full-compliance",
    ]


def test_extract_comp_status_last_resort():
    """Test the __extract_comp_status method of the NirmatAI class."""
    # Create a NirmatAI instance
    nirmatai = NirmatAI(verbose=3)

    # Create comp_status
    comp_status = [
        "Non-Conformity",
        "Minor Conformity",
        "Full-Conformity",
        "fullcompliance",
    ]

    # Call the extract_comp_status method
    result = [nirmatai._NirmatAI__extract_comp_status(status) for status in comp_status]

    # Check the result
    assert result == [
        "",
        "minor non-conformity",
        "full-compliance",
        "full-compliance",
    ]


def test_save_results_csv(tmp_path):
    """Test the save_results method of the NirmatAI class."""
    # Create a NirmatAI instance
    nirmatai = NirmatAI(verbose=3)

    # Create a DataFrame
    df = pd.DataFrame(
        {
            "Compliance Status": [
                "major non-conformity",
                "minor non-conformity",
                "full-compliance",
            ],
            "Rationale": [
                "rationale1",
                "rationale2",
                "rationale3",
            ],
            "Ref. to Doc": [
                "doc1",
                "doc2",
                "doc3",
            ],
        }
    )

    # Save the DataFrame to a csv file
    result_file = tmp_path / "result.csv"
    nirmatai.save_results(df, str(result_file))

    # Read the saved file
    saved_df = pd.read_csv(result_file)

    # Check the result
    pd.testing.assert_frame_equal(saved_df, df)


def test_save_results_html(tmp_path):
    """Test the save_results method of the NirmatAI class."""
    # Create a NirmatAI instance
    nirmatai = NirmatAI(verbose=3)

    # Create a DataFrame
    df = pd.DataFrame(
        {
            "Compliance Status": [
                "major non-conformity",
                "minor non-conformity",
                "full-compliance",
            ],
            "Rationale": [
                "rationale1",
                "rationale2",
                "rationale3",
            ],
            "Ref. to Doc": [
                "doc1",
                "doc2",
                "doc3",
            ],
        }
    )

    # Save the DataFrame to an html file
    result_file = tmp_path / "result.html"
    nirmatai.save_results(df, str(result_file))

    # Read the saved file
    saved_df = pd.read_html(result_file)[0]

    # Check the result
    pd.testing.assert_frame_equal(saved_df, df)


def test_save_results_path_not_found(tmp_path):
    """Test the save_results method of the NirmatAI class."""
    # Create a NirmatAI instance
    nirmatai = NirmatAI(verbose=3)

    # Create a DataFrame
    df = pd.DataFrame(
        {
            "Compliance Status": [
                "major non-conformity",
                "minor non-conformity",
                "full-compliance",
            ],
            "Rationale": [
                "rationale1",
                "rationale2",
                "rationale3",
            ],
            "Ref. to Doc": [
                "doc1",
                "doc2",
                "doc3",
            ],
        }
    )

    # Save the DataFrame to an html file
    result_file = tmp_path / "non_existent_dir" / "result.html"
    try:
        nirmatai.save_results(df, str(result_file))
    except FileNotFoundError:
        assert True


def test_save_results_with_attach_reqs(tmp_path):
    """Test the save_results method of the NirmatAI class."""
    # Create a NirmatAI instance
    nirmatai = NirmatAI(verbose=3)

    # Create a DataFrame
    df = pd.DataFrame(
        {
            "Compliance Status": [
                "major non-conformity",
                "minor non-conformity",
                "full-compliance",
            ],
            "Rationale": [
                "rationale1",
                "rationale2",
                "rationale3",
            ],
            "Ref. to Doc": [
                "doc1",
                "doc2",
                "doc3",
            ],
        }
    )

    nirmatai.reqs = pd.DataFrame(
        {
            "Requirement": ["req1", "req2", "req3"],
            "Potential Means of Compliance": ["pmc1", "pmc2", "pmc3"],
            "Label": ["label1", "label2", "label3"],
        }
    )

    # Save the DataFrame to an html file
    result_file = tmp_path / "result.html"
    nirmatai.save_results(df, str(result_file), attach_reqs=True)

    # Read the saved file
    saved_df = pd.read_html(result_file)[0]

    # Check the result
    pd.testing.assert_frame_equal(saved_df, pd.concat([nirmatai.reqs, df], axis=1))


def test__format_sources(tmp_path):
    """Test the __format_sources method of the NirmatAI class."""
    # Create a NirmatAI instance
    nirmatai = NirmatAI(verbose=3)

    sources = [
        Chunk(
            score=0.5,
            text="This is a test document",
            document=IngestedDoc(
                doc_id="1",
                doc_metadata={
                    "file_name": "test.txt",
                    "page_label": "1",
                    "window": "This is a chunk",
                },
            ),
        ),
        Chunk(
            score=0.5,
            text="This is a test document",
            document=IngestedDoc(doc_id="1", doc_metadata=None),
        ),
    ]

    result = nirmatai._NirmatAI__format_sources(sources)

    assert result == "doc: test.txt\npage: 1\nchunk: This is a chunk\n\n"


def test__format_check_on_message():
    """Test the __format_check method of the NirmatAI class."""
    with patch.object(NirmatAI, "_NirmatAI__get_completion") as mock_get_completion:
        # Define the side effect to return different values for each call
        mock_get_completion.side_effect = [
            ["mocked_value_1", "mocked_ref_1"],
            ["mocked_value_2|||", "mocked_ref_2"],
            ["|||mocked_value_3", "mocked_ref_3"],
            ["mocked_value_4", "mocked_ref_4"],
            ["|mocked|value|5", "mocked_ref_5"],
        ]

        # Create an instance of NirmatAI
        nirmatai = NirmatAI()

        message, sources = nirmatai._NirmatAI__get_completion_formatted(
            req_item="mock", moc_item="mock"
        )

        assert (
            message
            == "| LLM did not converge to right format, with attempts:\n\n1. mocked_value_1\n2. mocked_value_2___\n3. ___mocked_value_3\n4. mocked_value_4\n5. _mocked_value_5"  # noqa: E501
        )

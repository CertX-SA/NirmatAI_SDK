"""Core module for NirmataAI."""

import os
import time
from pathlib import Path
from time import strftime

import numpy as np
import pandas as pd
import pdfplumber
from pgpt_python.client import PrivateGPTApi
from pgpt_python.types import Chunk, HealthResponse, IngestedDoc
from pypdf import PdfReader

SYSTEM_PROMPT = """You are an expert Management System auditor.
Given the attached management system documents and the following requirement provide the compliance status to be one of:
["full compliance", "major non-conformity", "minor non-conformity"].
The output shall contain:
1. Compliance status which can be:
    a. major non-conformity: if you know the answer but it is not in the context.
    b. minor non-conformity: if the answer is partially provided in the context.
    c. full-compliance: if the answer is provided in the context.
2. One paragraph rationale describing the compliance status.
Separate each with a | .
Here you have a series of examples of output:
    - minor non-conformity| The written documentation does not explicitly state the certification body's processes for granting, refusing, maintaining, renewing, suspending, restoring or withdrawing certification or expanding or reducing the scope of certification.|
    - full-compliance| The requirement states that the certification body should retain authority for its decisions relating to certification. This is explicitly stated in the management system documents under "Certification Process" section, subsection 3.2.1, which clearly outlines the responsibility of the certification body and their decision-making process regarding certification.|
    - full-compliance| The certification body has demonstrated initial and ongoing evaluation of its finances and sources of income through written documentation. This ensures that commercial, financial or other pressures do not compromise the impartiality of the organization.|
"""  # noqa: E501

class NirmatAI:
    """NirmatAI class for user-facing functionalities.

    The class contains the core functionalities that the user can
    use to interact with the NirmataAI API. In order to function properly a pod needs to
    be set (look at the Makefile).

    :param system_prompt: The system prompt to be used by the LLM when processing the
        requirements.
    :type system_prompt: str, optional
    :param prompt: The prompt collecting the requirement and the means of compliance.
    :type prompt: str, optional
    :param base_url: The base URL of the NirmataAI API.
    :type base_url: str, optional
    :param timeout: The timeout for the API requests.
    :type timeout: int, optional
    :param verbose: The verbosity level.
    :type verbose: int, optional

    """

    def __init__(
        self,
        system_prompt: str = SYSTEM_PROMPT,
        prompt: str = """
            The requirement to be evaluated is: {req_item}
            The means of compliance is: {moc_item}
            """,
        base_url: str = "http://localhost:8000/",
        timeout: int = 60,
        verbose: int = 0,
    ) -> None:
        """Constructor method for initialiying the NirmatAI instance.

        This method sets up the instance with the provided or default system prompt,
        evaluation prompt, API base URL, request timeout, and verbosity level.
        It also initializes attributes for tracking true and predicted values,
        and a dictonary for storing ingested files.
        """
        # Initialize the API client with the specified base URL
        self.client = PrivateGPTApi(base_url=base_url)

        # Set the system prompt
        self.system_prompt = system_prompt

        # Set the evaluation prompt template
        self.prompt = prompt

        # Set the timeout for API requests
        self.timeout = timeout

        # Set the verbosity level for logging
        self.verbose = verbose

        # Initialize the arrays for true and predicted values
        self.y_true = np.array([np.nan])
        self.y_pred = np.array([np.nan])

        # Initialize an empty dictionary to store ingested files
        self.files: dict[str, list[IngestedDoc]] = {}

        # Initialize the stop and pause signal
        self.stop_signal = False
        self.pause_signal = False

    def health_check(self) -> HealthResponse:
        """Check the health of the NirmataAI instance.

        Method sends a health check request to the NirmatAI instance via the client.
        If the request is successful, it returns the health response.
        If an error occurs during the request, it catches the exception,
        logs the error, and returns a HealthResponse indicating the status "ko"

        :return: The health response of the NirmataAI instance.
        :rtype: HealthResponse
        """
        try:
            # health check request to the instance and returns the response
            return self.client.health.health()
        except Exception as error:
            # log the error and return a default HealthResponse with status "ko"
            print(f"Error checking health: {error!s}")
            return HealthResponse(status="ko")

    def __get_files(self, directory: str | Path) -> list[str]:
        """Get all files in a directory that are not hidden or DVC files.

        This method retrieves all non-hidden or DVC files from the specified directory.
        If the input is a file, it returns the file path.
        Handles both string and Path inputs and raises exceptions if directory is empty,
        does not exist, or is inaccessible.

        :param directory: The directory to search for files.
        :type directory: str or Path
        :return: A list of file paths that are not hidden or DVC files.
        :rtype: list[str]
        """
        # Convert Path object to string if necessary
        if isinstance(directory, Path):
            directory = str(directory)

        # Check if the path exists and is accessible
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory does not exist: {directory}")

        # Check if path is a file
        if os.path.isfile(directory):
            # Ensure it's a readable file
            if not os.access(directory, os.R_OK):
                raise PermissionError(f"File is not readable: {directory}")
            return [directory]

        # Ensure the directory path ends with a slash
        if not directory.endswith("/"):
            directory += "/"

        # Ensure the directory path is accessible and is a directory
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"Provided path is not a directory: {directory}")

        # Check for read permissions on the directory
        if not os.access(directory, os.R_OK):
            raise PermissionError(f"Directory is not accessible: {directory}")

        # Collect all files in the directory that are not hidden or DVC files
        files = [
            os.path.join(directory, file)
            for file in os.listdir(directory)
            if not file.startswith(".") and not file.endswith(".dvc")
        ]

        # Validate that the directory is not empty
        if not files:
            raise FileNotFoundError(
                f"No eligible files found in directory: {directory}"
            )

        # Check readability of each file
        unreadable_files = [
            file for file in files if not os.access(file, os.R_OK)
        ]
        if unreadable_files:
            raise PermissionError(
                f"Some files are not readable: {', '.join(unreadable_files)}"
            )

        # Optionally print files to be ingested
        if self.verbose >= 1:
            print("Files to ingest:")
            for file in files:
                print(file)

        return files

    def __is_malformed_pdf(self, file_path: str) -> bool:
        """Check if a PDF is malformed by attempting to read it."""
        try:
            reader = PdfReader(file_path)
            number_of_pages = len(reader.pages)
            if number_of_pages > 0:
                return False
        except Exception:
            return True
        return False

    def __is_scanned_pdf(self, file_path: str) -> bool:
        try:
            # Open the PDF file
            with pdfplumber.open(file_path) as pdf:
                # Iterate through each page in the PDF
                for page in pdf.pages:
                    # Try to extract text from the current page
                    text = page.extract_text()
                    # Check if any text is found on this page
                    if text and text.strip():
                        return False  # If text is found, it's not a scanned PDF
            # If no text is found in any page, assume it's a scanned PDF
            return True
        except Exception as e:
            print(f"Error processing the PDF: {e}")
            return False  # Return False in case of errors

    def ingest(self, directory: str | Path) -> None:
        """Ingest files from a given directory.

        The method finds the files in the directory through the __get_files method and
        ingests them into the NirmatAI instance. Ingestion is done using PrivateGPT.
        For the full list of supported files, refer to documentation:
        https://docs.privategpt.dev/manual/document-management/ingestion

        :param directory: The directory containing the files to be ingested.
        :type directory: str or Path
        """
        try:
            # Get all files in the directory, handling potential errors
            files_to_ingest = self.__get_files(directory)
        except Exception as e:
            print(f"Failed to retrieve files from directory {directory}. Error: {e}")
            raise

        # Prepare files dictionary with placeholders for ingestion results
        self.files = {
            file: [IngestedDoc(object="", doc_id="", doc_metadata=None)]
            for file in files_to_ingest
        }

        print("Starting ingestion...")
        for ingest_file_path in self.files:
            try:
                # Verify the file is accessible and readable
                if not os.path.isfile(
                    ingest_file_path
                ) or not os.access(
                    ingest_file_path, os.R_OK
                ):
                    raise PermissionError(f"File is not accessible: {ingest_file_path}")

                # Open and ingest the file with secure exception handling
                with open(ingest_file_path, "rb") as f:
                    try:
                        ingested_docs = self.client.ingestion.ingest_file(
                            file=f, timeout=self.timeout
                        ).data
                    except Exception as e:
                        print(
                            f"Ingestion failed for {ingest_file_path} due to an error."
                        )
                        raise RuntimeError(f"Client ingestion error: {e}") from e

                    # Store the ingested document data
                    self.files[ingest_file_path] = ingested_docs

                    # Validate ingestion: Ensure the document is now listed
                    ingested_ids = [
                        doc.doc_id for doc in self.client.ingestion.list_ingested().data
                    ]
                    if not all(
                        doc.doc_id in ingested_ids
                        for doc in self.files[ingest_file_path]
                    ):
                        # Specific error message for PDF files if ingestion issues occur
                        if ingest_file_path.endswith(".pdf"):
                            if self.__is_malformed_pdf(str(ingest_file_path)):
                                raise ValueError(
                                    "PDF ingestion failed; PDF is malformed or corrupted." # noqa: E501
                                )
                            elif self.__is_scanned_pdf(str(ingest_file_path)):
                                raise ValueError(
                                    "PDF ingestion failed; PDF is a scanned document without searchable text." # noqa: E501
                                )
                            else:
                                raise ValueError(
                                    "PDF ingestion failed for an unknown reason."
                                )
                        raise ValueError(
                            f"Ingestion failed for file: {ingest_file_path}"
                        )

                # Log successful ingestion if verbosity is enabled
                if getattr(self, "verbose", 0) >= 1:
                    print(
                        f"Ingested file successfully: {ingest_file_path}"
                    )

            except (
                FileNotFoundError, PermissionError, ValueError, RuntimeError
            ) as e:
                # Log error information and continue to the next file if feasible
                print(f"Error ingesting file: {ingest_file_path}")
                print(f"Error message: {e}")
                # Stop ingestion if the error is critical
                raise RuntimeError(
                    f"Critical ingestion error for {ingest_file_path}: {e}"
                ) from e

        if getattr(self, "verbose", 0) >= 1:
            print("Ingestion process completed.")

    def load_requirements(self, reqs_file: str | Path) -> None:
        """Load requirements from an Excel file.

        The method loads the requirements from an Excel file and stores them in a pandas
        DataFrame. If the column Label exists, it is stored in the y_true attribute.

        :param reqs_file: The path to the Excel file containing the requirements.
        :type reqs_file: str
        """
        # If reqs_file is a Path object, convert it to a string
        if isinstance(reqs_file, Path):
            reqs_file = str(reqs_file)

        # Check if the file exists
        if not os.path.exists(reqs_file):
            raise FileNotFoundError(
                "Requirements file not found."
            )

        # Check if the file is a .xlsx file
        if not reqs_file.endswith(".xlsx"):
            raise ValueError(
                "Requirements file is not an Excel file."
            )

        try:
            # Load the requirements from the Excel file into a pandas DataFrame
            self.reqs = pd.read_excel(reqs_file)

            # Ensure the file is not empty
            if self.reqs.empty:
                raise pd.errors.EmptyDataError(
                    f"The Excel file is empty: {reqs_file}"
                )

            # Check if the required columns exist
            required_columns = [
                "Requirement",
                "Potential Means of Compliance"
            ]
            for column in required_columns:
                if column not in self.reqs.columns:
                    raise ValueError(
                        f"Missing required column: {column}"
                    )
            # Check for "Label" column and handle accordingly
            if "Label" in self.reqs.columns:
                if self.reqs["Label"].isnull().all():
                    # Warn if Label column is present but contains no valid data
                    print(
                        "Warning: 'Label' column is present but contains no valid data."
                    )
                else:
                    print("Storing 'Label' column into y_true.")
                    # Save the Label column to y_true
                    self.y_true = self.reqs["Label"].to_numpy()
            else:
                print(
                    "'Label' column not found. Continuing without storing labels."
                )

        except pd.errors.EmptyDataError as e:
            raise pd.errors.EmptyDataError(
                f"Error reading Excel file: {e}"
            ) from e
        except ValueError as e:
            raise ValueError(
                f"Error with Excel file structure: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred while loading the requirements: {e}"
            ) from e

    def __get_completion(self, req_item: str, moc_item: str) -> tuple[str, list[Chunk]]:
        """Get the completion from the LLM.

        This method uses the LLM to generate a completion based on the given requirement
        and means of compliance. It retrieves the result, checks for errors, and returns
        the completion message along with the list of source chunks.

        :param req_item: The requirement item to be evaluated.
        :type req_item: str
        :param moc_item: The means of compliance item.
        :type moc_item: str
        :return: A tuple containing the completion message and list of source chunks
        :rtype: tuple[str, list[Chunk]]
        """
        # Generate the prompt and get the completion from the LLM
        result = self.client.contextual_completions.prompt_completion(
            system_prompt=self.system_prompt,
            prompt=self.prompt.format(req_item=req_item, moc_item=moc_item),
            use_context=True,
            include_sources=True,
            timeout=self.timeout,
        ).choices[0]

        # Check if the message is not None and is a string
        if result.message is None or not isinstance(result.message.content, str):
            raise ValueError("The message is None. Please check the prompt.")

        # Validate the result sources
        if not isinstance(result.sources, list):
            raise ValueError("The sources are not a list. Please check the prompt.")

        # Ensure there is at least one source in the list
        if not result.sources:
            result.sources.append(
                Chunk(
                    object="",
                    score=0.0,
                    text="Empty",
                    document=IngestedDoc(
                        object=None, doc_id="1", doc_metadata={"file_name": "none"}
                    ),
                    previous_texts=None,
                    next_texts=None,
                )
            )

        return result.message.content, result.sources

    def __format_sources(self, sources: list[Chunk]) -> str:
        """Format the sources for the output.

        This method formats a list of sources by extracting metadata information such as
        file name, page label, and chunk window from each source's document metadata.
        It joins these details into a single formatted string,
        with each source separated by a semicolon and newline.

        :param sources: A list of Chunk objects containing document metadata.
        :type sources: list[Chunk]
        :return: A formatted string containing the extracted metadata from each source.
        :rtype: str
        """
        # Initialize an empty list to store formatted source strings
        formatted_sources = []

        # Iterate over each source in the provided list
        for source in sources:
            try:
                # Ensure that the source has a 'document' attribute with 'doc_metadata'
                if not hasattr(
                    source,
                    "document"
                ) or not hasattr(source.document, "doc_metadata"):
                    raise AttributeError(
                        f"Source {source} is missing required document metadata."
                    )
                # Extract the document metadata from the source
                metadata = source.document.doc_metadata

                # Ensure metadata is a valid dictionary
                if not isinstance(metadata, dict):
                    raise ValueError(
                        f"Invalid metadata for source: {source}"
                    )

                # Proceed only if metadata is available
                if metadata:
                    # Extract file name, page label, and chunk window from metadata
                    file_name = metadata.get("file_name", "Unknown file")
                    page_label = metadata.get("page_label", "Unknown page")
                    chunk_window = metadata.get("window", "Unknown chunk")

                    # Format the extracted metadata into a structured string
                    formatted_source = (
                        f"doc: {file_name}\n"
                        f"page: {page_label}\n"
                        f"chunk: {chunk_window}\n\n"
                    )

                    # Add thr formatted string to the list of formatted sources
                    formatted_sources.append(formatted_source)
            except (AttributeError, ValueError, KeyError) as e:
                # Source does not have the required attributes or metadata
                print(
                    f"Warning: Skipping source due to error: {e}"
                )
                continue  # Skip this source and move to the next one
            except Exception as e:
                print(
                    f"Unexpected error occurred while formatting the requirements: {e}"
                )
                continue

        # Join all formatted sources into a single string, separated by a semicolon
        if formatted_sources:
            return "| ".join(formatted_sources)
        else:
            return "No valid sources were found."


    def __get_completion_formatted(
        self, req_item: str, moc_item: str, attempts: int = 5
    ) -> tuple[str, list[Chunk]]:
        """Get the completion from the LLM.

        This method uses the LLM to generate a completion based on the given
        requirement.
        It retrieves the result, checks for errors, and returns the completion message
        along with the list of source chunks. If the completion does not conform to the
        expected format, it retries the completion up to the specified number of
        attempts.

        :param req_item: The requirement item to be evaluated.
        :type req_item: str
        :param moc_item: The means of compliance item.
        :type moc_item: str
        :param attempts: The number of attempts to get the completion, defaults to 5.
        :type attempts: int, optional
        :return: A tuple containing the completion message and list of source chunks
        :rtype: tuple[str, list[Chunk]]
        """
        out = "| LLM did not converge to right format, with attempts:\n"
        for i in range(attempts):
            message, sources = self.__get_completion(req_item, moc_item)
            if message.count("|") in [1, 2]:
                out = message
                break
            else:
                out += f"\n{i+1}. {message.replace("|", "_")}"
        return out, sources

    def process_requirements(self) -> pd.DataFrame:
        """Process the requirements and get the results.

        The method processes the requirements, passing them to RAG, then extracts the
        response and formats it into a DataFrame. The DataFrame contains the compliance
        status, the rationale, and the reference to the document.

        :return: The DataFrame containing the compliance status, the rationale, and the
            reference to the document.
        :rtype: pd.DataFrame
        """
        try:
            print("Processing requirements...")

            if self.verbose >= 2:
                print("Prompts are as follows:")
                print(f"System Prompt: {self.system_prompt}")
                print(f"Prompt: {self.prompt}")

            # Initialize lists to store results
            comp_status, rationale, ref_to_doc = [], [], []

            # Check if DataFrame is empty
            if self.reqs.empty:
                print("No requirements to process. The DataFrame is empty.")
                return pd.DataFrame(
                    columns=["Compliance status", "Rationale", "Ref. to Doc"]
                )

            # Ensure required columns exist in the DataFrame
            required_columns = [
                "Requirement",
                "Potential Means of Compliance"
            ]
            if not all(
                col in self.reqs.columns for col in required_columns
            ):
                raise ValueError(
                    f"DataFrame must contain the following columns: {required_columns}"
                )

            print(f"Number of requirements to be processed: {len(self.reqs)}")

            # Iterate through each row in the requirements DataFrame
            for index, row in self.reqs.iterrows():
                # Stop if stop_signal is set
                if self.stop_signal:
                    print("Process stopped by the user.")
                    self.stop_signal = False
                    break

                # Pause the process if pause_signal is set
                while self.pause_signal:
                    print("Process paused by the user. Waiting to resume...")
                    time.sleep(15)

                # Error handling for missing or NaN values in the Requirement column
                if pd.isna(row["Requirement"]) or row["Requirement"] == "":
                    print(
                        f"Skipping row {index} due to missing 'Requirement' value."
                    )
                    comp_status.append("Error")
                    rationale.append(
                        f"Skipping row {index} due to missing 'Requirement' value."
                    )
                    ref_to_doc.append("N/A")
                    continue

                # Handle verbose logging
                if self.verbose >= 1:
                    print(f"Processing row {index}")
                if self.verbose >= 2:
                    print(f"Requirement: {row['Requirement']}")
                    print(f"Potential MoC: {row['Potential Means of Compliance']}")

                req_item = row["Requirement"]
                moc_item = row["Potential Means of Compliance"]

                # If moc_item is NaN, replace it with a string asking for documentation
                if pd.isna(moc_item) or moc_item == "":
                    moc_item = "Written documentation shall be provided to comply."

                try:
                    # Call internal function to get formatted message and sources
                    message, sources = self.__get_completion_formatted(
                        req_item,
                        moc_item
                    )
                except Exception as e:
                    print(f"Error processing row {index}: {e}")
                    comp_status.append("Error")
                    rationale.append(f"Error processing row {index}: {e}")
                    ref_to_doc.append("N/A")
                    continue

                # Split the message and handle potential issues with formatting
                try:
                    split_result = message.split("|")
                    if len(split_result) < 2:
                        raise ValueError("Invalid response format!")
                except Exception as e:
                    print(f"Error in formatting for row {index}: {e}")
                    comp_status.append("Error")
                    rationale.append(f"Error in formatting for row {index}: {e}")
                    ref_to_doc.append("N/A")
                    continue

                # Extract compliance status and rationale
                try:
                    split_result[0] = self.__extract_comp_status(split_result[0])
                except Exception as e:
                    print(f"Error extracting compliance status for row {index}: {e}")
                    comp_status.append("Error")
                    rationale.append(
                        f"Error extracting compliance status for row {index}: {e}"
                    )
                    ref_to_doc.append("N/A")
                    continue

                comp_status.append(split_result[0])
                rationale.append(split_result[1])

                # Format the source information
                try:
                    ref_to_doc.append(self.__format_sources(sources))
                except Exception as e:
                    print(f"Error formatting sources for row {index}: {e}")
                    ref_to_doc.append("N/A")

            # Create DataFrame with results
            llm_raw_output = pd.DataFrame(
                {
                    "Compliance status": comp_status,
                    "Rationale": rationale,
                    "Ref. to Doc": ref_to_doc,
                }
            )

            # Store Compliance status in y_pred for further scoring
            self.y_pred = llm_raw_output["Compliance status"].to_numpy()

            return llm_raw_output

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return pd.DataFrame(
                columns=["Compliance status", "Rationale", "Ref. to Doc"]
            )

    def stop_processing(self) -> None:
        """Stop the process_requirements method by setting the stop signal."""
        print(f"Time: {strftime('%c')}. Stopping process...")
        self.stop_signal = True

    def pause_processing(self) -> None:
        """Pause the process_requirements method."""
        print(f"Time: {strftime('%c')}. Pausing process...")
        self.pause_signal = True

    def resume_processing(self) -> None:
        """Resume the process_requirements method."""
        print(f"Time: {strftime('%c')}. Resuming process...")
        self.pause_signal = False

    def __extract_comp_status(self, compliance: str) -> str:
        """Extract the compliance status from the result.

        This method processes the given compliance string and checks for
        specific keywords that indicate the compliance status.
        If the status is not explicitly found, it defaults to itself.

        :param compliance: The input compliance.
        :type compliance: str
        :return: The extracted compliance status.
        :rtype: str
        """
        # Ensure the input is a valid string
        if not isinstance(compliance, str):
            raise ValueError("Compliance input must be a string.")

        # Lowercase and strip extra whitespace from the input for uniform processing
        compliance_status = compliance.strip().lower()

        # Define possible compliance statuses and their variations for robustness
        status_map = {
            "full-compliance": [
                "full-compliance", "full compliance", "compliant", "full-conformity",
                "compliance", "fully compliant", "meets requirements",
                "adhered to", "satisfied all conditions", "in full compliance",
                "no issues", "complete adherence", "perfect compliance",
                "fully aligned", "met all criteria", "in line with standards",
                "fully followed", "complete conformity", "total compliance"
            ],
            "minor non-conformity": [
                "minor non-conformity", "minor breach", "minor non conformity",
                "low-risk non-compliance", "small issue", "minor issue",
                "minor non-compliance", "small non-conformance",
                "slight non-compliance", "minor", "small non-adherence",
                "low severity non-conformity", "negligible non-compliance",
                "trivial non-conformity", "minor deviation"
            ],
            "major non-conformity": [
                "major non-conformity", "major", "major non conformity",
                "noncompliance", "non-compliance", "significant issue",
                "critical non-conformance", "high-risk non-conformity",
                "serious non-conformity","critical failure",
                "substantial non-compliance", "major issue",
                "severe non-compliance", "serious breach",
                "critical non-adherence", "large non-conformity",
                "severe breach", "major infraction"
            ]
        }

        # Attempt to match the compliance status using keywords in the status_map
        for status, synonyms in status_map.items():
            for synonym in synonyms:
                if synonym in compliance_status:
                    return status

        # If no match is found, log a warning and return a default value
        print(
            f"Warning: Unrecognized compliance status '{compliance_status}', defaulting to 'major non-conformity'." # noqa: E501
        )

        # Default to "major non-conformity" if no specific status is found
        return ""

    def save_results(
            self,
            dataframe: pd.DataFrame,
            output_path: str,
            attach_reqs: bool = False
    ) -> None:
        """Save the results to file.

        The method saves the results to a file. The file can be in CSV or HTML format.
        If the attach_reqs is True, the requirements are attached to the results.

        :param dataframe: The DataFrame containing the results.
        :type dataframe: pd.DataFrame
        :param output_path: The path to the output file.
        :type output_path: str
        :param attach_reqs: Whether to attach the requirements to the results, defaults
            to False.
        :type attach_reqs: bool, optional

        :raises FileNotFoundError: If the directory does not exist.
        :raises AttributeError: If attach_reqs is True but not a valid DataFrame.
        :raises ValueError: If the file extension is not supported.
        :raises IOError: If there are issues writing the file.
        """
        try:
            # Check if output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                raise FileNotFoundError(
                    f"Output directory does not exist: {output_dir}"
                )

            # Check if the DataFrame is empty
            if dataframe.empty:
                print("Warning: The DataFrame is empty. An empty file will be saved.")

            # Attach requirements if requested
            if attach_reqs:
                if hasattr(self, "reqs") and isinstance(self.reqs, pd.DataFrame):
                    if not self.reqs.empty:
                        dataframe = pd.concat([self.reqs, dataframe], axis=1)
                    else:
                        raise ValueError(
                            "'reqs' DataFrame is empty, cannot attach requirements."
                        )
                else:
                    raise AttributeError(
                        "The 'reqs' attribute must be a valid DataFrame to attach."
                    )

            # Check file extension and save the file accordingly
            file_extension = os.path.splitext(output_path)[1].lower()
            if file_extension == ".csv":
                # Save as CSV file
                dataframe.to_csv(output_path, index=False)
            elif file_extension == ".html":
                # Replace newline characters with <br> for HTML rendering
                dataframe = dataframe.replace("\n", "<br>", regex=True)
                # Save as HTML file
                dataframe.to_html(output_path, escape=False, index=False)
            else:
                raise ValueError(
                    f"Unsupported file extension: {file_extension}"
                )

            # Confirm file save
            if getattr(self, "verbose", 0) >= 1:
                print(f"Results saved to {output_path}")

        except FileNotFoundError as e:
            print(f"Error: {e}")
        except PermissionError:
            print(f"Error: Permission denied. Cannot write to {output_path}.")
        except AttributeError as e:
            print(f"Error: {e}")
        except ValueError as e:
            print(f"Error: {e}")
        except OSError as e:
            print(f"IOError: Failed to write the file due to {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def delete_all_documents(self) -> None:
        """Delete all ingested documents.

        The method deletes all the ingested documents from the RAG system
        by iterating through the list of ingested documents and delete each
        """
        try:
            # Retrieve the list of all ingested documents
            ingested_docs = self.client.ingestion.list_ingested().data

            # Check if there are any ingested documents to delete
            if not ingested_docs:
                print("No ingested documents found.")
                return

            # Iterate over each ingested document and delete it
            for doc in ingested_docs:
                try:
                    self.client.ingestion.delete_ingested(doc.doc_id)
                    if getattr(self, "verbose", 0) >= 1:
                        print(f"Deleted document: {doc.doc_id}")
                except Exception as error:
                    print(f"Error deleting document: {doc.doc_id}")
                    print(f"Error message: {error}")
                    raise
            if getattr(self, "verbose", 0) >= 1:
                print("All ingested documents have been deleted.")
        except Exception as error:
            print(f"An error occurred while retrieving the documents: {error}")
            raise

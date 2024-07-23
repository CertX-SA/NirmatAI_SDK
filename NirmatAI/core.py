"""Core module for NirmataAI."""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pgpt_python.client import PrivateGPTApi
from pgpt_python.types import Chunk, HealthResponse, IngestedDoc

SYSTEM_PROMPT = """You are an expert Management System auditor.
Given the attached management system documents and the following requirement provide the compliance status to be one of:
["full compliance", "major non-conformity", "minor non-conformity"].
The output shall contain:
1. Compliance status which can be:
    a. major non-conformity: if you know the answer but it is not in the context.
    b. minor non-conformity: if the answer is partially provided in the context.
    c. full-compliance: if the answer is provided in the context.
2. One paragraph rationale describing the compliance status.
Separate each with a ; .
Here you have a series of examples of output:
    - minor non-conformity; The written documentation does not explicitly state the certification body's processes for granting, refusing, maintaining, renewing, suspending, restoring or withdrawing certification or expanding or reducing the scope of certification.;
    - full-compliance; The requirement states that the certification body should retain authority for its decisions relating to certification. This is explicitly stated in the management system documents under "Certification Process" section, subsection 3.2.1, which clearly outlines the responsibility of the certification body and their decision-making process regarding certification.;
    - full-compliance; The certification body has demonstrated initial and ongoing evaluation of its finances and sources of income through written documentation. This ensures that commercial, financial or other pressures do not compromise the impartiality of the organization.;
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
            #log the error and return a default HealthResponse with status "ko"
            print(f"Error checking health: {error!s}")
            return HealthResponse(status="ko")

    def __get_files(self, directory: str | Path) -> list[str]:
        """Get all files in a directory that are not hidden or DVC files.

        This method retrieves all non-hidden or DVC files from the specified directory.
        If the input is a file, it returns the file path.
        The method handles both string and Path inputs
        for the directory and raises exceptions
        if the directory is empty or does not exist.

        :param directory: The directory to search for files.
        :type directory: str or Path
        :return: A list of file paths that are not hidden or DVC files.
        :rtype: list[str]
        """
        # Convert Path object to string if necessary
        if isinstance(directory, Path):
            directory = str(directory)
            # Check if the path is a file, and return if it is true
        if os.path.isfile(directory):
            return [directory]

        # Ensure the directory path ends with a slash
        if not directory.endswith("/"):
            directory += "/"

        # Check if the directory exists
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Check if the directory is empty
        if not os.listdir(directory):
            raise FileNotFoundError(f"Directory is empty: {directory}")

        # Get all files in the directory that are not hidden or DVC files
        files = [
            os.path.join(directory, file)
            for file in os.listdir(directory)
            if not (file.startswith(".") or file.endswith(".dvc"))
        ]

        # Optionally print files to be ingested
        if self.verbose >= 1:
            print("Files to ingest:")
            for file in files:
                print(file)

        return files

    def ingest(self, directory: str) -> None:
        """Ingest files from a given directory.

        The method finds the files in the directory through the `__get_files` method and
        ingests them into the NirmataAI instance. Ingestion is done using PrivateGPT.
        For the full list of supported files, refer to documentation:
        https://docs.privategpt.dev/manual/document-management/ingestion

        :param directory: The directory containing the files to be ingested.
        :type directory: str
        """
        # Get all files in the directory
        files_to_ingest = self.__get_files(directory)
        self.files = {
            file: [IngestedDoc(object="", doc_id="", doc_metadata=None)]
            for file in files_to_ingest
        }

        # Ingest files
        print("Starting ingestion...")
        for ingest_file_path in self.files:
            try:
                # Open the file in binary read mode
                with open(ingest_file_path, "rb") as f:
                    # Ingest the file and update the ingested documents
                    ingested_docs = self.client.ingestion.ingest_file(
                        file=f, timeout=self.timeout
                    ).data
                    self.files[ingest_file_path] = ingested_docs

                    # Check if the document was successfully ingested
                    if not all(
                        doc in self.client.ingestion.list_ingested().data
                        for doc in self.files[ingest_file_path]
                    ):
                        # If the document is a pdf file, be more specific
                        if ingest_file_path.endswith(".pdf"):
                            raise ValueError("PDF not ingested, perhaps scanned.")
                        raise ValueError("Document not ingested")
                if getattr(self, "verbose", 0) >= 1:
                    print(f"Ingested file: {ingest_file_path}")
            except Exception as e:
                print(f"Error ingesting file: {ingest_file_path}")
                print(f"Error message: {e}")
                # Throw an error if the document was not ingested
                raise
        if getattr(self, "verbose", 0) >= 1:
            print("Ingestion process completed.")

    def load_requirements(self, reqs_file: str | Path) -> None:
        """Load requirements from an Excel file.

        The method loads the requirements from an Excel file and stores them in a pandas
        DataFrame. If the column Label exists, it is stored in the `y_true` attribute.

        :param reqs_file: The path to the Excel file containing the requirements.
        :type reqs_file: str
        """
        # If reqs_file is a Path object, convert it to a string
        if isinstance(reqs_file, Path):
            reqs_file = str(reqs_file)
        # Check if the file exists
        if not os.path.exists(reqs_file):
            raise FileNotFoundError("Requirements file not found.")
        # Check if the file is a .xlsx file
        if not reqs_file.endswith(".xlsx"):
            raise ValueError("Requirements file is not an Excel file.")

        #Load the requirements from the Excel file into a pandas DataFrame
        self.reqs = pd.read_excel(reqs_file)

        # Check if the colums Requirement and Potential Means of Compliance exist
        required_columns = ["Requirement", "Potential Means of Compliance"]
        for column in required_columns:
            if column not in self.reqs.columns:
                raise ValueError(f"{column} column not found.")

        # If the column Label exits, print that you are using it
        if "Label" in self.reqs.columns:
            print("Storing Label column into y_true.")
            # Save it in y_true
            self.y_true = self.reqs["Label"].to_numpy()
        # I feel like else or a raise can make this function more robust

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
            # Extract the document metadata from the source
            metadata = source.document.doc_metadata

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

        # Join all formatted sources into a single string, separated by a semicolon
        return "; ".join(formatted_sources)

    def process_requirements(self) -> pd.DataFrame:
        """Process the requirements and get the results.

        The method processes the requirements passing them to the RAG then extracts the
        response and formats it into a DataFrame. The DataFrame contains the compliance
        status, the rationale, and the reference to the document.

        :return: The DataFrame containing the compliance status, the rationale, and the
            reference to the document.
        :rtype: pd.DataFrame

        """
        print("Processing requirements...")
        if self.verbose >= 2:
            print("System Prompt: ", self.system_prompt)
            print("Prompt: ", self.prompt)

        comp_status, rationale, ref_to_doc = [], [], []

        # Iterate through each row in the requirements DataFrame
        for index, row in self.reqs.iterrows():
            if self.verbose >= 1:
                print(f"Processing row {index}")
            if self.verbose >= 2:
                print(f"Requirement: {row['Requirement']}")
                print(f"Potential MoC: {row['Potential Means of Compliance']}")

            req_item = row["Requirement"]
            moc_item = row["Potential Means of Compliance"]

            # If moc_item is NaN, replace it with a string asking for documentation
            if pd.isna(moc_item):
                moc_item = "Written documentation shall be provided to comply."

            message, sources = self.__get_completion(req_item, moc_item)

            # If the number of ; in messagge is not between 1 and 2 raise a warning
            # Initialize a counter
            attempt_count = 0

            # Loop with a maximum of 5 attempts
            while message.count(";") not in [1, 2]:
                if attempt_count < 5:
                    warnings.warn(
                        "The completion does not have the correct format.",
                        UserWarning,
                        stacklevel=2,
                    )
                    message, sources = self.__get_completion(req_item, moc_item)
                    attempt_count += 1  # Increment the counter
                else:
                    # Set the message after 5 unsuccessful attempts and exit the loop
                    message = (
                        "major non-conformity; LLM did not converge to right format"
                    )
                    break

            if self.verbose >= 2:
                print("Result: ", message)
            split_result = message.split(";")

            # Extract from split_result[0] the compliance status that can be:
            # major non-conformity, minor non-conformity, full-compliance
            split_result[0] = self.__extract_comp_status(split_result[0])

            comp_status.append(split_result[0])
            rationale.append(split_result[1])

            # Extract the source used in the context
            ref_to_doc.append(self.__format_sources(sources))

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

    def __extract_comp_status(self, compliance: str) -> str:
        """Extract the compliance status from the result.

        This method processes the given compliance and checks for specific keywords
        that indicates the compliance status.
        If the status is not explicitly found, it defaults to "major non-conformity".

        :param compliance: The input compliance.
        :type compliance: str
        :return: The extracted compliance status.
        :rtype: str
        """
        # Lower the comp_status
        compliance_status = compliance.lower()

        # Define the possible compliance statuses
        statuses = ["major non-conformity", "minor non-conformity", "full-compliance"]

        # Check for each status in the compliance status string
        for status in statuses:
            if status in compliance_status:
                return status

        # Default to "major non-conformity" if no specific status is found
        return "major non-conformity"

    def save_results(
        self, dataframe: pd.DataFrame, output_path: str, attach_reqs: bool = False
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

        """
        # Check if the directory of the output_path exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            raise FileNotFoundError(f"Output directory does not exist: {output_dir}.")

        # Attach requirements to the results if attach_reqs is True
        if attach_reqs:
            if hasattr(self, "reqs") and isinstance(self.reqs, pd.DataFrame):
                dataframe = pd.concat([self.reqs, dataframe], axis=1)
            else:
                raise AttributeError("The 'reqs' attribute must be a dataframe!")

        # Determine the file extension and save the file accordingly
        file_extension = os.path.splitext(output_path)[1].lower()
        if file_extension == ".csv":
            # Save as csv file
            dataframe.to_csv(output_path, index=False)
        elif file_extension == ".html":
            # Replace all \n with <br> for HTML rendering
            dataframe = dataframe.replace("\n", "<br>", regex=True)
            # Save as HTML file
            dataframe.to_html(output_path, escape=False, index=False)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")

        # If verbose level is set to 1 or higher, print a confirmation message
        if getattr(self, "verbose", 0) >= 1:
            print(f"Results saved to {output_path}")

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

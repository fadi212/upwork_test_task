import os
import re
from urllib.parse import urlparse
from fastapi import UploadFile
import logging
from openai import OpenAI
from pathlib import Path
from pydantic import BaseModel, Field
import tiktoken


def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)

        logger.addHandler(ch)


setup_logger()

class InitialFieldsExtraction(BaseModel):
    title: str = Field(description="Title of Document")
    authors: list[str] =  Field(description="Author of Document")
    manufacturers: list[str] = Field(description="Manufacturers of Guns if mentioned")
    date_published: str = Field(description="Published Date of Document if mentioned")
    issue_date: str = Field(description="Issue Date of Document if mentioned")
    origin_type: str = Field(description="Source type")

client = OpenAI()

aryn_ignored_elements = [
    "PageBreak", "Header", "Footer", "CodeSnippet", "PageNumber", "UncategorizedText"]

def split_response_and_citations(answer: str):
    # Capture digits inside square brackets
    citation_pattern = r'\[(\d+(?:,\s*\d+)*)\]'

    citations = re.findall(citation_pattern, answer)

    citations = [list(map(int, citation.split(', ')))
                 for citation in citations]

    response = re.sub(citation_pattern, '', answer).strip()

    return response, citations


def rename_json_file(input_path):
    directory = os.path.dirname(input_path)
    new_name = os.path.basename(input_path).replace(".pdf.json", ".json")
    new_path = os.path.join(directory, new_name)

    os.rename(input_path, new_path)
    return new_path


def get_filenames_in_folder_local(folder_path):
    return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]


def table_description_extractor(before_table_text, table_content, after_table_text):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"""Given the following information, generate a highly detailed, fully descriptive, and table description in the form of first single paragraph that includes:
            
            - Every bit of information present in the table
            - All symbols, units, footnotes, and special formatting
            - Key numerical values, data trends, and relationships
            - Column and row headers with precise labels
            - Any equations, abbreviations, or annotations present
            - Structural and contextual importance of the table
            - Any relevant metadata that enhances understanding
            
            Ensure that the description preserves the full meaning of the table without omitting any details.
            Text Before Table:
            {before_table_text}
            
            Table:
            {table_content}
            
            Text After Table:
            {after_table_text}
"""
             }
        ]
    )

    return completion.choices[0].message.content


def extract_initials(initial_content):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": """You are an expert in structured data extraction and analysis. You will be provided with text content from a book or magazine related to firearms.
             Your task is to extract specific fields of information from this content in a structured format. Additionally, analyze and extract the origin type (e.g., book, magazine).
             For each provided text, identify, extract, and analyze the following fields as accurately as possible, ensuring that all information is clear, complete, and follows the given structure"""},
            {"role": "user", "content": f"{initial_content}"}
        ],
        response_format=InitialFieldsExtraction,
    )

    response = completion.choices[0].message.parsed

    return response


def get_file_object(local_file_path: str) -> UploadFile:
    """
    Creates an UploadFile-like object from a local file path.

    Args:
        local_file_path (str): The path to the local file.

    Returns:
        UploadFile: An UploadFile-like object for the file.
    """
    file_path = Path(local_file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"The file {local_file_path} does not exist.")

    # Open the file as a binary stream
    file_stream = file_path.open("rb")
    return UploadFile(filename=file_path.name, file=file_stream)

def count_tokens(text: str, model: str = "gpt-4") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base encoding if the model is not found
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

# sanitize for metadata
def sanitize_metadata(metadata):
    sanitized_metadata = {}
    for key, value in metadata.items():
        if value is None:
            # Assign default values based on expected type
            if key in ['layout_width', 'layout_height']:
                sanitized_metadata[key] = 0  # Default numerical value
            elif key in ['title', 'author', 'manufacturer', 'date_published', 'issue_date', 'website_url','origin_type']:
                sanitized_metadata[key] = ""
            else:
                sanitized_metadata[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            sanitized_metadata[key] = value
        elif isinstance(value, list):
            sanitized_metadata[key] = [str(item) for item in value]
        else:
            # Convert any other types to string
            sanitized_metadata[key] = str(value)
    return sanitized_metadata

def format_chat_history(messages) -> str:
    return "\n".join(f"{msg.role}: {msg.content}" for msg in messages)
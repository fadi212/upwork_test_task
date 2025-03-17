import os
import json
import pickle
import logging
import shutil
import tempfile
import time
from datetime import datetime
# Aryn library
from aryn_sdk.partition import partition_file, table_elem_to_dataframe
from aryn_sdk.config import ArynConfig
# LlamaIndex
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
# Pinecone
from pinecone import ServerlessSpec
# Local utilities
from src.utils.utils import (
    table_description_extractor,
    aryn_ignored_elements,
    extract_initials,
    sanitize_metadata,
    count_tokens
)
#config
from src.utils.config import (
    LOCAL_INDEX_LOGS_PATH,
    LOCAL_UPSERT_LOGS_PATH,
    LOCAL_CHUNK_OUTPUT_DIR,
    ARYN_API_KEY,
    INDEX_NAME,
    pc  
)


def load_local_logs(logs_path: str) -> list:
    if not os.path.exists(logs_path):
        return []
    try:
        with open(logs_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, FileNotFoundError):
        logging.warning(f"{logs_path} is empty or corrupted. Initializing new logs.")
        return []


def save_local_logs(logs_path: str, logs_data: list):
    try:
        with open(logs_path, "w", encoding="utf-8") as f:
            json.dump(logs_data, f, indent=4)
    except Exception as e:
        logging.error(f"Failed to save logs to {logs_path}: {e}")


def run_aryn_pipeline(
    local_pdf_path: str,
    output_folder: str,
    aryn_api_key: str = None
) -> str:
    """
    Use Aryn to partition and chunk a file. Save the result
    into a local output folder. Return the path to the serialized file.
    """
    if not os.path.exists(local_pdf_path):
        logging.error(f"File not found: {local_pdf_path}")
        return ""

    pdf_basename = os.path.basename(local_pdf_path)
    pdf_name_no_ext = os.path.splitext(pdf_basename)[0]

    # Create a temporary folder to do the chunking
    temp_dir = tempfile.mkdtemp(prefix=f"aryn_{pdf_name_no_ext}_")
    final_serialized_path = os.path.join(temp_dir, f"{pdf_name_no_ext}.json")

    try:
        with open(local_pdf_path, "rb") as f:
            chunking_opts = {
                "strategy": "context_rich",
                "tokenizer": "openai_tokenizer",
                "tokenizer_options": {
                    "model_name": "gpt-3.5-turbo"
                },
                "merge_across_pages": False,
                "max_tokens": 512
            }

            if aryn_api_key:
                aryn_config = ArynConfig(aryn_api_key=aryn_api_key)
            else:
                aryn_config = None

            # Partition the file with Aryn
            data = partition_file(
                f,
                aryn_api_key=aryn_api_key,
                aryn_config=aryn_config,
                extract_table_structure=True,
                extract_images=False,
                use_ocr=True,
                chunking_options=chunking_opts
            )
        try:
            with open(final_serialized_path, "w", encoding="utf-8") as out_f:
                json.dump(data, out_f, indent=4)
            logging.info(f"Serialized Aryn output to JSON at {final_serialized_path}")
        except TypeError:
            # If data not JSON-serializable, fallback to pickle
            final_pickle_path = os.path.join(temp_dir, f"{pdf_name_no_ext}.pickle")
            with open(final_pickle_path, "wb") as out_f:
                pickle.dump(data, out_f)
            logging.info(f"Serialized Aryn output to Pickle at {final_pickle_path}")
            final_serialized_path = final_pickle_path

        # Move final_serialized_path into output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)
        final_filename = os.path.basename(final_serialized_path)
        local_output_path = os.path.join(output_folder, final_filename)
        shutil.move(final_serialized_path, local_output_path)

        return local_output_path

    except Exception as e:
        logging.error(f"Error in run_aryn_pipeline for {local_pdf_path}: {e}")
        return ""

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def extract_llama_index_docs(
    json_or_pickle_path: str,
    source_type: str,
    website_url: str = "",
    local_pdf_path: str = ""
) -> list:
    if not os.path.exists(json_or_pickle_path):
        logging.error(f"File not found: {json_or_pickle_path}")
        return []

    # Load chunked data
    try:
        if json_or_pickle_path.endswith(".json"):
            with open(json_or_pickle_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif json_or_pickle_path.endswith(".pickle"):
            with open(json_or_pickle_path, "rb") as f:
                data = pickle.load(f)
        else:
            logging.error(f"Unsupported file format: {json_or_pickle_path}")
            return []
    except Exception as e:
        logging.error(f"Failed to load chunked data from {json_or_pickle_path}: {e}")
        return []

    elements = data.get("elements", [])
    if not elements:
        logging.warning(f"No 'elements' found in {json_or_pickle_path}.")
        return []

    # Extract metadata from first ~5 pages
    initial_book_content = ""
    for idx, item in enumerate(elements):
        page_number = item.get("properties", {}).get("page_number", 1)
        if page_number > 5:
            break

        element_type = item.get("type", "").lower()
        if element_type in [elem.lower() for elem in aryn_ignored_elements]:
            continue

        if element_type == "table":
            try:
                df = table_elem_to_dataframe(item)
                table_text = df.to_string(index=False)
                initial_book_content += table_text + "\n"
            except Exception as e:
                logging.warning(f"Failed to convert table at idx={idx}: {e}")
        else:
            text = item.get("text_representation") or ""
            initial_book_content += text + "\n"

    # Attempt to extract metadata from the first 5 pages
    book_initials = extract_initials(initial_book_content)
    logging.info(f"Extracted from first pages: {book_initials}")

    # Build the final Documents
    llama_docs = []
    for idx, item in enumerate(elements):
        metadata = item.get("metadata", {})
        properties = item.get("properties", {})

        # Fill in some metadata from "book_initials"
        if book_initials.title:
            metadata["title"] = book_initials.title
        if book_initials.authors:
            metadata["author"] = book_initials.authors
        if book_initials.manufacturers:
            metadata["manufacturer"] = book_initials.manufacturers
        if book_initials.date_published:
            metadata["date_published"] = book_initials.date_published
        if book_initials.issue_date:
            metadata["issue_date"] = book_initials.issue_date
        if book_initials.origin_type:
            metadata["origin_type"] = book_initials.origin_type
        if website_url:
            metadata["website_url"] = website_url

        # Standard metadata
        metadata["page_number"] = properties.get("page_number", 1)
        metadata["parent_id"] = properties.get("parent_id", "")
        metadata["filename"] = os.path.basename(local_pdf_path)
        metadata["type"] = item.get("type", "")
        metadata["element_id"] = metadata.get("element_id", "")

        element_type = item.get("type", "").lower()
        text_rep = item.get("text_representation") or ""

        # If table, merge text around it
        if element_type == "table":
            before_table_text = ""
            after_table_text = ""

            # Look backward
            cur_idx = idx
            while len(before_table_text.split()) < 400:
                cur_idx -= 1
                if cur_idx < 0:
                    break
                prev_item = elements[cur_idx]
                prev_type = prev_item.get("type", "").lower()
                if prev_type in ["table"] + [x.lower() for x in aryn_ignored_elements]:
                    continue
                prev_text = prev_item.get("text_representation") or ""
                before_table_text = prev_text + "\n" + before_table_text

            # Look forward
            cur_idx = idx
            while len(after_table_text.split()) < 400 and cur_idx < len(elements) - 1:
                cur_idx += 1
                next_item = elements[cur_idx]
                next_type = next_item.get("type", "").lower()
                if next_type in ["table"] + [x.lower() for x in aryn_ignored_elements]:
                    continue
                next_text = next_item.get("text_representation") or ""
                after_table_text += next_text + "\n"

            # Convert table to text
            try:
                df = table_elem_to_dataframe(item)
                table_str = df.to_string(index=False)
            except Exception as e:
                logging.warning(f"Failed converting table to DataFrame at idx={idx}: {e}")
                table_str = ""

            # Possibly run summarizer
            try:
                descriptive_paragraph = table_description_extractor(
                    before_table_text, table_str, after_table_text
                )
            except Exception as e:
                logging.warning(f"table_description_extractor error: {e}")
                descriptive_paragraph = ""

            text_rep = descriptive_paragraph + "\nTableContent:\n" + table_str
            item["text_representation"] = text_rep
        else:
            item["text_representation"] = text_rep

        # Clean the metadata
        metadata = sanitize_metadata(metadata)

        # Construct the Document
        llama_docs.append(Document(text=text_rep, metadata=metadata))

    # Re-save the updated chunked data
    try:
        if json_or_pickle_path.endswith(".json"):
            data["elements"] = elements
            with open(json_or_pickle_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        elif json_or_pickle_path.endswith(".pickle"):
            data["elements"] = elements
            with open(json_or_pickle_path, "wb") as f:
                pickle.dump(data, f)
    except Exception as e:
        logging.error(f"Failed to write updated data back to {json_or_pickle_path}: {e}")

    return llama_docs


def upload_and_index(
    local_file_path: str,
    title: str = "",
    source_type: str = "",
    website_url: str = ""
):
    """
    Main function to:
      1) Chunk document with Aryn
      2) Create/Use Pinecone index
      3) Embed & Upsert
      4) Save logs locally
    """

    # 1) Load logs
    index_logs_data = load_local_logs(LOCAL_INDEX_LOGS_PATH)
    upsert_logs_data = load_local_logs(LOCAL_UPSERT_LOGS_PATH)

    # 2) Check existing Pinecone indexes properly
    existing_indexes_dict = pc.list_indexes()  # returns a dict with "indexes" key
    index_info_list = existing_indexes_dict.get("indexes", [])
    index_names = [idx["name"] for idx in index_info_list]
    logging.debug(f"Index Names from Pinecone: {index_names}")

    # 3) Create the index only if not found
    if INDEX_NAME not in index_names:
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        logging.info(f"Created new Pinecone index: {INDEX_NAME}")
    else:
        logging.info(f"Using existing Pinecone index: {INDEX_NAME}")

    # 4) Get a reference to the existing index
    pinecone_index = pc.Index(INDEX_NAME)

    # 5) Chunk the doc with Aryn
    local_chunk_path = run_aryn_pipeline(
        local_pdf_path=local_file_path,
        output_folder=LOCAL_CHUNK_OUTPUT_DIR,
        aryn_api_key=ARYN_API_KEY
    )
    if not local_chunk_path:
        logging.warning("No chunked file was created. Exiting.")
        return {"message": "No chunked file created. Stopped indexing."}

    #Extract LlamaIndex docs
    documents = extract_llama_index_docs(
        json_or_pickle_path=local_chunk_path,
        source_type=source_type,
        website_url=website_url,
        local_pdf_path=local_file_path
    )
    if not documents:
        logging.warning(f"No documents extracted from {local_chunk_path}. Skipping embedding.")
        return {"message": "No documents extracted. Stopped indexing."}

    #Count tokens
    total_prompt_tokens = 0
    for doc in documents:
        tokens = count_tokens(doc.text, model="gpt-3.5-turbo")
        total_prompt_tokens += tokens
    logging.debug(f"Manual Token Count: {total_prompt_tokens} for file {local_chunk_path}")

    #Embed & Upsert
    try:
        embed_model = OpenAIEmbedding()
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index_obj = VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=storage_context,
            embed_model=embed_model
        )
        time.sleep(3)
    except Exception as e:
        logging.error(f"Embedding failed for {local_chunk_path}: {e}")
        return {"message": f"Embedding failed: {e}"}

    # 9) Update local index logs
    pdf_filename = os.path.basename(local_file_path)
    index_logs_data.append({
        "filename": pdf_filename,
        "time": datetime.now().isoformat(),
        "local_path": local_file_path
    })
    save_local_logs(LOCAL_INDEX_LOGS_PATH, index_logs_data)

    # 10) Update local upsert logs
    chunk_filename = os.path.basename(local_chunk_path)
    upsert_logs_data.append({
        "filename": chunk_filename,
        "time": datetime.now().isoformat(),
        "local_path": local_chunk_path
    })
    save_local_logs(LOCAL_UPSERT_LOGS_PATH, upsert_logs_data)

    #Log token usage
    embed_completion_tokens = 0
    total_tokens = total_prompt_tokens + embed_completion_tokens
    logging.info(
        f"[EMBEDDING] local_chunk_path={local_chunk_path}, "
        f"prompt_tokens={total_prompt_tokens}, "
        f"completion_tokens={embed_completion_tokens}, "
        f"total_tokens={total_tokens}"
    )

    #Persist index locally
    try:
        persist_dir = os.path.join("storage", INDEX_NAME)
        os.makedirs(persist_dir, exist_ok=True)
        index_obj.storage_context.persist(persist_dir=persist_dir)
        logging.info(f"Index persisted at {persist_dir}")
    except Exception as e:
        logging.error(f"Failed to persist index: {e}")

    return {
        "message": f"Document '{pdf_filename}' has been successfully indexed",
        "local_file_path": local_file_path
    }

from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from src.utils.config import (
    INDEX_NAME,
    pc  
)
from src.utils.utils import split_response_and_citations, logging
from src.ml.query.query_workflow import CitationQueryEngineWorkflow

async def query_index_async(query_text: str, session_id: str = "default") -> dict:
    #Pinecone index exists
    index_list_response = pc.list_indexes()
    available_indexes = [idx_info['name'] for idx_info in index_list_response.get('indexes', [])]
    logging.info(f"Available indexes before querying: {available_indexes}")

    if INDEX_NAME not in available_indexes:
        return {"error": f"Index '{INDEX_NAME}' does not exist in Pinecone."}

    pinecone_index = pc.Index(INDEX_NAME)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    embed_model = OpenAIEmbedding()
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )

    #Run the workflow
    workflow = CitationQueryEngineWorkflow(timeout=500)
    result = await workflow.run(
        query=query_text,
        index=index,
        session_id=session_id
    )

    # Extract fields from result
    response_text = result.get("response", "")
    prompt_tokens = result.get("prompt_tokens", 0)
    completion_tokens = result.get("completion_tokens", 0)
    total_tokens = result.get("total_tokens", 0)

    logging.info(
        f"[QUERY] prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, total_tokens={total_tokens}"
    )

    source_nodes = result.get("source_nodes", [])
    chat_history = result.get("chat_history")
    model_used = result.get("model_used", "gpt-4o-mini")

    #Derive citations
    citation_map = {}
    all_citations = []
    for idx, node_with_score in enumerate(source_nodes):
        citation_key = str(idx + 1)
        citation_map[citation_key] = node_with_score.node

        # Collect minimal metadata for each source node
        node_meta = node_with_score.node.metadata or {}
        all_citations.append({
            "text": node_with_score.node.text,
            "type": node_meta.get("type"),
            "author": node_meta.get("author"),
            "date_published": node_meta.get("published_date"),
            "issue_date": node_meta.get("issue_date"),
            "manufacturer": node_meta.get("manufacturer"),
            "origin_type": node_meta.get("origin_type"),
            "bbox": node_meta.get("bbox", []),
            "page_number": node_meta.get("page_number"),
            "title": node_meta.get("title"),
            "s3_URL": node_meta.get("s3_URL")
        })

    logging.info(f"All citations collected: {all_citations}")

    response_text_stripped, citation_numbers = split_response_and_citations(response_text)

    # Build metadata_list for references used in the response
    metadata_list = []
    for citation_list in citation_numbers:
        for cit_num in citation_list:
            node = citation_map.get(str(cit_num))
            if node:
                node_meta = node.metadata or {}
                metadata_list.append({
                    "citation": str(cit_num),
                    "text": node.text,
                    "type": node_meta.get("type"),
                    "bbox": node_meta.get("bbox", []),
                    "page_number": node_meta.get("page_number"),
                    "title": node_meta.get("title"),
                    "origin_type": node_meta.get("origin_type"),
                    "manufacturer": node_meta.get("manufacturer"),
                    "date_published": node_meta.get("published_date"),
                    "author": node_meta.get("author"),
                    "issue_date": node_meta.get("issue_date"),
                    "s3_URL": node_meta.get("s3_URL"),
                })
            else:
                # The model cited something that's not in citation_map
                metadata_list.append({
                    "citation": str(cit_num),
                    "error": "Citation not found"
                })

    return {
        "response": response_text_stripped,
        "citations": metadata_list,       
        "all_citations": all_citations,    # All retrieved nodes
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "chat_history": chat_history,
        "model_used": model_used
    }

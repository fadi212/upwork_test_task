from typing import Union
from collections import defaultdict

from llama_index.core.response_synthesizers import (
    ResponseMode,
    get_response_synthesizer
)
from llama_index.llms.openai import OpenAI
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
    Event
)
from llama_index.core.schema import NodeWithScore
from llama_index.core.prompts import PromptTemplate
from src.ml.query.general_prompt import GENERAL_QA_PROMPT
from src.ml.query.memory import session_chat_memory
from src.utils.utils import logging, format_chat_history, count_tokens

from llama_index.core.base.llms.types import ChatMessage, MessageRole

class RetrieverEvent(Event):
    """Event that includes retrieved nodes."""
    nodes: list[NodeWithScore]

class CitationQueryEngineWorkflow(Workflow):
    def __init__(self, timeout=500):
        super().__init__(timeout=timeout)

    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> Union[RetrieverEvent, None]:
        """Retrieves top-k relevant nodes from the index."""
        query = ev.get("query")
        session_id = ev.get("session_id", "default")

        logging.info(f"Query: {query} (Session: {session_id})")

        # Retrieve chat history from the session memory
        memory = session_chat_memory[session_id]
        messages = memory.get_all()  # returns list of ChatMessage
        history_str = format_chat_history(messages)
        logging.debug(f"Current chat history:\n{history_str}")

        # Store values in context
        await ctx.set("query", query)
        await ctx.set("session_id", session_id)
        await ctx.set("history_str", history_str)

        if ev.index is None:
            raise ValueError("Index is empty, load documents before querying!")

        # Retrieve top-k relevant nodes
        retriever = ev.index.as_retriever(similarity_top_k=5)
        nodes = retriever.retrieve(query)

        logging.info(f"Retrieved {len(nodes)} node(s).")

        # Add the user's query to the chat memory
        memory.put(ChatMessage(role=MessageRole.USER, content=query))

        return RetrieverEvent(nodes=nodes)

    @step
    async def synthesize(self, ctx: Context, ev: RetrieverEvent) -> StopEvent:
        llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
        query = await ctx.get("query")
        session_id = await ctx.get("session_id")
        nodes = ev.nodes
        context_str = self.build_context_str(nodes)

        full_qa_template = GENERAL_QA_PROMPT.format(context_str=context_str, query_str=query)
        refine_template = None  

        qa_prompt_template = PromptTemplate(full_qa_template)
        refine_prompt_template = PromptTemplate(refine_template.format(
            context_msg=context_str, query_str=query, existing_answer="{existing_answer}"
        )) if refine_template else None

        prompt_tokens = count_tokens(full_qa_template)
        logging.debug(f"Manual Prompt Token Count: {prompt_tokens}")

        # Build response synthesizer
        synthesizer = get_response_synthesizer(
            llm=llm,
            text_qa_template=qa_prompt_template,
            refine_template=refine_prompt_template,
            response_mode=ResponseMode.COMPACT,
            use_async=True,
        )
        response = await synthesizer.asynthesize(query=query, nodes=nodes)

        # Extract final text
        try:
            completion_text = response.response
        except AttributeError:
            # Fallback if some other format
            try:
                completion_text = response.generations[0].text
            except Exception:
                completion_text = str(response)

        logging.debug(f"LLM Completion Text: {completion_text}")

        completion_tokens = count_tokens(completion_text)
        total_tokens = prompt_tokens + completion_tokens

        logging.debug(f"Manual Completion Token Count: {completion_tokens}")
        logging.debug(f"Manual Total Token Count: {total_tokens}")

        # Add the assistant's response to chat memory
        memory = session_chat_memory[session_id]
        memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=completion_text))

        # Return the final result
        return StopEvent(result={
            "response": completion_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "source_nodes": nodes,
            "chat_history": format_chat_history(memory.get_all()),
            "model_used": "gpt-4o-mini"
        })

    def build_context_str(self, nodes: list[NodeWithScore]) -> str:
        """Format nodes into a labeled context"""
        source_dict = defaultdict(list)
        for idx, node_with_score in enumerate(nodes):
            source_dict[idx + 1].append(node_with_score.node.text)

        context_str = ""
        for source_num in sorted(source_dict.keys()):
            texts = "\n".join(source_dict[source_num])
            context_str += f"Source {source_num}:\n{texts}\n\n"

        return context_str

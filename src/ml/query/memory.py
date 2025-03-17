from collections import defaultdict
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer

#global chat memory store.
session_chat_memory = defaultdict(
    lambda: ChatMemoryBuffer.from_defaults(token_limit=3000)
)

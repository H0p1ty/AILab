"""
Chatbot with context and memory.
"""
import os
from typing import Union
from document_reading import context_from_file_wo_kss
from ollama import chat
from ollama import ChatResponse

model = 'llama3.2:3b'

# Chat roles
SYSTEM = "system"
USER = "user"
ASSISTANT = "assistant"

DEBUG_READ_FILE_QUERY = "Read the file"

class FileContext:
    file_type: str = None
    context: str = None

    def __init__(self, file_type: str, context: str):
        file_type = file_type
        context = context

class Chatbot:
    """Chat with an LLM using RAG. Keeps chat history in memory."""

    chat_history = None
    model = None
    file_context_list = None
    file_paths = None

    def __init__(self):
        self.chat_history = []
        self.model = model
        self.file_context_list = []     
        self.file_paths = []   

    def _summarize_user_intent(self, query: str) -> str:
        """
        Creates a user message containing the user intent, by summarizing the chat
        history and user query.
        """
        chat_history_str = ""
        for entry in self.chat_history:
            chat_history_str += f"{entry['role']}: {entry['content']}\n"
        messages = [
            {
                "role": SYSTEM,
                "content": (
                    "You're an AI assistant reading the transcript of a conversation "
                    "between a user and an assistant. Given the chat history and "
                    "user's query, infer user real intent."
                    f"Chat history: ```{chat_history_str}```\n"
                    f"User's query: ```{query}```\n"
                ),
            }
        ]
        chat_intent_completion: ChatResponse = chat(
            model=self.model,
            messages=messages,
        )
        user_intent = chat_intent_completion['message']['content']

        return user_intent
    
    def _get_context(self, file_path: str, query: str) -> FileContext:
        if file_path is None:
            print("no file_path")
            return None
        if not (file_path.endswith(".pdf") or file_path.endswith(".txt")):
            print("not a \".txt\" or \".pdf\"")
            return None
        intent = self._summarize_user_intent(query)
        context = None
        try:
            context = context_from_file_wo_kss(file_path)
        except ...:
            print(f"""
Something went wrong during file reading...
""")
        file_type = file_path.split('.')[-1]
        file_context = FileContext(file_type, context)
        file_context.context = context
        file_context.file_type = file_type
        return file_context

    def _rag_system_message_content(self, query: str) -> str:
        if len(self.file_paths) == 0:
            return (
                "You're a helpful assistant.\n"
                "Please answer the user's question."
            )
        else:
            prelude = (
                "You're a helpful assistant.\n"
                "Please answer the user's question using only information you can "
                "find in the context, that was in the files given by the user.\n"
                "If the user's question is unrelated to the information in the "
                "context, say you don't know.\n"
            )
            context_innards_list = []
            intent = self._summarize_user_intent(query)
            for file_path in self.file_paths:
                context = self._get_context(file_path, intent)
                context_innards = f"Context of the \".{context.file_type}\" file: ```{context.context}```\n"
                context_innards_list.append(context_innards)
            return prelude + "\n".join(context_innards_list)
    def _rag(self, query: str) -> str:
        """
        Asks the LLM to answer the user's query with the context provided.
        """
        user_message = {"role": USER, "content": query}
        self.chat_history.append(user_message)
        content = self._rag_system_message_content(query)

        messages = [
            {
                "role": SYSTEM,
                "content": content,
            }
        ]
        messages = messages + self.chat_history

        chat_completion: ChatResponse = chat(
            model=self.model,
            messages=messages,
        )
        response = chat_completion['message']['content']
        assistant_message = {"role": ASSISTANT, "content": response}
        self.chat_history.append(assistant_message)

        return response
    
    def clear_context(self) -> None:
        """
        Clears context
        """
        self.chat_history = []
        self.file_context_list = []
        self.file_paths = []

    def ask(self, query: str) -> str:
        """
        Queries an LLM using RAG.
        """
        response = self._rag(query)
        return response
    
    def add_file(self, file_path: str) -> bool:
        context = self._get_context(file_path, DEBUG_READ_FILE_QUERY)
        if context is None:
            return False
        self.file_paths.append(file_path)
        return True
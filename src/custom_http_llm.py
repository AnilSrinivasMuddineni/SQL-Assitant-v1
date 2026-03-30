import logging
import requests
from typing import Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
from langchain_core.outputs import ChatResult, ChatGeneration

try:
    from pydantic.v1 import Extra
except ImportError:
    from pydantic import Extra

logger = logging.getLogger(__name__)

class CustomHTTPChatLLM(BaseChatModel):
    """
    A custom LangChain ChatModel that translates messages into the exact
    HTTP POST payload expected by the configured Custom endpoint.
    """
    api_base: str
    api_key: str
    model_name: str
    temperature: float = 0.0
    
    class Config:
        extra = Extra.ignore if hasattr(Extra, "ignore") else "ignore"
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "custom_http_chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Translate Langchain messages to the required dict format
        formatted_messages = []
        for msg in messages:
            role = "user"
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, AIMessage):
                role = "assistant"
                
            formatted_messages.append({
                "role": role,
                "content": str(msg.content)
            })

        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", self.temperature)
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        logger.info(f"[CustomHTTPChatLLM] Sending request to {self.api_base} for model {self.model_name}")
        
        try:
            response = requests.post(
                self.api_base,
                json=payload,
                headers=headers,
                timeout=120
            )
            response.raise_for_status()
            response_json = response.json()
            
            content = ""
            if "choices" in response_json and len(response_json["choices"]) > 0:
                message = response_json["choices"][0].get("message", {})
                content = message.get("content", "")
            else:
                logger.warning(f"[CustomHTTPChatLLM] Unexpected response shape: {response_json}")

            ai_message = AIMessage(content=content)
            generation = ChatGeneration(message=ai_message)
            return ChatResult(generations=[generation])

        except Exception as e:
            logger.error(f"[CustomHTTPChatLLM] API call failed: {e}")
            raise e

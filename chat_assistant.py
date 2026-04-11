#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
💬 Chat Assistant - Modo Interativo para Análise de Imagens
===========================================================
Permite conversar sobre uma imagem com streaming de respostas.
"""

import logging
import base64
from pathlib import Path
from typing import Optional, Generator, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO

logger = logging.getLogger(__name__)

# Verificar dependências
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class ChatMessage:
    """Uma mensagem no chat."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: str = ""
    image_data: Optional[str] = None  # base64
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%H:%M:%S")


@dataclass
class ChatSession:
    """Sessão de chat sobre uma imagem."""
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    model: str = ""
    model_type: str = ""  # "ollama" ou "openai"
    messages: List[ChatMessage] = field(default_factory=list)
    context: str = ""  # Contexto inicial (OCR, detecções, etc.)
    
    def add_message(self, role: str, content: str, image_data: str = None):
        """Adiciona mensagem ao histórico."""
        msg = ChatMessage(role=role, content=content, image_data=image_data)
        self.messages.append(msg)
        return msg
    
    def get_history_for_api(self) -> List[Dict]:
        """Retorna histórico formatado para API."""
        history = []
        
        for msg in self.messages:
            if msg.role == "system":
                history.append({"role": "system", "content": msg.content})
            elif msg.role == "user":
                if msg.image_data:
                    history.append({
                        "role": "user",
                        "content": msg.content,
                        "images": [msg.image_data]
                    })
                else:
                    history.append({"role": "user", "content": msg.content})
            else:
                history.append({"role": "assistant", "content": msg.content})
        
        return history
    
    def clear(self):
        """Limpa o histórico de mensagens."""
        self.messages = []


class ChatAssistant:
    """Assistente de chat para análise de imagens."""
    
    def __init__(self, model: str, model_type: str = "ollama"):
        """
        Inicializa o assistente.
        
        Args:
            model: Nome do modelo (ex: "gemma3:12b-it-q8_0")
            model_type: Tipo do modelo ("ollama" ou "openai")
        """
        self.model = model
        self.model_type = model_type
        self.session: Optional[ChatSession] = None
    
    def start_session(
        self, 
        image_base64: str, 
        image_path: str = "",
        initial_context: str = ""
    ) -> ChatSession:
        """
        Inicia uma nova sessão de chat com uma imagem.
        
        Args:
            image_base64: Imagem em base64
            image_path: Caminho da imagem (opcional)
            initial_context: Contexto inicial (OCR, detecções, etc.)
        """
        self.session = ChatSession(
            image_path=image_path,
            image_base64=image_base64,
            model=self.model,
            model_type=self.model_type,
            context=initial_context
        )
        
        # Adicionar mensagem de sistema se houver contexto
        if initial_context:
            system_msg = f"""Você é um assistente especializado em análise de imagens.
O usuário vai fazer perguntas sobre uma imagem.

CONTEXTO PRÉ-ANALISADO:
{initial_context}

Responda de forma precisa e objetiva, baseando-se no que você pode ver na imagem.
Se não tiver certeza de algo, indique isso claramente."""
            self.session.add_message("system", system_msg)
        
        return self.session
    
    def chat(self, user_message: str, include_image: bool = True) -> str:
        """
        Envia mensagem e retorna resposta completa.
        
        Args:
            user_message: Pergunta do usuário
            include_image: Se deve incluir a imagem na mensagem
            
        Returns:
            Resposta do modelo
        """
        if not self.session:
            raise RuntimeError("Nenhuma sessão iniciada. Use start_session() primeiro.")
        
        # Adicionar mensagem do usuário
        image_data = self.session.image_base64 if include_image and len(self.session.messages) <= 1 else None
        self.session.add_message("user", user_message, image_data)
        
        # Chamar API
        if self.model_type == "ollama":
            response = self._chat_ollama(user_message, include_image)
        else:
            response = self._chat_openai(user_message, include_image)
        
        # Adicionar resposta ao histórico
        self.session.add_message("assistant", response)
        
        return response
    
    def chat_stream(self, user_message: str, include_image: bool = True) -> Generator[str, None, None]:
        """
        Envia mensagem e retorna resposta em streaming.
        
        Args:
            user_message: Pergunta do usuário
            include_image: Se deve incluir a imagem
            
        Yields:
            Chunks da resposta
        """
        if not self.session:
            raise RuntimeError("Nenhuma sessão iniciada. Use start_session() primeiro.")
        
        # Adicionar mensagem do usuário
        image_data = self.session.image_base64 if include_image and len(self.session.messages) <= 1 else None
        self.session.add_message("user", user_message, image_data)
        
        # Chamar API com streaming
        full_response = ""
        
        if self.model_type == "ollama":
            for chunk in self._chat_ollama_stream(user_message, include_image):
                full_response += chunk
                yield chunk
        else:
            for chunk in self._chat_openai_stream(user_message, include_image):
                full_response += chunk
                yield chunk
        
        # Adicionar resposta completa ao histórico
        self.session.add_message("assistant", full_response)
    
    def _chat_ollama(self, user_message: str, include_image: bool) -> str:
        """Chat com Ollama (sem streaming)."""
        if not OLLAMA_AVAILABLE:
            return "[Erro: Ollama não disponível]"
        
        messages = self._build_ollama_messages(user_message, include_image)
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={"temperature": 0.7, "num_ctx": 4096}
            )
            return response['message']['content']
        except Exception as e:
            return f"[Erro Ollama: {e}]"
    
    def _chat_ollama_stream(self, user_message: str, include_image: bool) -> Generator[str, None, None]:
        """Chat com Ollama (streaming)."""
        if not OLLAMA_AVAILABLE:
            yield "[Erro: Ollama não disponível]"
            return
        
        messages = self._build_ollama_messages(user_message, include_image)
        
        try:
            stream = ollama.chat(
                model=self.model,
                messages=messages,
                options={"temperature": 0.7, "num_ctx": 4096},
                stream=True
            )
            
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
                    
        except Exception as e:
            yield f"[Erro Ollama: {e}]"
    
    def _build_ollama_messages(self, user_message: str, include_image: bool) -> List[Dict]:
        """Constrói mensagens para Ollama."""
        messages = []
        
        # Adicionar contexto do sistema
        for msg in self.session.messages:
            if msg.role == "system":
                messages.append({"role": "system", "content": msg.content})
        
        # Adicionar histórico (sem a última mensagem do usuário que acabamos de adicionar)
        for msg in self.session.messages[:-1]:
            if msg.role == "user":
                m = {"role": "user", "content": msg.content}
                if msg.image_data:
                    m["images"] = [msg.image_data]
                messages.append(m)
            elif msg.role == "assistant":
                messages.append({"role": "assistant", "content": msg.content})
        
        # Adicionar mensagem atual
        current_msg = {"role": "user", "content": user_message}
        if include_image and self.session.image_base64 and len([m for m in self.session.messages if m.role == "user"]) <= 1:
            current_msg["images"] = [self.session.image_base64]
        messages.append(current_msg)
        
        return messages
    
    def _chat_openai(self, user_message: str, include_image: bool) -> str:
        """Chat com OpenAI (sem streaming)."""
        if not OPENAI_AVAILABLE:
            return "[Erro: OpenAI não disponível]"
        
        import os
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        messages = self._build_openai_messages(user_message, include_image)
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[Erro OpenAI: {e}]"
    
    def _chat_openai_stream(self, user_message: str, include_image: bool) -> Generator[str, None, None]:
        """Chat com OpenAI (streaming)."""
        if not OPENAI_AVAILABLE:
            yield "[Erro: OpenAI não disponível]"
            return
        
        import os
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        messages = self._build_openai_messages(user_message, include_image)
        
        try:
            stream = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            yield f"[Erro OpenAI: {e}]"
    
    def _build_openai_messages(self, user_message: str, include_image: bool) -> List[Dict]:
        """Constrói mensagens para OpenAI."""
        messages = []
        
        # Sistema
        for msg in self.session.messages:
            if msg.role == "system":
                messages.append({"role": "system", "content": msg.content})
        
        # Histórico
        for msg in self.session.messages[:-1]:
            if msg.role == "user":
                if msg.image_data:
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": msg.content},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{msg.image_data}"}}
                        ]
                    })
                else:
                    messages.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                messages.append({"role": "assistant", "content": msg.content})
        
        # Mensagem atual
        if include_image and self.session.image_base64 and len([m for m in self.session.messages if m.role == "user"]) <= 1:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.session.image_base64}"}}
                ]
            })
        else:
            messages.append({"role": "user", "content": user_message})
        
        return messages


def create_assistant(model: str, model_type: str = "ollama") -> ChatAssistant:
    """Cria um novo assistente de chat."""
    return ChatAssistant(model, model_type)

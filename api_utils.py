#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔐 API Utils - Validação, Rate Limiting e Retry
===============================================
Utilitários para chamadas de API robustas.
"""

import time
import logging
import functools
from typing import Optional, Callable, Any, Dict
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import Lock
import random

logger = logging.getLogger(__name__)


# ============================================================================
# RETRY COM BACKOFF EXPONENCIAL
# ============================================================================

# Marcadores de erro permanente: repetir uma chave inválida ou um payload malformado
# só queima tempo (5 tentativas com backoff chegam a ~2 minutos) e nunca vai passar.
_PERMANENT_ERROR_MARKERS = (
    "invalid_api_key",
    "incorrect api key",
    "authentication",
    "unauthorized",
    "permission_denied",
    "invalid_request_error",
    "model_not_found",
    "does not exist",
    "content_policy",
    "billing",
    "insufficient_quota",
)

_PERMANENT_STATUS_CODES = {400, 401, 403, 404, 405, 422}


def is_transient_error(exc: BaseException) -> bool:
    """Decide se vale repetir a chamada.

    Erros de rede, timeout, 429 e 5xx são transitórios. Erros de autenticação,
    payload inválido e modelo inexistente são permanentes — repetir não ajuda.
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if isinstance(status, int):
        if status == 429 or status >= 500:
            return True
        if status in _PERMANENT_STATUS_CODES:
            return False

    message = str(exc).lower()
    if any(marker in message for marker in _PERMANENT_ERROR_MARKERS):
        return False

    return True


@dataclass
class RetryConfig:
    """Configuração de retry."""
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_exceptions: tuple = (Exception,)
    # Filtro adicional aplicado depois de retry_exceptions. Retornar False aborta na hora.
    should_retry: Optional[Callable[[BaseException], bool]] = None


def retry_with_backoff(config: RetryConfig = None):
    """
    Decorator para retry com backoff exponencial.
    
    Uso:
        @retry_with_backoff(RetryConfig(max_retries=3))
        def minha_funcao():
            ...
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except config.retry_exceptions as e:
                    last_exception = e

                    if config.should_retry is not None and not config.should_retry(e):
                        logger.error(
                            f"❌ {func.__name__}: erro permanente, sem retry. {str(e)[:120]}"
                        )
                        raise

                    if attempt == config.max_retries:
                        logger.error(f"❌ {func.__name__}: Falhou após {config.max_retries} tentativas")
                        raise
                    
                    # Calcular delay
                    delay = min(
                        config.initial_delay * (config.exponential_base ** attempt),
                        config.max_delay
                    )
                    
                    # Adicionar jitter
                    if config.jitter:
                        delay = delay * (0.5 + random.random())
                    
                    logger.warning(
                        f"⚠️ {func.__name__}: Tentativa {attempt + 1}/{config.max_retries} falhou. "
                        f"Erro: {str(e)[:50]}... Retry em {delay:.1f}s"
                    )
                    
                    time.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator


# ============================================================================
# RATE LIMITING
# ============================================================================

@dataclass
class RateLimitConfig:
    """Configuração de rate limiting."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    min_interval: float = 0.1  # Intervalo mínimo entre requests


class RateLimiter:
    """
    Rate limiter thread-safe para APIs.
    
    Uso:
        limiter = RateLimiter(RateLimitConfig(requests_per_minute=30))
        limiter.wait()  # Espera se necessário
        fazer_request()
    """
    
    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        self._lock = Lock()
        self._requests: list = []
        self._last_request_time: float = 0
    
    def _cleanup_old_requests(self):
        """Remove requests antigos da lista."""
        now = time.time()
        hour_ago = now - 3600
        self._requests = [t for t in self._requests if t > hour_ago]
    
    def _count_recent_requests(self, seconds: int) -> int:
        """Conta requests nos últimos N segundos."""
        cutoff = time.time() - seconds
        return sum(1 for t in self._requests if t > cutoff)
    
    def _can_proceed_locked(self) -> tuple[bool, float]:
        """Lógica de can_proceed. Exige que o lock já esteja adquirido."""
        self._cleanup_old_requests()
        now = time.time()

        # Verificar intervalo mínimo
        time_since_last = now - self._last_request_time
        if time_since_last < self.config.min_interval:
            return False, self.config.min_interval - time_since_last

        # Verificar limite por minuto
        recent = [t for t in self._requests if t > now - 60]
        if len(recent) >= self.config.requests_per_minute:
            wait_time = 60 - (now - min(recent)) + 0.1
            return False, wait_time

        # Verificar limite por hora
        if len(self._requests) >= self.config.requests_per_hour:
            wait_time = 3600 - (now - min(self._requests)) + 0.1
            return False, wait_time

        return True, 0.0

    def can_proceed(self) -> tuple[bool, float]:
        """
        Verifica se pode fazer um request.
        Retorna: (pode_prosseguir, tempo_de_espera)
        """
        with self._lock:
            return self._can_proceed_locked()
    
    def _try_acquire(self) -> tuple[bool, float]:
        """Checa o limite e registra o request na MESMA seção crítica.

        Separar a checagem do registro (can_proceed + append) abre uma janela em que
        duas threads passam juntas e estouram o limite.
        """
        with self._lock:
            can_go, wait_time = self._can_proceed_locked()
            if can_go:
                now = time.time()
                self._requests.append(now)
                self._last_request_time = now
            return can_go, wait_time

    def wait(self) -> float:
        """
        Espera até poder fazer um request.
        Retorna o tempo esperado.
        """
        total_waited = 0.0

        while True:
            can_go, wait_time = self._try_acquire()

            if can_go:
                return total_waited

            logger.debug(f"⏳ Rate limit: aguardando {wait_time:.1f}s")
            time.sleep(wait_time)
            total_waited += wait_time
    
    def record_request(self):
        """Registra um request feito."""
        with self._lock:
            now = time.time()
            self._requests.append(now)
            self._last_request_time = now
    
    def get_status(self) -> Dict[str, Any]:
        """Retorna status atual do rate limiter."""
        with self._lock:
            self._cleanup_old_requests()
            now = time.time()
            
            return {
                "requests_last_minute": self._count_recent_requests(60),
                "requests_last_hour": len(self._requests),
                "limit_per_minute": self.config.requests_per_minute,
                "limit_per_hour": self.config.requests_per_hour,
                "time_since_last": now - self._last_request_time if self._last_request_time else None
            }


# ============================================================================
# TIMEOUT WRAPPER
# ============================================================================

class APITimeoutError(Exception):
    """Erro de timeout de API."""
    pass


# ============================================================================
# VALIDAÇÃO DE INPUT
# ============================================================================

@dataclass
class ValidationConfig:
    """Configuração de validação de arquivos."""
    max_file_size_mb: float = 20.0
    min_image_dimension: int = 10
    max_image_dimension: int = 10000
    allowed_extensions: set = None
    
    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = {
                '.jpg', '.jpeg', '.png', '.gif', '.bmp',
                '.tiff', '.tif', '.webp', '.heic', '.heif',
                '.avif', '.ico', '.raw', '.cr2', '.nef', '.arw', '.orf', '.rw2'
            }


class InputValidator:
    """Validador de inputs."""
    
    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
    
    def validate_file(self, filepath) -> tuple[bool, str]:
        """
        Valida um arquivo de imagem.
        Retorna: (é_válido, mensagem)
        """
        from pathlib import Path
        filepath = Path(filepath)
        
        # Verificar existência
        if not filepath.exists():
            return False, f"Arquivo não encontrado: {filepath}"
        
        # Verificar extensão
        ext = filepath.suffix.lower()
        if ext not in self.config.allowed_extensions:
            return False, f"Extensão não suportada: {ext}"
        
        # Verificar tamanho
        size_mb = filepath.stat().st_size / (1024 * 1024)
        if size_mb > self.config.max_file_size_mb:
            return False, f"Arquivo muito grande: {size_mb:.1f}MB (máx: {self.config.max_file_size_mb}MB)"
        
        return True, "OK"
    
    def validate_image(self, image) -> tuple[bool, str]:
        """
        Valida uma imagem PIL.
        Retorna: (é_válido, mensagem)
        """
        try:
            width, height = image.size
            
            # Verificar dimensões mínimas
            if width < self.config.min_image_dimension or height < self.config.min_image_dimension:
                return False, f"Imagem muito pequena: {width}x{height}px"
            
            # Verificar dimensões máximas
            if width > self.config.max_image_dimension or height > self.config.max_image_dimension:
                return False, f"Imagem muito grande: {width}x{height}px"
            
            return True, "OK"
            
        except Exception as e:
            return False, f"Erro ao validar imagem: {e}"
    
    def sanitize_filename(self, filename: str) -> str:
        """Remove caracteres inválidos de nome de arquivo."""
        import re
        
        # Remover caracteres inválidos
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Remover caracteres de controle
        filename = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', filename)
        
        # Limitar tamanho
        if len(filename) > 200:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = name[:195] + ('.' + ext if ext else '')
        
        return filename.strip()


# ============================================================================
# INSTÂNCIAS GLOBAIS
# ============================================================================

# Rate limiters para diferentes APIs
_openai_limiter: Optional[RateLimiter] = None
_ollama_limiter: Optional[RateLimiter] = None


def get_openai_limiter() -> RateLimiter:
    """Retorna rate limiter para OpenAI."""
    global _openai_limiter
    if _openai_limiter is None:
        _openai_limiter = RateLimiter(RateLimitConfig(
            requests_per_minute=50,  # Tier 1 limit
            requests_per_hour=500,
            min_interval=0.5
        ))
    return _openai_limiter


def get_ollama_limiter() -> RateLimiter:
    """Retorna rate limiter para Ollama (local)."""
    global _ollama_limiter
    if _ollama_limiter is None:
        _ollama_limiter = RateLimiter(RateLimitConfig(
            requests_per_minute=120,  # Mais permissivo para local
            requests_per_hour=5000,
            min_interval=0.1
        ))
    return _ollama_limiter


# Configurações de retry padrão
DEFAULT_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    initial_delay=1.0,
    max_delay=30.0,
    exponential_base=2.0,
    jitter=True
)

API_RETRY_CONFIG = RetryConfig(
    max_retries=5,
    initial_delay=2.0,
    max_delay=60.0,
    exponential_base=2.0,
    jitter=True,
    retry_exceptions=(ConnectionError, APITimeoutError, Exception),
    should_retry=is_transient_error,
)

from dataclasses import dataclass, field
import os
import logging
from typing import (
    TYPE_CHECKING,
    Optional,
)
from openai import OpenAI

from .parser import FlexibleArgumentParser

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    model_name: str = "localhost"
    max_tokens: int = 2048
    temperature: float = 1.0
    stop: list[str] = field(default_factory=list)
    top_p: float = 1.0
    min_p: float = 0.0
    echo: bool = False
    port: int = 8000
    logprobs: int = 0
    tokenizer: Optional[str] = None

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        parser.add_argument(
            "--max-tokens",
            type=int,
            help=f"The number of maximum tokens to generate (default: {LLMConfig.max_tokens})",
            default=LLMConfig.max_tokens,
        )
        parser.add_argument(
            "--temperature",
            type=float,
            help=f"Decoding temperature (default: {LLMConfig.temperature})",
            default=LLMConfig.temperature,
        )
        parser.add_argument(
            "--model-name",
            "-m",
            type=str,
            help=f"model name or localhost if local serving (default: {LLMConfig.model_name})",
            default=LLMConfig.model_name,
        )
        parser.add_argument(
            "--top-p",
            type=float,
            help=f"Nucleus sampling parameter (default: {LLMConfig.top_p})",
            default=LLMConfig.top_p,
        )
        parser.add_argument(
            "--port",
            type=int,
            help=f"localhost port number (default: {LLMConfig.port})",
            default=LLMConfig.port,
        )
        parser.add_argument("--echo", type=bool, default=LLMConfig.echo)
        parser.add_argument(
            "--logprobs",
            type=int,
            help=f"The number of logprobs to present (default: {LLMConfig.logprobs})",
            default=LLMConfig.logprobs,
        )
        parser.add_argument(
            "--tokenizer",
            type=str,
            help=f"Tokenizer to use for the model (default: {LLMConfig.tokenizer})",
            default=LLMConfig.tokenizer,    
        )
        return parser

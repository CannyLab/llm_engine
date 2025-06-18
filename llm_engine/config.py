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
    need_tokenizer: bool = False
    tokenizer: Optional[str] = None
    is_instruct: bool = False
    is_reasoning: bool = False
    num_gpus: int = -1

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
            "--need-tokenizer",
            type=bool,
            help=f"Whether the model needs a tokenizer (default: {LLMConfig.need_tokenizer})",
            default=LLMConfig.need_tokenizer,
        )
        parser.add_argument(
            "--tokenizer",
            type=str,
            help=f"Tokenizer to use for the model (default: {LLMConfig.tokenizer})",
            default=LLMConfig.tokenizer,
        )
        parser.add_argument(
            "--is-instruct",
            type=bool,
            help=f"Whether the model is a chat model (instruction following model). This is used for custom LLM models that are not listed in models.yaml (default: {LLMConfig.is_instruct})",
            default=LLMConfig.is_instruct,
        )
        parser.add_argument(
            "--is-reasoning",
            type=bool,
            help=f"Whether the model is a reasoning model. This is used for custom LLM models that are not listed in models.yaml (default: {LLMConfig.is_reasoning})",
            default=LLMConfig.is_reasoning,
        )
        parser.add_argument(
            "--num-gpus",
            type=int,
            help=f"Number of GPUs to use (default: {LLMConfig.num_gpus})",
            default=LLMConfig.num_gpus,
        )
        return parser

    @classmethod
    def from_args(cls, args):
        return cls(
            **{k: v for k, v in vars(args).items() if k in cls.__dataclass_fields__}
        )

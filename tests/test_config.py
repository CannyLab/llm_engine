import sys
import types
from pathlib import Path

# Add repository root to sys.path to allow local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_add_cli_args_parses_values(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "openai",
        types.SimpleNamespace(
            AuthenticationError=Exception, BadRequestError=Exception, OpenAI=object
        ),
    )
    monkeypatch.setitem(
        sys.modules, "yaml", types.SimpleNamespace(safe_load=lambda f: {})
    )


from llm_engine.config import LLMConfig, FlexibleArgumentParser


def test_add_cli_args_parses_values(monkeypatch):
    parser = FlexibleArgumentParser()
    LLMConfig.add_cli_args(parser)
    args = parser.parse_args(["--max-tokens", "42", "--temperature", "0.5"])
    assert args.max_tokens == 42
    assert args.temperature == 0.5

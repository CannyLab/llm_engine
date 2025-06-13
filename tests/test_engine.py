import sys
import types
from pathlib import Path

# Add repository root to sys.path to allow local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Provide stub modules used by engine
sys.modules["openai"] = types.SimpleNamespace(
    AuthenticationError=Exception,
    BadRequestError=Exception,
    OpenAI=object,
)
sys.modules["vllm"] = types.SimpleNamespace(LLM=lambda *args, **kwargs: object())
sys.modules["transformers"] = types.SimpleNamespace(
    AutoTokenizer=types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: object()
    )
)
sys.modules["tqdm"] = types.SimpleNamespace(
    tqdm=lambda iterable=None, **kwargs: iterable
)
sys.modules["yaml"] = types.SimpleNamespace(
    safe_load=lambda f: {
        "models": [
            {
                "model_name": "dummy",
                "api_provider": "dummy",
                "is_instruct": True,
                "is_reasoning": False,
            }
        ]
    }
)

from llm_engine.engine import LLMEngine, DummyClient
from llm_engine.config import LLMConfig


class DummyTextClient:
    def __init__(self, text="text"):
        self._text = text
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        class Choice:
            def __init__(self, text):
                self.text = text

        class Response:
            def __init__(self, text):
                self.choices = [Choice(text)]

        return Response(self._text)


def test_prompt_llm_auto_returns_text_by_default_chat():
    cfg = LLMConfig(model_name="dummy")
    engine = LLMEngine(cfg)
    engine.client = DummyClient("hello")
    assert engine.prompt_llm_auto("hi") == "hello"


def test_prompt_llm_auto_return_raw_when_requested():
    cfg = LLMConfig(model_name="dummy")
    engine = LLMEngine(cfg)
    engine.client = DummyClient("hi")
    res = engine.prompt_llm_auto("hi", return_raw=True)
    assert hasattr(res, "choices")


def test_prompt_llm_auto_handles_completion_text():
    cfg = LLMConfig(model_name="dummy")
    engine = LLMEngine(cfg)
    engine.is_instruct = False
    engine.client = DummyTextClient("done")
    assert engine.prompt_llm_auto("prompt") == "done"

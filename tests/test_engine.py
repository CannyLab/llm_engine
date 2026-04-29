import sys
import types
from pathlib import Path
from unittest.mock import patch

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


class RecordingChatClient:
    def __init__(self):
        self.chat = self
        self.calls = []

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return DummyClient("ok").create(**kwargs)


class RecordingVLLMClient:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def stub_load_models(self):
    self.model_list = []


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


def test_prompt_llm_reasoning_uses_reasoning_effort_for_openai_models():
    cfg = LLMConfig(model_name="dummy")
    engine = LLMEngine(cfg)
    engine.is_reasoning = True
    engine._api_provider = "openai"
    engine.client = RecordingChatClient()

    engine.prompt_llm_reasoning("hi")

    assert engine.client.calls[0]["reasoning_effort"] == "none"
    assert "extra_body" not in engine.client.calls[0]


def test_prompt_llm_reasoning_with_messages_uses_reasoning_effort_for_google_models():
    cfg = LLMConfig(model_name="dummy")
    engine = LLMEngine(cfg)
    engine.is_reasoning = True
    engine._api_provider = "google"
    engine.client = RecordingChatClient()

    engine.prompt_llm_reasoning_with_messages([{"role": "user", "content": "hi"}])

    assert engine.client.calls[0]["reasoning_effort"] == "none"
    assert "extra_body" not in engine.client.calls[0]


def test_prompt_llm_reasoning_uses_chat_template_kwargs_for_localhost_models():
    cfg = LLMConfig(model_name="dummy")
    engine = LLMEngine(cfg)
    engine.is_reasoning = True
    engine._api_provider = "localhost"
    engine.client = RecordingChatClient()

    engine.prompt_llm_reasoning("hi")

    assert engine.client.calls[0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_prompt_llm_reasoning_omits_disable_thinking_when_config_disabled():
    cfg = LLMConfig(model_name="dummy", disable_thinking=False)
    engine = LLMEngine(cfg)
    engine.is_reasoning = True
    engine.client = RecordingChatClient()

    engine.prompt_llm_reasoning("hi")

    assert "extra_body" not in engine.client.calls[0]


def test_offline_vllm_uses_all_visible_gpus_when_num_gpus_is_unset():
    cfg = LLMConfig(model_name="local/test-model", num_gpus=-1)

    sys.modules["torch"] = types.SimpleNamespace(
        cuda=types.SimpleNamespace(device_count=lambda: 4)
    )
    with patch("llm_engine.engine.LLMEngine._load_models", new=stub_load_models):
        sys.modules["vllm"] = types.SimpleNamespace(LLM=RecordingVLLMClient)
        engine = LLMEngine(cfg)

    assert engine.hosted_vllm is True
    assert engine.client.kwargs["tensor_parallel_size"] == 4


def test_offline_vllm_respects_explicit_cpu_override():
    cfg = LLMConfig(model_name="local/test-model", num_gpus=0)

    with patch("llm_engine.engine.LLMEngine._load_models", new=stub_load_models):
        sys.modules["vllm"] = types.SimpleNamespace(LLM=RecordingVLLMClient)
        engine = LLMEngine(cfg)

    assert engine.hosted_vllm is True
    assert "tensor_parallel_size" not in engine.client.kwargs

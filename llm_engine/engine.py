import logging
from multiprocessing import pool
from time import sleep  # Import sleep from the time module
import random
import os
import yaml
from pathlib import Path
from typing import Any, Callable, Collection, Optional, Type, TypeVar

from openai import AuthenticationError, BadRequestError, OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

from .config import LLMConfig

Q = TypeVar("Q", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


class DummyClient:
    @property
    def completions(self):
        return self

    def __init__(self, dummy_message: str = "(A) debug") -> None:
        self.dummy_message = dummy_message
        self.chat = self

    def create(self, **kwargs):
        dummy_message = self.dummy_message

        class DummyChoice:
            def __init__(self):
                self.message = type("DummyMessage", (), {"content": dummy_message})()
                self.logprobs = None

        class DummyResponse:
            def __init__(self):
                self.choices = [DummyChoice()]

        return DummyResponse()


class LLMEngine:
    def __init__(self, llm_config: LLMConfig) -> None:
        self._config = llm_config
        self._load_models()
        self.prepare_llm(
            model_name=llm_config.model_name,
            port=llm_config.port,
        )
        
        logger.info(f"Initialized LLM Engine with model: {self._model_name_str}")
        logger.info(f"API Provider: {self._api_provider}")
        logger.info(f"is_instruct: {self._is_instruct}")
        logger.info(f"is_reasoning: {self._is_reasoning}")

    def _load_models(self) -> None:
        models_path = Path(__file__).parent.parent / "models" / "models.yaml"
        try:
            with open(models_path, 'r') as f:
                models = yaml.safe_load(f)
                self.model_list = models["models"]
        except Exception as e:
            logger.error(f"Error loading models.yaml: {e}")
            self.model_list = []

    def prepare_llm(self, model_name: str, port: int) -> None:
        if "localhost" == model_name:
            self.client = OpenAI(
                api_key="EMPTY",
                base_url=f"http://localhost:{port}/v1",  # base_url=f"http://127.0.0.1:{port}/v1",
            )
            # get model name
            models = self.client.models.list()
            model = next(iter(models)).id
            self._model_name_str = model.split("/")[-1]

            llm = next((llm for llm in self.model_list if llm["model_name"] == model), None)
            self._model_name = model
            self._api_provider = "localhost"
            if llm is None:
                logger.error(f"Unrecognized local model: {model}")
                self._is_instruct = False
                self._is_reasoning = False
            else:
                self._is_instruct = llm["is_instruct"]
                self._is_reasoning = llm["is_reasoning"]
        else:
            # get match in LLMS
            llm = next((llm for llm in self.model_list if llm["model_name"] == model_name), None)
            if llm is None:
                raise ValueError(f"Invalid model name: {model_name}")
            model = llm["model_name"]
            self._model_name = model
            self._model_name_str = model.split("/")[-1]
            self._api_provider = llm["api_provider"]
            self._is_instruct = llm["is_instruct"]
            self._is_reasoning = llm["is_reasoning"]
            if self._api_provider == "openai":  # openai
                self.client = OpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    base_url="https://api.openai.com/v1",
                )

                if self._config.stop:
                    if len(self._config.stop) > 4:
                        self._config.stop = self._config.stop[:4]

            elif self._api_provider == "deepseek":  # deepseek
                self.client = OpenAI(
                    api_key=os.environ.get("DEEPSEEK_API_KEY"),
                    base_url="https://api.deepseek.com",
                )

            elif self._api_provider == "google":  # google
                self.client = OpenAI(
                    api_key=os.environ.get("GOOGLE_API_KEY"),
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                )

            elif self._api_provider == "dummy":  # dummy
                self.client = DummyClient()

            else:
                raise ValueError(f"Invalid API provider: {self._api_provider}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model)
        except Exception as e:
            logger.error(f"Could not load tokenizer for model: {model}")
            # Default to Llama-3.1 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.1-70B-Instruct"
            )

    def reinitialize(self) -> None:
        self.prepare_llm(
            model_name=self._model_name,
            port=self._config.port,
        )

    def update_config(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self._config, key, value)
            # for each key, value in kwargs
            logging.info(f"Updated config: {key} = {value}")

    # define a retry decorator
    def retry_with_exponential_backoff(
        initial_delay: float = 1,
        exponential_base: float = 2,
        jitter: bool = True,
        max_retries: int = 10,
        no_retry_on: Optional[Collection[Type[Exception]]] = None,
    ) -> Callable[[Q], Q]:
        """Retry a function with exponential backoff."""

        def decorator(func: Q) -> Q:
            def wrapper(*args, **kwargs):
                # Initialize variables
                num_retries = 0
                delay = initial_delay
                error = None

                # Loop until a successful response or max_retries is hit or an exception is raised
                while num_retries <= max_retries:
                    try:
                        return func(*args, **kwargs)
                    # Raise exceptions for any errors specified
                    except Exception as e:
                        if no_retry_on is not None and type(e) in no_retry_on:
                            raise e
                        # Sleep for the delay
                        sleep(delay)
                        # Increment the delay
                        delay *= exponential_base * (1 + jitter * random.random())
                        # Set the error to the last exception
                        error = e
                        # Increment retries
                        num_retries += 1
                        logger.warning(
                            f"Retrying {func.__name__} after error: {e} (retry {num_retries} of {max_retries})"
                        )
                if error is not None:
                    raise error

            return wrapper

        return decorator

    def prompt_llm_auto(
        self,
        model_prompt,
        system_prompt: str = None,
        n=1,
    ):
        try:
            if self._is_reasoning:
                assert isinstance(model_prompt, str)
                return self.prompt_llm_reasoning_auto(
                    model_prompt=model_prompt, system_prompt=system_prompt, n=n
                )
            if not self._is_instruct:
                assert isinstance(model_prompt, str)
                return self.prompt_llm(prompt=model_prompt, n=n)
            else:
                if isinstance(model_prompt, str) and system_prompt is not None:
                    return self.prompt_llm_chat(
                        system_prompt=system_prompt, user_prompt=model_prompt, n=n
                    )
                elif isinstance(model_prompt, list):
                    return self.prompt_llm_chat_with_messages(
                        messages=model_prompt, n=n
                    )
                else:
                    messages = [
                        {"role": "user", "content": model_prompt},
                    ]
                    return self.prompt_llm_chat_with_messages(messages=messages, n=n)
        except BadRequestError as e:
            logger.error(f"Error: {e}")
            return None

    @retry_with_exponential_backoff(
        max_retries=20,
        no_retry_on=(AuthenticationError, BadRequestError),
    )
    def prompt_llm(
        self,
        prompt: str,
        n: int = 1,
    ):
        assert n >= 1
        model_name = self._model_name
        max_tokens = self._config.max_tokens
        temperature = self._config.temperature
        stop = self._config.stop
        logprobs = self._config.logprobs
        top_p = self._config.top_p
        min_p = self._config.min_p
        echo = self._config.echo

        if logprobs > 0:
            return self.client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                logprobs=logprobs,
                top_p=top_p,
                echo=echo,
                n=n,
                top_logprobs=logprobs,
                extra_headers={"min_p": f"{min_p}"},
            )
        else:
            return self.client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                top_p=top_p,
                echo=echo,
                n=n,
                extra_headers={"min_p": f"{min_p}"},
            )

    @retry_with_exponential_backoff(
        max_retries=20,
        no_retry_on=(AuthenticationError, BadRequestError),
    )
    def prompt_llm_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        n=1,
    ):
        assert n >= 1
        model_name = self._model_name
        max_tokens = self._config.max_tokens
        temperature = self._config.temperature
        stop = self._config.stop
        top_p = self._config.top_p
        logprobs = self._config.logprobs
        min_p = self._config.min_p

        if logprobs > 0:
            return self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
                logprobs=logprobs,
                top_logprobs=logprobs,
                extra_headers={"min_p": f"{min_p}"},
            )
        else:
            return self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
                extra_headers={"min_p": f"{min_p}"},
            )

    @retry_with_exponential_backoff(
        max_retries=20,
        no_retry_on=(AuthenticationError, BadRequestError),
    )
    def prompt_llm_reasoning_auto(
        self,
        model_prompt: str,
        system_prompt: str = None,
        n=1,
    ):
        assert n >= 1
        model_name = self._model_name

        return self.client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": model_prompt},
            ],
            n=n,
        )

    @retry_with_exponential_backoff(
        max_retries=20,
        no_retry_on=(AuthenticationError, BadRequestError),
    )
    def prompt_llm_chat_with_messages(
        self,
        messages: list,
        n=1,
    ):
        assert n >= 1
        model_name = self._model_name
        max_tokens = self._config.max_tokens
        temperature = self._config.temperature
        stop = self._config.stop
        top_p = self._config.top_p
        logprobs = self._config.logprobs
        min_p = self._config.min_p
        if logprobs > 0:
            return self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
                logprobs=logprobs,
                top_logprobs=logprobs,
                extra_headers={"min_p": f"{min_p}"},
            )
        else:
            return self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
                extra_headers={"min_p": f"{min_p}"},
            )

    def parallel_generator(
        self,
        f,
        input_list: list[Any],
        output_dict: dict,
        n: int,
        parallel_case_ids: list[int],
        num_processes: int = 4,
    ):
        with pool.ThreadPool(num_processes) as p:
            list(
                tqdm(
                    p.imap(
                        lambda case_id: f(
                            case_id,
                            input_data=input_list[case_id],
                            output_dict=output_dict,
                            n=n,
                        ),
                        parallel_case_ids,
                    ),
                    total=len(parallel_case_ids),
                    desc=f"Running {f.__name__} with {self._model_name_str}",
                )
            )

    def __repr__(self) -> str:
        return (
            f"LLMEngine("
            f"model_name='{self._model_name_str}', "
            f"api_provider='{self._api_provider}', "
            f"is_instruct={self._is_instruct}, "
            f"is_reasoning={self._is_reasoning}, "
            f"max_tokens={self._config.max_tokens}, "
            f"temperature={self._config.temperature}, "
            f"top_p={self._config.top_p}, "
            f"min_p={self._config.min_p}"
            f")"
        )

    def __getattr__(self, name):
        config_attrs = {
            "temperature",
            "top_p",
            "max_tokens",
            "min_p",
            "stop",
            "logprobs",
            "echo",
        }
        if name in config_attrs:
            return getattr(self._config, name)
        raise AttributeError(f"No attribute '{name}'")
    
    
    @property
    def model_name(self) -> str:
        return self._model_name_str

    @property
    def api_provider(self) -> str:
        return self._api_provider

    @property
    def is_instruct(self) -> bool:
        return self._is_instruct

    @property
    def is_reasoning(self) -> bool:
        return self._is_reasoning

    @property
    def config(self) -> LLMConfig:
        return self._config

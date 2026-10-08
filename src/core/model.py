"""
Module containing wrappers for local (HuggingFace) and API-based LLMs.
"""

import abc
import typing
from pathlib import Path
import logging
import time

logging.getLogger("httpx").setLevel(logging.ERROR)

import transformers
from openai import OpenAI, APIError as OpenAIAPIError
from anthropic import Anthropic, APIError as AnthropicAPIError
from google import genai
from google.genai import errors as GeminiErrors
import torch
import gc

logger = logging.getLogger(Path(__file__).name)


class BaseModel(abc.ABC):
    """
    Interface for all local LLM wrappers
    """

    def __init__(
        self,
        name: str,
        max_out_tokens: int,
        stop_list: list[str] | None = None,
    ):
        self.name = name
        self.max_out_tokens = max_out_tokens
        self.stop_list = stop_list if stop_list is not None else []

    @typing.final
    def prompt(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Generate the model's response based on a prompt.

        :param system_prompt: The system prompt.
        :type system_prompt: str
        :param user_prompt: The user prompt.
        :type user_prompt: str
        :return: the model's response
        :rtype: str
        """
        response = self._generate_response(system_prompt, user_prompt)
        for remove_word in self.stop_list:
            response = response.replace(remove_word, "")

        return response

    @typing.final
    def get_name(self) -> str:
        """
        Get the model's assigned pseudoname.

        :return: The name of the model.
        :rtype: str
        """
        return self.name

    @abc.abstractmethod
    def _generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Model-specific method which generates the LLM's response

        :param system_prompt: The system prompt.
        :type system_prompt: str
        :param user_prompt: The user prompt.
        :type user_prompt: str
        :return: The model's response
        :rtype: str
        """
        raise NotImplementedError("Abstract class call")


class TransformersModel(BaseModel):
    """
    HuggingFace Transformers model wrapper.
    """

    def __init__(
        self,
        model_path: str | Path,
        name: str,
        max_out_tokens: int = 512,
        max_in_tokens: int = 4096,
        remove_string_list: list[str] | None = None,
        chat_template_kwargs: dict | None = None,
        top_p: float = 0.95,
        top_k: int = 20,
        temperature: float = 0.7,
        do_sample: bool = True,
    ):
        super().__init__(name, max_out_tokens, remove_string_list)

        self.chat_template_kwargs = chat_template_kwargs or {}
        self.chat_template_kwargs.setdefault("enable_thinking", False)
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.do_sample = do_sample
        self.max_in_tokens = max_in_tokens

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto"
        )
        self.model.eval()

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

        model_size = self.model.get_memory_footprint() / 2**20
        logger.info(f"Model memory footprint: {model_size:.2f} MB")

    @torch.inference_mode()
    def _generate_response(self, system_prompt: str, user_prompt: str) -> str:
        assert type(system_prompt) is str
        assert type(user_prompt) is str
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **self.chat_template_kwargs,
                )
            except TypeError:
                prompt_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            logger.warning("Tokenizer has no chat template; falling back.")
            prompt_text = (
                f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
            )
        
        if self.max_in_tokens:
            inputs = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_in_tokens,
            ).to(self.model.device)
        else:
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(
                self.model.device
            )

        generate_kwargs = {
            "max_new_tokens": self.max_out_tokens,
            "max_length": None,
            "do_sample": self.do_sample,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        output_ids = self.model.generate(**inputs, **generate_kwargs)

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )

        return response.strip()

    def unload(self):
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()


class VLMModel(BaseModel):
    """
    Vision-Language Model wrapper for models that use AutoProcessor.
    Supports text-only input for VLMs like Gemma 4.
    """

    def __init__(
        self,
        model_path: str | Path,
        name: str,
        max_out_tokens: int = 512,
        max_in_tokens: int = 4096,
        remove_string_list: list[str] | None = None,
        chat_template_kwargs: dict | None = None,
        top_p: float = 0.95,
        top_k: int = 20,
        temperature: float = 0.7,
        do_sample: bool = True,
    ):
        super().__init__(name, max_out_tokens, remove_string_list)

        self.chat_template_kwargs = chat_template_kwargs or {}
        self.chat_template_kwargs.setdefault("enable_thinking", False)
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.do_sample = do_sample
        self.max_in_tokens = max_in_tokens

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.eval()

        self.processor = transformers.AutoProcessor.from_pretrained(model_path)

        model_size = self.model.get_memory_footprint() / 2**20
        logger.info(f"Initialized VLM model: {name} ({model_size:.2f} MB)")

    @torch.inference_mode()
    def _generate_response(self, system_prompt: str, user_prompt: str) -> str:
        assert type(system_prompt) is str
        assert type(user_prompt) is str

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )

        if self.max_in_tokens:
            inputs = self.processor(
                text=text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_in_tokens,
            ).to(self.model.device)
        else:
            inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        generate_kwargs = {
            "max_new_tokens": self.max_out_tokens,
            "do_sample": self.do_sample,
        }
        if self.do_sample:
            generate_kwargs["top_p"] = self.top_p
            generate_kwargs["top_k"] = self.top_k
            generate_kwargs["temperature"] = self.temperature

        outputs = self.model.generate(**inputs, **generate_kwargs)
        response = self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)

        return response.strip()

    def unload(self):
        del self.model
        del self.processor
        gc.collect()
        torch.cuda.empty_cache()


class OpenAIModel(BaseModel):
    """
    OpenAI API-compatible model wrapper.

    Supports OpenAI API and compatible endpoints.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        name: str,
        max_out_tokens: int,
        temperature: float = 0.0,
        remove_string_list: list[str] | None = None,
    ):
        super().__init__(name, max_out_tokens, remove_string_list)

        self.model_name = model_name
        self.temperature = temperature
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        logger.info(f"Initialized OpenAI model: {model_name} at {base_url}")

    def _generate_response(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_completion_tokens=self.max_out_tokens,
                    temperature=self.temperature,
                )
                return response.choices[0].message.content.strip()

            except OpenAIAPIError as e:
                if attempt < 2:
                    logger.warning(f"Attempt {attempt + 1} failed. Waiting 30s...")
                    time.sleep(30)
                else:
                    logger.error("Final attempt failed.")
                    raise e


class AnthropicModel(BaseModel):
    """
    Anthropic API-compatible model wrapper.

    Supports Anthropic API and compatible endpoints.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        name: str,
        max_out_tokens: int,
        temperature: float = 0.0,
        remove_string_list: list[str] | None = None,
    ):
        super().__init__(name, max_out_tokens, remove_string_list)

        self.model_name = model_name
        self.temperature = temperature
        self.client = Anthropic(
            api_key=api_key,
            max_retries=5
        )

        logger.info(f"Initialized Anthropic model: {model_name}")

    def _generate_response(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(3):
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    system=system_prompt,
                    messages=messages,
                    max_tokens=self.max_out_tokens,
                    temperature=self.temperature,
                )
                return response.content[0].text.strip()

            except AnthropicAPIError as e:
                if attempt < 2:
                    logger.warning(f"Attempt {attempt + 1} failed. Waiting 30s...")
                    time.sleep(30)
                else:
                    logger.error("Final attempt failed.")
                    raise e


class GeminiModel(BaseModel):
    """
    Google API-compatible model wrapper.

    Supports Google API and compatible endpoints.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        name: str,
        max_out_tokens: int,
        temperature: float = 0.0,
        remove_string_list: list[str] | None = None,
    ):
        super().__init__(name, max_out_tokens, remove_string_list)

        self.model_name = model_name
        self.temperature = temperature
        self.client = genai.Client(
            api_key=api_key
        )

        logger.info(f"Initialized Google model: {model_name}")

    def _generate_response(self, system_prompt: str, user_prompt: str) -> str:
        messages = {"text": user_prompt}

        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    config={
                        "system_instruction": system_prompt,
                        "temperature": self.temperature,
                        "max_output_tokens": self.max_out_tokens,
                    },
                    contents=messages
                )
                return response.text.strip()

            except GeminiErrors.APIError as e:
                if attempt < 2:
                    logger.warning(f"Attempt {attempt + 1} failed. Waiting 30s...")
                    time.sleep(30)
                else:
                    logger.error("Final attempt failed.")
                    raise e
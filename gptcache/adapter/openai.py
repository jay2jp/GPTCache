import base64
import json
import os
import time
from io import BytesIO
from typing import Any, AsyncGenerator, Iterator, List

# GPTCache imports
from gptcache import cache
from gptcache.adapter.adapter import aadapt, adapt
from gptcache.adapter.base import BaseCacheLLM
from gptcache.manager.scalar_data.base import Answer, DataType
from gptcache.utils import import_openai, import_pillow
from gptcache.utils.error import wrap_error
from gptcache.utils.response import (
    get_audio_text_from_openai_answer,
    get_image_from_openai_b64,
    get_image_from_openai_url,
    get_message_from_openai_answer,
    get_stream_message_from_openai_answer,
    get_text_from_openai_answer,
)
from gptcache.utils.token import token_counter

# Import OpenAI, and bring in the new 1.0+ error class
import_openai()
from openai import OpenAI, AsyncOpenAI

# Initialize default clients as None
client = None
aclient = None

def init_clients(custom_client=None, custom_async_client=None):
    """Initialize OpenAI clients with custom instances or default configuration"""
    global client, aclient
    
    if custom_client:
        client = custom_client
    else:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )
    
    if custom_async_client:
        aclient = custom_async_client
    else:
        aclient = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )

# Initialize with default clients
init_clients()

from openai import OpenAIError

from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

# ------------------------------------------------------------------------------
# Utility functions (unchanged)
# ------------------------------------------------------------------------------


async def async_iter(input_list):
    for item in input_list:
        yield item


def _construct_resp_from_cache(return_message, saved_token):
    return {
        "gptcache": True,
        "saved_token": saved_token,
        "choices": [
            {
                "message": {"role": "assistant", "content": return_message},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "created": int(time.time()),
        "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
        "object": "chat.completion",
    }


def _construct_stream_resp_from_cache(return_message, saved_token):
    created = int(time.time())
    chunks = []
    
    # First chunk with role
    chunks.append(ChatCompletionChunk(
        id="chat-chunk-" + str(created),
        choices=[Choice(
            delta=ChoiceDelta(role="assistant"),
            finish_reason=None,
            index=0
        )],
        created=created,
        model="cached",
        object="chat.completion.chunk"
    ))
    
    # Content chunk
    chunks.append(ChatCompletionChunk(
        id="chat-chunk-" + str(created),
        choices=[Choice(
            delta=ChoiceDelta(content=return_message),
            finish_reason=None,
            index=0
        )],
        created=created,
        model="cached",
        object="chat.completion.chunk"
    ))
    
    # Final chunk with finish reason
    chunks.append(ChatCompletionChunk(
        id="chat-chunk-" + str(created),
        choices=[Choice(
            delta=ChoiceDelta(),
            finish_reason="stop",
            index=0
        )],
        created=created,
        model="cached",
        object="chat.completion.chunk",
        # Custom fields for gptcache
        gptcache=True,
        saved_token=saved_token
    ))
    
    return chunks


def _construct_text_from_cache(return_text):
    return {
        "gptcache": True,
        "choices": [
            {
                "text": return_text,
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "created": int(time.time()),
        "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
        "object": "text_completion",
    }


def _construct_image_create_resp_from_cache(image_data, response_format, size):
    import_pillow()
    from PIL import Image as PILImage  # pylint: disable=C0415

    img_bytes = base64.b64decode((image_data))
    img_file = BytesIO(img_bytes)  # convert image to file-like object
    img = PILImage.open(img_file)
    new_size = tuple(int(a) for a in size.split("x"))
    if new_size != img.size:
        img = img.resize(new_size)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
    else:
        buffered = img_file

    if response_format == "url":
        target_url = os.path.abspath(str(int(time.time())) + ".jpeg")
        with open(target_url, "wb") as f:
            f.write(buffered.getvalue())
        image_data = target_url
    elif response_format == "b64_json":
        image_data = base64.b64encode(buffered.getvalue()).decode("ascii")
    else:
        raise AttributeError(
            f"Invalid response_format: {response_format} is not one of ['url', 'b64_json']"
        )

    return {
        "gptcache": True,
        "created": int(time.time()),
        "data": [{response_format: image_data}],
    }


def _construct_audio_text_from_cache(return_text):
    return {
        "gptcache": True,
        "text": return_text,
    }


def _num_tokens_from_messages(messages):
    tokens_per_message = 3
    tokens_per_name = 1

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += token_counter(value)
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


# ------------------------------------------------------------------------------
# Modernized Wrappers
# ------------------------------------------------------------------------------

class ChatCompletion(BaseCacheLLM):
    """
    Modernized OpenAI ChatCompletion wrapper,
    calling openai.ChatCompletion methods directly instead
    of subclassing them.
    """

    @classmethod
    def _llm_handler(cls, *llm_args, **llm_kwargs):
        """
        Synchronous ChatCompletion call.
        """
        try:
            if cls.llm is None:  # if no custom LLM, call OpenAI
                return client.chat.completions.create(*llm_args, **llm_kwargs)
            else:  # otherwise call custom LLM
                return cls.llm(*llm_args, **llm_kwargs)
        except OpenAIError as e:
            raise wrap_error(e) from e

    @classmethod
    async def _allm_handler(cls, *llm_args, **llm_kwargs):
        """
        Asynchronous ChatCompletion call.
        """
        try:
            if cls.llm is None:
                return await aclient.chat.completions.create(*llm_args, **llm_kwargs)
            else:
                return await cls.llm(*llm_args, **llm_kwargs)
        except OpenAIError as e:
            raise wrap_error(e) from e

    @staticmethod
    def _update_cache_callback(llm_data, update_cache_func, *args, **kwargs):
        try:
            if isinstance(llm_data, AsyncGenerator):
                async def hook_openai_data(it):
                    total_answer = ""
                    async for item in it:
                        content = get_stream_message_from_openai_answer(item) or ""
                        total_answer += content
                        yield item
                    update_cache_func(Answer(total_answer, DataType.STR))
                return hook_openai_data(llm_data)
            elif not isinstance(llm_data, Iterator):
                # Handle modern OpenAI response object
                try:
                    if llm_data and hasattr(llm_data, 'choices') and len(llm_data.choices) > 0:
                        choice = llm_data.choices[0]
                        if hasattr(choice, 'message') and choice.message:
                            message_content = choice.message.content or ""
                            update_cache_func(Answer(message_content, DataType.STR))
                        else:
                            # Fallback for other response types
                            update_cache_func(Answer(get_message_from_openai_answer(llm_data) or "", DataType.STR))
                    else:
                        # Handle empty or invalid response
                        update_cache_func(Answer("", DataType.STR))
                except (AttributeError, IndexError) as e:
                    # Log the error but don't crash
                    print(f"Error processing response: {e}")
                    update_cache_func(Answer("", DataType.STR))
                return llm_data
            else:
                # streaming in an iterable
                def hook_openai_data(it):
                    total_answer = ""
                    for item in it:
                        # Handle modern OpenAI streaming response
                        if hasattr(item, 'choices') and len(item.choices) > 0:
                            choice = item.choices[0]
                            if hasattr(choice, 'delta'):
                                content = choice.delta.content if hasattr(choice.delta, 'content') else ""
                                total_answer += content or ""  # Handle None case
                                yield item
                        else:
                            # Fallback for older format
                            content = get_stream_message_from_openai_answer(item) or ""
                            total_answer += content
                            yield item
                    update_cache_func(Answer(total_answer, DataType.STR))
                return hook_openai_data(llm_data)
        except Exception as e:
            print(f"Unexpected error in update_cache_callback: {e}")
            update_cache_func(Answer("", DataType.STR))
            return llm_data

    @classmethod
    def create(cls, *args, **kwargs):
        """
        Sync create that integrates with GPTCache adapt.
        """
        chat_cache = kwargs.get("cache_obj", cache)
        enable_token_counter = chat_cache.config.enable_token_counter

        def cache_data_convert(cache_data):
            if enable_token_counter:
                input_token = _num_tokens_from_messages(kwargs.get("messages"))
                output_token = token_counter(cache_data)
                saved_token = [input_token, output_token]
            else:
                saved_token = [0, 0]

            if kwargs.get("stream", False):
                return _construct_stream_resp_from_cache(cache_data, saved_token)
            return _construct_resp_from_cache(cache_data, saved_token)

        kwargs = cls.fill_base_args(**kwargs)
        return adapt(
            cls._llm_handler,
            cache_data_convert,
            cls._update_cache_callback,
            *args,
            **kwargs,
        )

    @classmethod
    async def acreate(cls, *args, **kwargs):
        """
        Async create that integrates with GPTCache aadapt.
        """
        chat_cache = kwargs.get("cache_obj", cache)
        enable_token_counter = chat_cache.config.enable_token_counter

        def cache_data_convert(cache_data):
            if enable_token_counter:
                input_token = _num_tokens_from_messages(kwargs.get("messages"))
                output_token = token_counter(cache_data)
                saved_token = [input_token, output_token]
            else:
                saved_token = [0, 0]

            if kwargs.get("stream", False):
                return async_iter(
                    _construct_stream_resp_from_cache(cache_data, saved_token)
                )
            return _construct_resp_from_cache(cache_data, saved_token)

        kwargs = cls.fill_base_args(**kwargs)
        return await aadapt(
            cls._allm_handler,
            cache_data_convert,
            cls._update_cache_callback,
            *args,
            **kwargs,
        )


class Completion(BaseCacheLLM):
    """
    Modernized OpenAI Completion wrapper,
    calling openai.Completion methods directly.
    """

    @classmethod
    def _llm_handler(cls, *llm_args, **llm_kwargs):
        try:
            if not cls.llm:
                return client.completions.create(*llm_args, **llm_kwargs)
            else:
                return cls.llm(*llm_args, **llm_kwargs)
        except OpenAIError as e:
            raise wrap_error(e) from e

    @classmethod
    async def _allm_handler(cls, *llm_args, **llm_kwargs):
        try:
            if cls.llm is None:
                return await aclient.completions.create(*llm_args, **llm_kwargs)
            else:
                return await cls.llm(*llm_args, **llm_kwargs)
        except OpenAIError as e:
            raise wrap_error(e) from e

    @staticmethod
    def _cache_data_convert(cache_data):
        return _construct_text_from_cache(cache_data)

    @staticmethod
    def _update_cache_callback(llm_data, update_cache_func, *args, **kwargs):
        try:
            if isinstance(llm_data, AsyncGenerator):
                async def hook_openai_data(it):
                    total_answer = ""
                    async for item in it:
                        content = get_stream_message_from_openai_answer(item) or ""
                        total_answer += content
                        yield item
                    update_cache_func(Answer(total_answer, DataType.STR))
                return hook_openai_data(llm_data)
            elif not isinstance(llm_data, Iterator):
                # Handle modern OpenAI response object
                try:
                    if llm_data and hasattr(llm_data, 'choices') and len(llm_data.choices) > 0:
                        choice = llm_data.choices[0]
                        if hasattr(choice, 'message') and choice.message:
                            message_content = choice.message.content or ""
                            update_cache_func(Answer(message_content, DataType.STR))
                        else:
                            # Fallback for other response types
                            update_cache_func(Answer(get_message_from_openai_answer(llm_data) or "", DataType.STR))
                    else:
                        # Handle empty or invalid response
                        update_cache_func(Answer("", DataType.STR))
                except (AttributeError, IndexError) as e:
                    # Log the error but don't crash
                    print(f"Error processing response: {e}")
                    update_cache_func(Answer("", DataType.STR))
                return llm_data
            else:
                # streaming in an iterable
                def hook_openai_data(it):
                    total_answer = ""
                    for item in it:
                        # Handle modern OpenAI streaming response
                        if hasattr(item, 'choices') and len(item.choices) > 0:
                            choice = item.choices[0]
                            if hasattr(choice, 'delta'):
                                content = choice.delta.content if hasattr(choice.delta, 'content') else ""
                                total_answer += content or ""  # Handle None case
                                yield item
                        else:
                            # Fallback for older format
                            content = get_stream_message_from_openai_answer(item) or ""
                            total_answer += content
                            yield item
                    update_cache_func(Answer(total_answer, DataType.STR))
                return hook_openai_data(llm_data)
        except Exception as e:
            print(f"Unexpected error in update_cache_callback: {e}")
            update_cache_func(Answer("", DataType.STR))
            return llm_data

    @classmethod
    def create(cls, *args, **kwargs):
        kwargs = cls.fill_base_args(**kwargs)
        return adapt(
            cls._llm_handler,
            cls._cache_data_convert,
            cls._update_cache_callback,
            *args,
            **kwargs,
        )

    @classmethod
    async def acreate(cls, *args, **kwargs):
        kwargs = cls.fill_base_args(**kwargs)
        return await aadapt(
            cls._allm_handler,
            cls._cache_data_convert,
            cls._update_cache_callback,
            *args,
            **kwargs,
        )


class Audio:
    """
    Modernized OpenAI Audio wrapper, calling openai.Audio.* methods directly.
    """

    @classmethod
    def transcribe(cls, model: str, file: Any, *args, **kwargs):
        def llm_handler(*llm_args, **llm_kwargs):
            try:
                return client.audio.transcribe(*llm_args, **llm_kwargs)
            except OpenAIError as e:
                raise wrap_error(e) from e

        def cache_data_convert(cache_data):
            return _construct_audio_text_from_cache(cache_data)

        def update_cache_callback(llm_data, update_cache_func, *a, **kw):
            update_cache_func(
                Answer(get_audio_text_from_openai_answer(llm_data), DataType.STR)
            )
            return llm_data

        return adapt(
            llm_handler,
            cache_data_convert,
            update_cache_callback,
            model=model,
            file=file,
            *args,
            **kwargs,
        )

    @classmethod
    def translate(cls, model: str, file: Any, *args, **kwargs):
        def llm_handler(*llm_args, **llm_kwargs):
            try:
                return client.audio.translate(*llm_args, **llm_kwargs)
            except OpenAIError as e:
                raise wrap_error(e) from e

        def cache_data_convert(cache_data):
            return _construct_audio_text_from_cache(cache_data)

        def update_cache_callback(llm_data, update_cache_func, *a, **kw):
            update_cache_func(
                Answer(get_audio_text_from_openai_answer(llm_data), DataType.STR)
            )
            return llm_data

        return adapt(
            llm_handler,
            cache_data_convert,
            update_cache_callback,
            model=model,
            file=file,
            *args,
            **kwargs,
        )


class Image(BaseCacheLLM):
    """
    Modernized OpenAI Image wrapper, calling openai.Image methods directly.
    """

    @classmethod
    def _llm_handler(cls, *llm_args, **llm_kwargs):
        try:
            return client.images.generate(*llm_args, **llm_kwargs)
        except OpenAIError as e:
            raise wrap_error(e) from e

    @classmethod
    def create(cls, *args, **kwargs):
        response_format = kwargs.pop("response_format", "url")
        size = kwargs.pop("size", "256x256")

        def cache_data_convert(cache_data):
            return _construct_image_create_resp_from_cache(
                image_data=cache_data,
                response_format=response_format,
                size=size,
            )

        def update_cache_callback(llm_data, update_cache_func, *a, **kw):
            if response_format == "b64_json":
                img_b64 = get_image_from_openai_b64(llm_data)
                if isinstance(img_b64, str):
                    img_b64 = img_b64.encode("ascii")
                update_cache_func(Answer(img_b64, DataType.IMAGE_BASE64))
            elif response_format == "url":
                update_cache_func(
                    Answer(get_image_from_openai_url(llm_data), DataType.IMAGE_URL)
                )
            return llm_data

        return adapt(
            cls._llm_handler,
            cache_data_convert,
            update_cache_callback,
            response_format=response_format,
            size=size,
            *args,
            **kwargs,
        )


class Moderation(BaseCacheLLM):
    """
    Modernized OpenAI Moderation wrapper, calling openai.Moderation methods directly.
    """

    @classmethod
    def _llm_handler(cls, *llm_args, **llm_kwargs):
        try:
            if not cls.llm:
                return client.moderations.create(*llm_args, **llm_kwargs)
            else:
                return cls.llm(*llm_args, **llm_kwargs)
        except OpenAIError as e:
            raise wrap_error(e) from e

    @classmethod
    def _cache_data_convert(cls, cache_data):
        return json.loads(cache_data)

    @classmethod
    def _update_cache_callback(cls, llm_data, update_cache_func, *args, **kwargs):
        try:
            if isinstance(llm_data, AsyncGenerator):
                async def hook_openai_data(it):
                    total_answer = ""
                    async for item in it:
                        content = get_stream_message_from_openai_answer(item) or ""
                        total_answer += content
                        yield item
                    update_cache_func(Answer(total_answer, DataType.STR))
                return hook_openai_data(llm_data)
            elif not isinstance(llm_data, Iterator):
                # Handle modern OpenAI response object
                try:
                    if llm_data and hasattr(llm_data, 'choices') and len(llm_data.choices) > 0:
                        choice = llm_data.choices[0]
                        if hasattr(choice, 'message') and choice.message:
                            message_content = choice.message.content or ""
                            update_cache_func(Answer(message_content, DataType.STR))
                        else:
                            # Fallback for other response types
                            update_cache_func(Answer(get_message_from_openai_answer(llm_data) or "", DataType.STR))
                    else:
                        # Handle empty or invalid response
                        update_cache_func(Answer("", DataType.STR))
                except (AttributeError, IndexError) as e:
                    # Log the error but don't crash
                    print(f"Error processing response: {e}")
                    update_cache_func(Answer("", DataType.STR))
                return llm_data
            else:
                # streaming in an iterable
                def hook_openai_data(it):
                    total_answer = ""
                    for item in it:
                        # Handle modern OpenAI streaming response
                        if hasattr(item, 'choices') and len(item.choices) > 0:
                            choice = item.choices[0]
                            if hasattr(choice, 'delta'):
                                content = choice.delta.content if hasattr(choice.delta, 'content') else ""
                                total_answer += content or ""  # Handle None case
                                yield item
                        else:
                            # Fallback for older format
                            content = get_stream_message_from_openai_answer(item) or ""
                            total_answer += content
                            yield item
                    update_cache_func(Answer(total_answer, DataType.STR))
                return hook_openai_data(llm_data)
        except Exception as e:
            print(f"Unexpected error in update_cache_callback: {e}")
            update_cache_func(Answer("", DataType.STR))
            return llm_data

    @classmethod
    def create(cls, *args, **kwargs):
        """
        Ensures the cached result has the same length in 'results' as the input
        might require. If they don't match, we force skip the cache.
        """
        kwargs = cls.fill_base_args(**kwargs)
        res = adapt(
            cls._llm_handler,
            cls._cache_data_convert,
            cls._update_cache_callback,
            *args,
            **kwargs,
        )

        input_request_param = kwargs.get("input")
        expect_res_len = 1
        if isinstance(input_request_param, List):
            expect_res_len = len(input_request_param)

        if len(res.get("results", [])) != expect_res_len:
            kwargs["cache_skip"] = True
            res = adapt(
                cls._llm_handler,
                cls._cache_data_convert,
                cls._update_cache_callback,
                *args,
                **kwargs,
            )
        return res

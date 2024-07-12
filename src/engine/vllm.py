import uuid
from typing import AsyncGenerator, AsyncIterator, Dict, List, Sequence

from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.outputs import RequestOutput

from src.config.model import VLLMConfig
from src.utils.template import Template, get_template_and_fix_tokenizer

from .base import BaseEngine, LLMResult


class VLLMEngine(BaseEngine, VLLMConfig):
    model: AsyncLLMEngine
    template: Template
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
    processor: AutoProcessor | None = None  # use for multi-modal model

    @classmethod
    def from_config(cls, config: VLLMConfig):
        model = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(
                model=config.path,
                max_model_len=config.max_seq_len,
                dtype=config.dtype,
                tensor_parallel_size=config.tensor_parallel_size,
                gpu_memory_utilization=config.gpu_memory_utilization,
                disable_log_stats=True,
                disable_log_requests=True,
                trust_remote_code=True,
            )
        )
        tokenizer = AutoTokenizer.from_pretrained(config.path, use_fast=True, trust_remote_code=True)
        template = get_template_and_fix_tokenizer(tokenizer, config.template)

        kwargs = config.model_dump()
        kwargs.pop("template")

        return cls(model=model, tokenizer=tokenizer, template=template, **kwargs)

    async def _generate(
        self,
        messages: Sequence[Dict[str, str]],
        system: str | None = None,
        tools: str | None = None,
        top_p: float | None = 1.0,
        top_k: int | None = -1,
        temperature: float | None = 1.0,
        repetition_penalty: float = 1.0,
        max_tokens: int | None = 512,
        n: int = 1,
        num_beams: int = 1,
    ) -> AsyncIterator[RequestOutput]:
        request_id = "chatcmpl-{}".format(uuid.uuid4().hex)
        sampling_params = SamplingParams(
            n=n,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p or 1.0,
            temperature=temperature or 1.0,
            use_beam_search=num_beams > 1,
            stop_token_ids=[self.tokenizer.eos_token_id, *self.tokenizer.additional_special_tokens_ids],
            max_tokens=max_tokens or 512,
            skip_special_tokens=True,
        )

        # TODO support multi-modal
        multi_modal_data = None
        paired_messages = messages + [{"role": "assistant", "content": ""}]
        prompt_token_ids, _ = self.template.encode_oneturn(
            tokenizer=self.tokenizer,
            messages=paired_messages,
            system=system,
            tools=tools,
        )

        output_generator = self.model.generate(
            prompt=None,
            sampling_params=sampling_params,
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            multi_modal_data=multi_modal_data,
        )

        return output_generator

    async def invoke(
        self,
        messages: Sequence[Dict[str, str]],
        system: str | None = None,
        tools: str | None = None,
        **generate_kwargs,
    ) -> List[LLMResult]:
        final_output = None
        generator = await self._generate(messages=messages, system=system, tools=tools, **generate_kwargs)
        async for request_output in generator:
            final_output = request_output

        results = []
        for output in final_output.outputs:
            results.append(
                LLMResult(
                    response_text=output.text,
                    response_length=len(output.token_ids),
                    prompt_length=len(final_output.prompt_token_ids),
                    finish_reason=output.finish_reason,
                )
            )

        return results

    async def stream(
        self,
        messages: Sequence[Dict[str, str]],
        system: str | None = None,
        tools: str | None = None,
        **generate_kwargs,
    ) -> AsyncGenerator[LLMResult, None]:

        generator = await self._generate(messages=messages, system=system, tools=tools, **generate_kwargs)
        generated_text = ""
        async for result in generator:
            delta_text = result.outputs[0].text[len(generated_text) :]
            generated_text = result.outputs[0].text
            yield LLMResult(
                response_text=delta_text,
                response_length=1,
                prompt_length=len(result.prompt_token_ids),
                finish_reason=result.outputs[0].finish_reason,
            )

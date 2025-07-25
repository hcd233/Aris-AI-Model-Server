from typing import (Any, AsyncGenerator, AsyncIterator, Dict, List, Literal,
                    Sequence)

import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import generate_step, load
from pydantic import BaseModel
from transformers import AutoProcessor

from src.config.model import MLXConfig
from src.utils.template import Template, get_template_and_fix_tokenizer

from ..base import BaseEngine, LLMResult


class StepOutput(BaseModel):
    text: str
    token_ids: List[int]
    finish_reason: Literal["finish", "length"]


class RequestOutput(BaseModel):
    outputs: List[StepOutput]
    prompt_token_ids: List[int]


class MLXEngine(BaseEngine, MLXConfig):
    model: Any  # nn.Module
    template: Template
    tokenizer: TokenizerWrapper
    processor: AutoProcessor | None = None  # use for multi-modal model

    @classmethod
    def from_config(cls, config: MLXConfig):
        model, tokenizer = load(
            config.path,
        )

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
        # TODO support multi-modal
        multi_modal_data = None
        paired_messages = messages + [{"role": "assistant", "content": ""}]
        prompt_token_ids, _ = self.template.encode_oneturn(
            tokenizer=self.tokenizer,
            messages=paired_messages,
            system=system,
            tools=tools,
        )

        max_tokens = max_tokens or self.max_seq_len
        top_p = top_p or 1.0
        temperature = temperature or 1.0

        output_generator = generate_step(
            prompt=mx.array(prompt_token_ids),
            model=self.model,
            temp=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            repetition_context_size=None,
        )

        detokenizer = self.tokenizer.detokenizer
        detokenizer.reset()
        cnt = 0
        segment = ""
        for token, _ in output_generator:
            if token == self.tokenizer.eos_token_id or cnt >= max_tokens:
                break
            detokenizer.add_token(token)
            segment += detokenizer.last_segment
            # Yield the last segment if streaming
            yield RequestOutput(
                outputs=[
                    StepOutput(
                        text=segment,
                        token_ids=detokenizer.tokens,
                        finish_reason="length",
                    )
                ],
                prompt_token_ids=prompt_token_ids,
            )

        detokenizer.finalize()
        yield RequestOutput(
            outputs=[
                StepOutput(
                    text=detokenizer.text,
                    token_ids=detokenizer.tokens,
                    finish_reason="finish" if cnt < max_tokens else "length",
                )
            ],
            prompt_token_ids=prompt_token_ids,
        )

    async def _invoke(
        self,
        messages: Sequence[Dict[str, str]],
        system: str | None = None,
        tools: str | None = None,
        **generate_kwargs,
    ) -> List[LLMResult]:
        final_output = None
        generator = self._generate(messages=messages, system=system, tools=tools, **generate_kwargs)
        async for request_output in generator:
            final_output: RequestOutput = request_output

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

        generator = self._generate(messages=messages, system=system, tools=tools, **generate_kwargs)
        generated_text = ""
        async for result in generator:
            delta_text = result.outputs[0].text[len(generated_text) :]
            generated_text = result.outputs[0].text
            finish_reason = result.outputs[0].finish_reason
            yield LLMResult(
                response_text=delta_text,
                response_length=1,
                prompt_length=len(result.prompt_token_ids),
                finish_reason=finish_reason,
            )

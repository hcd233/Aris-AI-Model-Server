import json
from time import time
from typing import AsyncGenerator, Dict, List, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from sse_starlette import EventSourceResponse

from src.api.auth.bearer import auth_secret_key
from src.api.model.chat_cmpl import (
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionResponseUsage,
    ChatCompletionStreamResponse,
    ChatMessage,
    Finish,
    Function,
    FunctionCall,
    ModelCard,
    ModelList,
    Role,
)
from src.config.gbl import VLLM_MAPPING
from src.logger import logger
from src.utils.template import Role as DataRole

chat_completion_router = APIRouter()

role_mapping = {
    Role.USER: DataRole.USER.value,
    Role.ASSISTANT: DataRole.ASSISTANT.value,
    Role.SYSTEM: DataRole.SYSTEM.value,
    Role.FUNCTION: DataRole.FUNCTION.value,
    Role.TOOL: DataRole.OBSERVATION.value,
}


def _parse_chat_message(messages: List[ChatMessage]) -> List[Dict[Literal["role", "content"], str]]:
    if not messages:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No messages provided")

    if len(messages) % 2 == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Your messages order should be (system), user, assistant, user(at last) etc."
        )

    parsed_messages = []
    for i, message in enumerate(messages):
        if i % 2 == 0 and message.role not in [Role.USER, Role.TOOL]:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid role")
        elif i % 2 == 1 and message.role not in [Role.ASSISTANT, Role.FUNCTION]:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid role")

        if message.role == Role.ASSISTANT and isinstance(message.tool_calls, list) and len(message.tool_calls):
            name = message.tool_calls[0].function.name
            arguments = message.tool_calls[0].function.arguments
            content = json.dumps({"name": name, "argument": arguments}, ensure_ascii=False)
            parsed_messages.append({"role": role_mapping[Role.FUNCTION], "content": content})
        else:
            parsed_messages.append({"role": role_mapping[message.role], "content": message.content})

    return parsed_messages


async def _wrap_stream_tokens(model: str, stream_tokens: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
    token_cnt = 0
    start_time = time()

    async for new_token in stream_tokens:
        if not new_token:
            continue
        choice_data = ChatCompletionResponseStreamChoice(index=0, delta=ChatCompletionMessage(content=new_token), finish_reason=None)
        chunk = ChatCompletionStreamResponse(model=model, choices=[choice_data])

        token_cnt += 1
        yield json.dumps(chunk.model_dump(exclude_unset=True))

    choice_data = ChatCompletionResponseStreamChoice(index=0, delta=ChatCompletionMessage(), finish_reason=Finish.STOP)
    chunk = ChatCompletionStreamResponse(model=model, choices=[choice_data])

    elapsed_time = time() - start_time
    logger.info(f"[Chat Completions] num_tokens: {token_cnt} num_seconds: {elapsed_time:.2f}s rate: {token_cnt / elapsed_time:.2f} tokens/sec")
    yield json.dumps(chunk.model_dump(exclude_unset=True))
    yield "[DONE]"


@chat_completion_router.get("/models", response_model=ModelList, dependencies=[Depends(auth_secret_key)])
async def list_models():
    model_cards = [ModelCard(id=alias) for alias in VLLM_MAPPING.keys()]
    return ModelList(data=model_cards)


@chat_completion_router.post("/chat/completions", response_model=ChatCompletionResponse, dependencies=[Depends(auth_secret_key)])
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse | EventSourceResponse:
    if not request.messages:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid messages")

    if request.messages[0].role == Role.SYSTEM:
        system = request.messages.pop(0).content
    else:
        system = ""

    parsed_messages = _parse_chat_message(request.messages)

    tool_list = request.tools
    if isinstance(tool_list, list) and tool_list:
        try:
            tools = json.dumps([tool.function.model_dump(exclude_unset=True) for tool in tool_list], ensure_ascii=False)
        except Exception:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid tools")
    else:
        tools = ""

    try:
        engine = VLLM_MAPPING[request.model]
    except KeyError:
        logger.error(f"[Chat Completions] Invalid model name: {request.model}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Invalid model name: {request.model}")

    if request.stream:
        if tools:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot stream function calls.")

        stream_tokens = engine.stream(
            parsed_messages,
            system,
            tools,
            n=request.n,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )

        stream_events = _wrap_stream_tokens(request.model, stream_tokens)

        return EventSourceResponse(stream_events, media_type="text/event-stream")

    responses = await engine.invoke(
        parsed_messages,
        system,
        tools,
        n=request.n,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
    )

    prompt_length, response_length = 0, 0
    choices = []

    for i, response in enumerate(responses):
        if tools:
            content = response["response_text"].strip().strip('"').strip("```")
            result = engine.template.format_tools.extract(content)
        else:
            result = response["response_text"]

        if isinstance(result, tuple):
            name, arguments = result
            function = Function(name=name, arguments=arguments)
            response_message = ChatCompletionMessage(role=Role.ASSISTANT, tool_calls=[FunctionCall(function=function)])
            finish_reason = Finish.TOOL
        else:
            response_message = ChatCompletionMessage(role=Role.ASSISTANT, content=result)
            finish_reason = Finish.STOP if response["finish_reason"] == Finish.STOP else Finish.LENGTH

        choices.append(ChatCompletionResponseChoice(index=i, message=response_message, finish_reason=finish_reason))

        prompt_length = response["prompt_length"]
        response_length += response["response_length"]

    usage = ChatCompletionResponseUsage(
        prompt_tokens=prompt_length,
        completion_tokens=response_length,
        total_tokens=prompt_length + response_length,
    )

    return ChatCompletionResponse(model=request.model, choices=choices, usage=usage)

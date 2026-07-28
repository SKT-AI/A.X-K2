#!/usr/bin/env python
"""Assertion suite for the A.X K2 chat template.

Renders `chat_template.jinja` the way `tokenizer.apply_chat_template` does — same
sandboxed Jinja environment, same `tojson` filter — and asserts on the result
instead of eyeballing it. The cases cover the message shapes that
OpenAI-compatible clients actually send, which is where chat templates usually
break: `content: null` on tool-call turns, `content` as a list of parts,
`arguments` as a JSON string rather than an object, parallel tool calls, and the
thinking / non-thinking prefixes.

    pip install jinja2 huggingface_hub

    python test_chat_template.py                          # skt/A.X-K2 from the Hub
    python test_chat_template.py path/to/chat_template.jinja
    python test_chat_template.py /path/to/local/checkpoint

Exits non-zero if any case fails, so it can gate a template change.
"""
import json
import os
import sys

from jinja2.sandbox import ImmutableSandboxedEnvironment

DEFAULT_SOURCE = "skt/A.X-K2"


def load_template(source):
    """Accept a template file, a checkpoint directory, or a Hugging Face repo id."""
    if os.path.isdir(source):
        source = os.path.join(source, "chat_template.jinja")
    if os.path.isfile(source):
        with open(source, encoding="utf-8") as fh:
            return source, fh.read()
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(source, "chat_template.jinja")
    with open(path, encoding="utf-8") as fh:
        return f"{source} (Hub)", fh.read()


PATH, TEMPLATE = load_template(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOURCE)

env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False, **kw)
TPL = env.from_string(TEMPLATE)

TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


def call(name="get_weather", arguments=None, nested=True):
    body = {"name": name}
    if arguments is not None:
        body["arguments"] = arguments
    return {"type": "function", "function": body} if nested else body


def render(**kw):
    return TPL.render(**kw)


def turns(text, role):
    """The rendered turns for `role`, without the leading marker or trailing <|im_end|>."""
    out = []
    for seg in text.split("<|im_start|>")[1:]:
        if seg.startswith(role):
            out.append(seg[len(role) :].lstrip("\n").split("<|im_end|>")[0])
    return out


def tool_call_payloads(text):
    """Every <tool_call> body in the assistant turns, parsed as JSON.

    Assistant turns only: the tools system prompt documents the format with a
    `<tool_call>` example whose placeholders are not valid JSON.
    """
    payloads = []
    for turn in turns(text, "assistant"):
        for chunk in turn.split("<tool_call>")[1:]:
            payloads.append(json.loads(chunk.split("</tool_call>")[0].strip()))
    return payloads


CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


# --------------------------------------------------------------- generation prompt
@check
def thinking_flag_controls_the_prefix():
    user = [{"role": "user", "content": "안녕"}]
    assert render(messages=user, add_generation_prompt=True, enable_thinking=True).endswith(
        "<|im_start|>assistant\n<think>"
    )
    for kw in ({"enable_thinking": False}, {}):
        assert render(messages=user, add_generation_prompt=True, **kw).endswith(
            "<|im_start|>assistant\n</think>"
        ), f"non-thinking prefix wrong for {kw}"
    # no generation prompt: the transcript ends with the last turn
    assert render(
        messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        add_generation_prompt=False,
    ).endswith("<|im_end|>")


@check
def roles_and_markers_are_wellformed():
    text = render(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ],
        add_generation_prompt=True,
    )
    assert turns(text, "system") == ["sys"]
    assert turns(text, "user") == ["u"]
    assert text.count("<|im_start|>") == text.count("<|im_end|>") + 1  # last one is open


# --------------------------------------------------------------- null / odd content
@check
def null_content_never_leaks_or_raises():
    """OpenAI-compatible clients send content=None on tool-call turns."""
    text = render(
        messages=[
            {"role": "user", "content": None},
            {"role": "assistant", "content": None, "tool_calls": [call(arguments={"city": "Seoul"})]},
            {"role": "tool", "content": None},
        ],
        tools=[TOOL],
        add_generation_prompt=True,
    )
    assert "None" not in text, "python None leaked into the prompt"
    assert turns(text, "user") == [""]


@check
def content_parts_are_flattened_to_text():
    """content may be [{"type": "text", "text": ...}] rather than a string."""
    text = render(
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "안녕"}, {"type": "text", "text": "!"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "반갑습니다"}]},
            {"role": "user", "content": "다시"},
        ],
        add_generation_prompt=True,
    )
    assert turns(text, "user")[0] == "안녕!", turns(text, "user")
    assert "반갑습니다" in turns(text, "assistant")[0]
    assert "'type'" not in text and "{" not in turns(text, "user")[0], "raw repr leaked"


@check
def reasoning_is_split_out_of_inline_think_tags():
    text = render(
        messages=[
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "<think>속내</think>답"},
            {"role": "user", "content": "q2"},
        ],
        add_generation_prompt=True,
    )
    turn = turns(text, "assistant")[0]
    assert turn.startswith("</think>") and turn.endswith("답"), turn
    assert "속내" not in turn, "stale reasoning kept on a closed turn"


# --------------------------------------------------------------- tool calls
@check
def arguments_render_as_an_object_for_both_wire_formats():
    """`arguments` is a JSON *string* over the OpenAI wire, a dict from python callers.
    Both must land as a JSON object, since the tool-call spec in the system prompt
    says `arguments` is a dictionary."""
    for arguments in ('{"city": "Seoul"}', {"city": "Seoul"}):
        text = render(
            messages=[
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": None, "tool_calls": [call(arguments=arguments)]},
            ],
            tools=[TOOL],
            add_generation_prompt=True,
        )
        (payload,) = tool_call_payloads(text)
        assert payload == {"name": "get_weather", "arguments": {"city": "Seoul"}}, (arguments, payload)


@check
def tool_calls_work_nested_and_flat():
    for nested in (True, False):
        text = render(
            messages=[
                {"role": "user", "content": "q"},
                {"role": "assistant", "tool_calls": [call(arguments={"city": "Seoul"}, nested=nested)]},
            ],
            tools=[TOOL],
            add_generation_prompt=True,
        )
        (payload,) = tool_call_payloads(text)
        assert payload["name"] == "get_weather", (nested, payload)


@check
def parallel_tool_calls_all_render():
    text = render(
        messages=[
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [call(arguments={"city": "Seoul"}), call(arguments={"city": "Busan"})],
            },
            {"role": "tool", "content": "20C"},
            {"role": "tool", "content": "22C"},
        ],
        tools=[TOOL],
        add_generation_prompt=True,
    )
    cities = [p["arguments"]["city"] for p in tool_call_payloads(text)]
    assert cities == ["Seoul", "Busan"], cities
    assert turns(text, "tool") == ["<tool_response>\n20C\n</tool_response>", "<tool_response>\n22C\n</tool_response>"]


@check
def arguments_may_be_absent():
    text = render(
        messages=[{"role": "user", "content": "q"}, {"role": "assistant", "tool_calls": [call()]}],
        tools=[TOOL],
        add_generation_prompt=True,
    )
    assert tool_call_payloads(text) == [{"name": "get_weather"}]


@check
def tools_block_appears_only_when_tools_are_given():
    for tools in (None, []):
        assert "<tools>" not in render(
            messages=[{"role": "user", "content": "q"}], tools=tools, add_generation_prompt=True
        )
    text = render(messages=[{"role": "user", "content": "q"}], tools=[TOOL], add_generation_prompt=True)
    body = text.split("<tools>\n")[1].split("</tools>")[0]
    assert json.loads(body.strip()) == TOOL


# --------------------------------------------------------------- thinking retention
@check
def show_thinking_overrides_retention():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "reasoning_content": "속내", "content": "답"},
        {"role": "user", "content": "q2"},
    ]
    shown = render(messages=messages, add_generation_prompt=True, show_thinking=True)
    hidden = render(messages=messages, add_generation_prompt=True, show_thinking=False)
    assert "<think>속내</think>" in turns(shown, "assistant")[0]
    assert "속내" not in hidden


@check
def reasoning_is_kept_on_a_toolcall_turn_still_in_flight():
    """After the last user message, an assistant tool-call turn keeps its reasoning."""
    text = render(
        messages=[
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "reasoning_content": "도구를 쓰자",
                "content": "",
                "tool_calls": [call(arguments={"city": "Seoul"})],
            },
            {"role": "tool", "content": "20C"},
        ],
        tools=[TOOL],
        add_generation_prompt=True,
    )
    assert "<think>도구를 쓰자</think>" in text, text


@check
def reasoning_is_dropped_once_a_new_user_turn_closes_the_round():
    text = render(
        messages=[
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "reasoning_content": "도구를 쓰자",
                "content": "",
                "tool_calls": [call(arguments={"city": "Seoul"})],
            },
            {"role": "tool", "content": "20C"},
            {"role": "assistant", "reasoning_content": "정리", "content": "20도입니다"},
            {"role": "user", "content": "q2"},
        ],
        tools=[TOOL],
        add_generation_prompt=True,
    )
    assert "도구를 쓰자" not in text and "정리" not in text, text


if __name__ == "__main__":
    print(f"template: {PATH}\n")
    failed = 0
    for fn in CHECKS:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}\n          {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {fn.__name__}\n          {type(e).__name__}: {e}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed")
    raise SystemExit(1 if failed else 0)

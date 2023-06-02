#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : consts.py
@Author  : wenge-research
@Ref     : https://github.com/databrickslabs/dolly/blob/master/training/consts.py
'''

DEFAULT_INPUT_MODEL = "wenge-research/yayi-7b"
INTRO = """A chat between a human and an AI assistant named YaYi.
YaYi is a helpful and harmless language model developed by Beijing Wenge Technology Co.,Ltd."""
INTRO_KEY = "<|System|>"
INSTRUCTION_KEY = "<|Human|>"
RESPONSE_KEY = "<|YaYi|>"
END_KEY = "<|End|>"
DEFAULT_SEED = 515

# For training. Without "input".
PROMPT_NO_INPUT_FORMAT = """{intro_key}:
{intro}

{instruction_key}:
{instruction}

{response_key}:
{response}

{end_key}""".format(
    intro_key=INTRO_KEY,
    intro=INTRO,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
    response="{response}",
    end_key=END_KEY,
)

# For training. With "input".
PROMPT_WITH_INPUT_FORMAT = """{intro_key}:
{intro}

{instruction_key}:
{instruction}
{input}

{response_key}:
{response}

{end_key}""".format(
    intro_key=INTRO_KEY,
    intro=INTRO,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    input="{input}",
    response_key=RESPONSE_KEY,
    response="{response}",
    end_key=END_KEY,
)

# For inference
PROMPT_FOR_GENERATION_FORMAT = """{intro_key}:
{intro}

{instruction_key}:
{instruction}

{response_key}:
""".format(
    intro_key=INTRO_KEY,
    intro=INTRO,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)

import time
import openai
import logging

from openai import OpenAI


def prompt_to_chatml(prompt : str, start_token : str = "<|im_start|>", end_token : str = "<|im_end|>") :
    """Convert a text prompt to ChatML formal

    Examples
    --------
    >>> prompt = "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>system
    name=example_user\nKnock knock.\n<|im_end|>\n<|im_start|>system name=example_assistant\nWho's
    there?\n<|im_end|>\n<|im_start|>user\nOrange.\n<|im_end|>"
    >>> print(prompt)
    <|im_start|>system
    You are a helpful assistant.
    <|im_end|>
    <|im_start|>system name=example_user
    Knock knock.
    <|im_end|>
    <|im_start|>system name=example_assistant
    Who's there?
    <|im_end|>
    <|im_start|>user
    Orange.
    <|im_end|>
    >>> prompt_to_chatml(prompt)
    [{'role': 'system', 'content': 'You are a helpful assistant.'},
     {'role': 'user', 'content': 'Knock knock.'},
     {'role': 'assistant', 'content': "Who's there?"},
     {'role': 'user', 'content': 'Orange.'}]
    """
    prompt = prompt.strip()
    assert prompt.startswith(start_token) and prompt.endswith(end_token)

    def string_to_dict(to_convert) :
        """Converts a string with equal signs to dictionary. E.g.
        >>> string_to_dict(" name=user university=stanford")
        {'name': 'user', 'university': 'stanford'}
        """
        return {s.split("=", 1)[0] : s.split("=", 1)[1] for s in to_convert.split(" ") if len(s) > 0}

    message = []
    for p in prompt.split("<|im_start|>")[1 :] :
        newline_splitted = p.split("\n", 1)
        role = newline_splitted[0].strip()
        content = newline_splitted[1].split(end_token, 1)[0].strip()

        if role.startswith("system") and role != "system" :
            # based on https://github.com/openai/openai-cookbook/blob/main/examples
            # /How_to_format_inputs_to_ChatGPT_models.ipynb
            # and https://github.com/openai/openai-python/blob/main/chatml.md it seems that system can specify a
            # dictionary of other args
            other_params = string_to_dict(role.split("system", 1)[-1])
            role = "system"
        else :
            other_params = dict()

        message.append(dict(content = content, role = role, **other_params))

    return message


def create_OpenAIclient(args) :
    return OpenAI(**args)


def openai_completion(
    client,
    chatml,
    openai_kwargs : dict,
    sleep_time : int = 5,
) :
    assert "model" in openai_kwargs, "'model' not in openai_kwargs"
    openai_kwargs = openai_kwargs.copy()

    while True :
        try :
            completion_batch = client.chat.completions.create(
                messages = chatml,
                **openai_kwargs,
            )
            break
        except openai.RateLimitError as e :
            logging.warning("OpenAI API request exceeded rate limit: {}".format(e))
            time.sleep(sleep_time)
        except Exception as e : # Bad Handling ... but it's fine for now
            logging.error("OpenAI API request failed: {}".format(e))
            time.sleep(sleep_time)
            '''
            return dict(
                responnse = None,
                cost = 0.0,
            )
            '''
    
    message = completion_batch.choices[0].message
    assert message.role == "assistant", "completion role is not `assistant`"
    def cost_calculation(usage) :
        if openai_kwargs["model"] in ("gpt-4o", "gpt-4o-2024-08-06", )  :
            return usage.prompt_tokens / 1000000 * 2.5 + usage.completion_tokens / 1000000 * 10.0
        elif openai_kwargs["model"] in ("gpt-4o-mini", "gpt-4o-mini-2024-07-18", ) :
            return usage.prompt_tokens / 1000000 * 0.15 + usage.completion_tokens / 1000000 * 0.6
        else :
            raise NotImplementedError
    return dict(
        response = message.content,
        cost = cost_calculation(completion_batch.usage),
    )


def openai_embedding(
        client,
        text : str,
        model : str,
        sleep_time : int = 5,
) :
    while True :
        try :
            completion_batch = client.embeddings.create(
                input = text,
                model = model,
            )
            break
        except openai.RateLimitError as e :
            logging.warning("OpenAI API request exceeded rate limit: {}".format(e))
            time.sleep(sleep_time)
        except Exception as e : # Bad Handling ... but it's fine for now
            logging.error("OpenAI API request failed: {}".format(e))
            time.sleep(sleep_time)
            '''
            return dict(
                responnse = None,
                cost = 0.0,
            )
            '''
    
    def cost_calculation(usage) :
        if model in ("text-embedding-3-small", )  :
            return usage.prompt_tokens / 1000000 * 0.02
        elif model in ("text-embedding-3-large", ) :
            return usage.prompt_tokens / 1000000 * 0.13
        else :
            raise NotImplementedError
    return dict(
        embedding = completion_batch.data[0].embedding,
        cost = cost_calculation(completion_batch.usage),
    )


def openai_moderation(
    client,
    text,
    model = "omni-moderation-2024-09-26",
    sleep_time : int = 5,
) :
    while True :
        try :
            moderation = client.moderations.create(input = text, model = model).model_dump()
            break
        except openai.RateLimitError as e :
            logging.warning("OpenAI API request exceeded rate limit: {}".format(e))
            time.sleep(sleep_time)
        except Exception as e :
            return None
    return moderation
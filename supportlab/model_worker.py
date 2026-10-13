"""One bounded SDK request; only sanitized public response fields leave the worker."""

import json
import logging
import sys

from .configuration import DeepSeekConfig


def request_once(payload: dict, *, client_factory=None) -> dict:
    # Config validation is repeated at the network boundary.
    config = DeepSeekConfig(**payload["config"])
    for name in ("openai", "httpx2", "httpcore2", "httpx", "httpcore"):
        logging.getLogger(name).disabled = True
    import openai
    factory = client_factory or openai.OpenAI
    try:
        with factory(api_key=config.api_key, base_url=config.base_url,
                     organization=None, project=None, max_retries=0, timeout=20.0) as client:
            tool_options = {"tools": payload["tools"], "tool_choice": "auto"} if payload["tools"] else {}
            response = client.chat.completions.create(
                model=config.model, messages=payload["messages"], **tool_options,
                stream=False, max_tokens=payload["max_tokens"],
                extra_body={"thinking": {"type": "disabled"}},
            )
        choice = response.choices[0]
        message = choice.message
        # Never serialize reasoning_content, headers, raw HTTP errors, or the key.
        return {"status": "ok", "finish_reason": choice.finish_reason,
                "message": {"content": message.content,
                            "tool_calls": [{"id": item.id, "type": item.type,
                                            "function": {"name": item.function.name, "arguments": item.function.arguments}}
                                           for item in message.tool_calls or []]},
                "usage": response.usage.model_dump() if response.usage else None}
    except openai.AuthenticationError:
        return {"status": "error", "error": "model_auth_error"}
    except openai.RateLimitError:
        return {"status": "error", "error": "model_rate_limit"}
    except openai.APITimeoutError:
        return {"status": "error", "error": "model_timeout"}
    except openai.APIConnectionError:
        return {"status": "error", "error": "model_connection_error"}
    except openai.APIStatusError:
        return {"status": "error", "error": "model_http_error"}
    except Exception:
        return {"status": "error", "error": "model_protocol_error"}


def main():
    try:
        raw = sys.stdin.read(100001)
        if len(raw) > 100000:
            raise ValueError("oversized worker request")
        result = request_once(json.loads(raw))
    except Exception:
        result = {"status": "error", "error": "model_configuration_error"}
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()

"""Read only this project's DeepSeek settings, without exposing credential values."""

from dataclasses import dataclass, field
import os
from pathlib import Path
import re

from .storage import PROJECT_ROOT


CONFIG_KEYS = ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL")
OFFICIAL_BASE_URLS = {"https://api.deepseek.com", "https://api.deepseek.com/v1"}


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class DeepSeekConfig:
    api_key: str = field(repr=False)
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-flash"

    def __post_init__(self):
        if (not isinstance(self.api_key, str) or not 8 <= len(self.api_key) <= 512
                or not self.api_key.isascii() or any(char.isspace() for char in self.api_key)):
            raise ConfigurationError("请在本地配置有效的 DEEPSEEK_API_KEY；不要将密钥发到聊天。")
        if self.base_url not in OFFICIAL_BASE_URLS:
            raise ConfigurationError("本阶段仅允许已确认的 DeepSeek 官网地址。")
        if not isinstance(self.model, str) or re.fullmatch(r"deepseek-[a-z0-9._-]{1,60}", self.model) is None:
            raise ConfigurationError("请检查 DEEPSEEK_MODEL，使用官方模型标识。")

    def safe_summary(self):
        return {"provider": "deepseek", "model": self.model, "base_url": self.base_url,
                "api_key_present": True, "network_checked": False}


def load_deepseek_config(path: Path = PROJECT_ROOT / ".env.local", *, environ=None) -> DeepSeekConfig:
    values = {}
    try:
        if path.exists():
            if path.stat().st_size > 8192:
                raise ConfigurationError("本地配置文件过大。")
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, separator, value = line.partition("=")
                key = key.strip()
                if key not in CONFIG_KEYS:
                    continue  # Do not use unrelated credentials or shell instructions.
                if not separator or key in values:
                    raise ConfigurationError("本地配置存在重复字段或格式错误。")
                value = value.strip()
                if value[:1] in ("'", '"'):
                    if len(value) < 2 or value[-1] != value[0]:
                        raise ConfigurationError("本地配置引号未闭合。")
                    value = value[1:-1]
                values[key] = value
    except (OSError, UnicodeError):
        raise ConfigurationError("无法读取本地配置文件。") from None
    environment = os.environ if environ is None else environ
    for key in CONFIG_KEYS:
        if key in environment:
            values[key] = environment[key]
    return DeepSeekConfig(api_key=values.get("DEEPSEEK_API_KEY", ""),
                          base_url=values.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
                          model=values.get("DEEPSEEK_MODEL") or "deepseek-flash")

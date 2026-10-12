"""QuipQuill cloud: auth, LLM / embeddings gateway and message queue."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("quipquill-cloud")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0"

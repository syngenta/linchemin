import importlib.metadata

from linchemin.configuration.config import settings

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown version"

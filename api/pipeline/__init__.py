__all__ = [
    "DEFAULT_MODEL_PATH",
    "REGIME_NAMES",
    "SlicedWassersteinRegimePipeline",
]


def __getattr__(name: str):
    if name in __all__:
        from . import regime_pipeline

        return getattr(regime_pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

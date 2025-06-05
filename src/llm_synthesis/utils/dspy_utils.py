import dspy

from llm_synthesis.utils.llms import LLM_REGISTRY, LLMConfig


def get_llm_from_name(llm_name: str, model_kwargs: dict) -> dspy.LM:
    """
    Get a dspy.LM from a given LLM name.

    Args:
        llm_name: The name of the LLM to get. cf. LLM_REGISTRY
        model_kwargs: A dictionary of model kwargs to pass to the LLM.

    Returns:
        A dspy.LM object.
    """
    try:
        cfg: LLMConfig = LLM_REGISTRY.configs[llm_name]
    except KeyError:
        available_models = list(LLM_REGISTRY.configs.keys())
        raise ValueError(
            f"LLM name {llm_name!r} not supported. Available: {available_models}"
        )

    if cfg.api_key:
        model_kwargs["api_key"] = cfg.api_key
        model_kwargs["api_base"] = cfg.api_base

    return dspy.LM(cfg.model, **model_kwargs)


def configure_dspy(lm: str, model_kwargs: dict = {}) -> None:
    """
    Configure dspy with a selected LLM.

    Args:
        lm: LLM key to configure (cf. LLM_REGISTRY).
        model_kwargs: Additional model kwargs (e.g., {"temperature": 0.7}).
    """
    dspy.settings.configure(
        track_usage=True,
        lm=get_llm_from_name(lm, model_kwargs),
    )
    # TODO: add system prompt here
    print(f"Configured dspy with {lm!r} and model_kwargs={model_kwargs}")

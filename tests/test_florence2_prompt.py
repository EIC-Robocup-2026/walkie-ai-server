"""Unit tests for Florence-2 caption-prompt normalization.

Florence-2 only acts on its fixed task tokens; a free-text or unsupported prompt
cannot be honored and, fed to ``generate`` verbatim, yields a generic or even
degenerate caption. The provider guards against this by mapping every
non-token prompt to ``default_prompt`` (see
``Florence2ImageCaptionProvider._normalize_task_prompt``).

These tests exercise only the pure normalization / construction logic — they
construct the provider but never call ``load_model``/``caption``, so no weights
are downloaded or loaded and they run offline without a GPU.
"""

import warnings

import pytest

from services.image_caption.providers.florence2 import (
    Florence2ImageCaptionProvider,
)


def _provider(**config) -> Florence2ImageCaptionProvider:
    """Construct the provider without loading the model (CPU, no weights)."""
    return Florence2ImageCaptionProvider({"variant": "base", "device": "cpu", **config})


# ---------------------------------------------------------------------------
# Prompt pass-through and fallback
# ---------------------------------------------------------------------------

def test_none_prompt_uses_default():
    prov = _provider()
    assert prov._normalize_task_prompt(None) == prov.default_prompt


@pytest.mark.parametrize("token", sorted(Florence2ImageCaptionProvider.CAPTION_TASK_TOKENS))
def test_recognized_task_tokens_pass_through(token):
    prov = _provider()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a passthrough must not warn
        assert prov._normalize_task_prompt(token) == token


def test_surrounding_whitespace_is_tolerated():
    prov = _provider()
    assert prov._normalize_task_prompt("  <CAPTION>\n") == "<CAPTION>"


@pytest.mark.parametrize(
    "bad",
    [
        "Describe the clothing the person is wearing.",  # documented-but-ignored free text
        "What color is the tie?",                        # a question Florence-2 can't answer
        "Describe the person.",                          # per_detection prompt_template output
        "<UNSUPPORTED_TOKEN>",                           # well-formed but not a caption task
        "",                                              # empty string
    ],
)
def test_non_token_prompts_fall_back_and_warn(bad):
    prov = _provider()
    with pytest.warns(UserWarning, match="task token"):
        result = prov._normalize_task_prompt(bad)
    assert result == prov.default_prompt
    assert result in Florence2ImageCaptionProvider.CAPTION_TASK_TOKENS


# ---------------------------------------------------------------------------
# default_prompt config validation
# ---------------------------------------------------------------------------

def test_valid_default_prompt_override_is_accepted():
    prov = _provider(default_prompt="<MORE_DETAILED_CAPTION>")
    assert prov.default_prompt == "<MORE_DETAILED_CAPTION>"
    assert prov._normalize_task_prompt("anything else") == "<MORE_DETAILED_CAPTION>"


@pytest.mark.parametrize("bad_default", ["Describe this image.", "<OD>", ""])
def test_invalid_default_prompt_is_rejected_at_construction(bad_default):
    # The fallback target must itself be a valid token, so a non-token
    # default_prompt is a configuration error, not a silently-broken provider.
    with pytest.raises(ValueError, match="default_prompt"):
        _provider(default_prompt=bad_default)

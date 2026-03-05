"""Unit tests for orchestration flows."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from llm_synthesis.metrics.judge.general_synthesis_judge import (
    GeneralSynthesisEvaluation,
)
from llm_synthesis.models.ontologies import GeneralSynthesisOntology
from llm_synthesis.models.paper import Paper


def _make_paper(name="paper1", paper_id="id1"):
    return Paper(name=name, id=paper_id, publication_text="Some text")


def _make_mock_components():
    """Return mocks for all pipeline components."""
    mock_loader = MagicMock()
    mock_loader.load.return_value = [_make_paper()]

    mock_material_extractor = MagicMock()
    mock_material_extractor.forward.return_value = "Iron oxide"

    mock_synthesis_extractor = MagicMock()
    mock_synthesis_extractor.forward.return_value = MagicMock(
        spec=GeneralSynthesisOntology,
        model_dump=lambda: {},
    )

    mock_judge = MagicMock()
    mock_judge.forward.return_value = MagicMock(
        spec=GeneralSynthesisEvaluation
    )

    mock_result_gather = MagicMock()

    return (
        mock_loader,
        mock_material_extractor,
        mock_synthesis_extractor,
        mock_judge,
        mock_result_gather,
    )


def _make_config(result_dir="/tmp/results"):
    return {
        "data_loader": {"architecture": {"_target_": "mock.loader"}},
        "material_extraction": {
            "architecture": {"_target_": "mock.material"}
        },
        "synthesis_extraction": {
            "architecture": {"_target_": "mock.synthesis"}
        },
        "judge": {"architecture": {"_target_": "mock.judge"}},
        "result_save": {
            "architecture": {
                "_target_": "mock.result",
                "result_dir": result_dir,
            }
        },
        "orchestration": {
            "max_workers": 1,
            "retries": {
                "material_extraction": 3,
                "synthesis_extraction": 3,
                "judge_evaluation": 2,
                "result_save": 1,
            },
            "retry_delay_seconds": 5,
        },
    }


@pytest.mark.unit
def test_flow_loads_papers_and_saves_results(tmp_path):
    from llm_synthesis.orchestration.flows import synthesis_extraction_flow

    (
        mock_loader,
        mock_material_extractor,
        mock_synthesis_extractor,
        mock_judge,
        mock_result_gather,
    ) = _make_mock_components()

    config = _make_config(result_dir=str(tmp_path))

    with patch("hydra.utils.instantiate") as mock_instantiate:
        mock_instantiate.side_effect = [
            mock_loader,
            mock_material_extractor,
            mock_synthesis_extractor,
            mock_judge,
            mock_result_gather,
        ]

        synthesis_extraction_flow(config)

    mock_loader.load.assert_called_once()
    mock_material_extractor.forward.assert_called_once()
    mock_result_gather.gather.assert_called_once()


@pytest.mark.unit
def test_flow_skips_paper_with_no_materials(tmp_path):
    from llm_synthesis.orchestration.flows import synthesis_extraction_flow

    (
        mock_loader,
        mock_material_extractor,
        mock_synthesis_extractor,
        mock_judge,
        mock_result_gather,
    ) = _make_mock_components()
    mock_material_extractor.forward.return_value = ""  # no materials

    config = _make_config(result_dir=str(tmp_path))

    with patch("hydra.utils.instantiate") as mock_instantiate:
        mock_instantiate.side_effect = [
            mock_loader,
            mock_material_extractor,
            mock_synthesis_extractor,
            mock_judge,
            mock_result_gather,
        ]

        synthesis_extraction_flow(config)

    mock_synthesis_extractor.forward.assert_not_called()
    mock_result_gather.gather.assert_not_called()


@pytest.mark.unit
def test_flow_skips_already_processed_papers(tmp_path):
    from llm_synthesis.orchestration.flows import synthesis_extraction_flow

    paper = _make_paper(paper_id="id1")
    # Create a file named "id1" in result_dir to simulate already processed
    (tmp_path / "id1").mkdir()

    (
        mock_loader,
        mock_material_extractor,
        mock_synthesis_extractor,
        mock_judge,
        mock_result_gather,
    ) = _make_mock_components()
    mock_loader.load.return_value = [paper]

    config = _make_config(result_dir=str(tmp_path))

    with patch("hydra.utils.instantiate") as mock_instantiate:
        mock_instantiate.side_effect = [
            mock_loader,
            mock_material_extractor,
            mock_synthesis_extractor,
            mock_judge,
            mock_result_gather,
        ]

        synthesis_extraction_flow(config)

    mock_material_extractor.forward.assert_not_called()
    mock_result_gather.gather.assert_not_called()


@pytest.mark.unit
def test_system_prefixed_lm_cost_thread_safety():
    """Verify threading.Lock prevents cost corruption under concurrent access."""
    from llm_synthesis.utils.llms import SystemPrefixedLM

    lm = SystemPrefixedLM.__new__(SystemPrefixedLM)
    lm._cumulative_cost_usd = 0.0
    lm._cost_lock = threading.Lock()

    increments = 1000
    delta = 0.001

    def increment_cost():
        for _ in range(increments):
            with lm._cost_lock:
                lm._cumulative_cost_usd += delta

    threads = [threading.Thread(target=increment_cost) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    expected = 10 * increments * delta
    assert abs(lm._cumulative_cost_usd - expected) < 1e-9

"""Unit tests for orchestration tasks."""

from unittest.mock import MagicMock

import pytest

from llm_synthesis.models.ontologies import GeneralSynthesisOntology
from llm_synthesis.models.paper import Paper, PaperWithSynthesisOntologies


@pytest.mark.unit
def test_load_papers_returns_list_of_papers():
    from llm_synthesis.orchestration.tasks import load_papers

    mock_loader = MagicMock()
    papers = [
        Paper(name="paper1", id="id1", publication_text="text1"),
        Paper(name="paper2", id="id2", publication_text="text2"),
    ]
    mock_loader.load.return_value = papers

    result = load_papers(mock_loader)

    assert result == papers
    mock_loader.load.assert_called_once()


@pytest.mark.unit
def test_extract_materials_returns_list_of_strings():
    from llm_synthesis.orchestration.tasks import extract_materials

    mock_extractor = MagicMock()
    mock_extractor.forward.return_value = "Iron oxide, Nickel nitrate, Water"

    result = extract_materials(mock_extractor, "some paper text")

    assert result == ["Iron oxide", "Nickel nitrate", "Water"]
    mock_extractor.forward.assert_called_once_with(input="some paper text")


@pytest.mark.unit
def test_extract_materials_handles_newline_separated():
    from llm_synthesis.orchestration.tasks import extract_materials

    mock_extractor = MagicMock()
    mock_extractor.forward.return_value = "Iron oxide\nNickel nitrate\nWater"

    result = extract_materials(mock_extractor, "some paper text")

    assert result == ["Iron oxide", "Nickel nitrate", "Water"]


@pytest.mark.unit
def test_extract_materials_returns_empty_list_for_empty_result():
    from llm_synthesis.orchestration.tasks import extract_materials

    mock_extractor = MagicMock()
    mock_extractor.forward.return_value = ""

    result = extract_materials(mock_extractor, "some paper text")

    assert result == []


@pytest.mark.unit
def test_extract_synthesis_returns_ontology():
    from llm_synthesis.orchestration.tasks import extract_synthesis

    mock_extractor = MagicMock()
    ontology = MagicMock(spec=GeneralSynthesisOntology)
    mock_extractor.forward.return_value = ontology

    result = extract_synthesis(mock_extractor, "paper text", "Iron oxide")

    assert result is ontology
    mock_extractor.forward.assert_called_once_with(
        input=("paper text", "Iron oxide")
    )


@pytest.mark.unit
def test_evaluate_synthesis_returns_evaluation():
    from llm_synthesis.orchestration.tasks import evaluate_synthesis

    mock_judge = MagicMock()
    evaluation = {"score": 0.9}
    mock_judge.forward.return_value = evaluation

    result = evaluate_synthesis(
        mock_judge, "paper text", '{"synthesis": "..."}', "Iron oxide"
    )

    assert result == evaluation
    mock_judge.forward.assert_called_once_with(
        ("paper text", '{"synthesis": "..."}', "Iron oxide")
    )


@pytest.mark.unit
def test_save_paper_results_calls_gather():
    from llm_synthesis.orchestration.tasks import save_paper_results

    mock_result_gather = MagicMock()
    paper = PaperWithSynthesisOntologies(
        name="paper1",
        id="id1",
        publication_text="text",
        all_syntheses=[],
    )

    save_paper_results(mock_result_gather, paper)

    mock_result_gather.gather.assert_called_once_with(paper)

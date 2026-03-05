"""Prefect tasks for llm_synthesis pipeline steps."""

from prefect import get_run_logger, task

from llm_synthesis.data_loader.paper_loader.base import PaperLoaderInterface
from llm_synthesis.metrics.judge.base import SynthesisJudgeInterface
from llm_synthesis.models.ontologies import GeneralSynthesisOntology
from llm_synthesis.models.paper import Paper, PaperWithSynthesisOntologies
from llm_synthesis.result_gather.base import ResultGatherInterface
from llm_synthesis.transformers.base import ExtractorInterface
from llm_synthesis.utils.markdown_utils import clean_text


@task
def load_papers(data_loader: PaperLoaderInterface) -> list[Paper]:
    """Load papers from the configured data source."""
    logger = get_run_logger()
    papers = data_loader.load()
    logger.info(f"Loaded {len(papers)} papers")
    return papers


@task(retries=3, retry_delay_seconds=5)
def extract_materials(
    extractor: ExtractorInterface[str, str],
    paper_text: str,
) -> list[str]:
    """Extract material names from paper text."""
    logger = get_run_logger()
    result: str = extractor.forward(input=clean_text(paper_text))
    if not result:
        return []
    materials = [
        m.strip()
        for m in result.replace("\n", ",").split(",")
        if m.strip()
    ]
    logger.info(f"Extracted {len(materials)} materials")
    return materials


@task(retries=3, retry_delay_seconds=5)
def extract_synthesis(
    extractor: ExtractorInterface[tuple[str, str], GeneralSynthesisOntology],
    paper_text: str,
    material: str,
) -> GeneralSynthesisOntology:
    """Extract synthesis procedure for a given material."""
    logger = get_run_logger()
    result = extractor.forward(input=(clean_text(paper_text), material))
    logger.info(f"Extracted synthesis for material: {material}")
    return result


@task(retries=2, retry_delay_seconds=5)
def evaluate_synthesis(
    judge: SynthesisJudgeInterface,
    paper_text: str,
    synthesis_json: str,
    material: str,
):
    """Evaluate extracted synthesis using the judge."""
    logger = get_run_logger()
    result = judge.forward((clean_text(paper_text), synthesis_json, material))
    logger.info(f"Evaluated synthesis for material: {material}")
    return result


@task(retries=1, retry_delay_seconds=5)
def save_paper_results(
    result_gather: ResultGatherInterface[PaperWithSynthesisOntologies],
    paper_with_results: PaperWithSynthesisOntologies,
) -> None:
    """Persist paper extraction results."""
    logger = get_run_logger()
    result_gather.gather(paper_with_results)
    logger.info(f"Saved results for paper: {paper_with_results.name}")

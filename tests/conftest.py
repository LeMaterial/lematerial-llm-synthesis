"""Shared test fixtures for orchestration tests."""

import pytest
from prefect.testing.utilities import prefect_test_harness


@pytest.fixture(autouse=True)
def prefect_harness():
    with prefect_test_harness():
        yield

"""Shared pytest fixtures for the SMART test suite."""

import pytest


@pytest.fixture(scope="session")
def tiny_problem():
    """A small synthetic low-rank regression instance used across tests."""
    from smart import generate_data

    return generate_data(n=200, p=40, q=20, sigma0=0.01,
                         r_star=3, r0_star=6, random_seed=0)

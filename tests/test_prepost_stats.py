"""Checks the statistics against published reference output.

Run with:  pytest
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepost_stats import analyse, apa_text, clean_pairs, format_p  # noqa: E402

# Student's (1908) sleep data, as in R's datasets::sleep.
# R: t.test(extra ~ group, data = sleep, paired = TRUE)  ->  t = -4.0621, df = 9, p-value = 0.002833,
#    95% CI -2.4598858 -0.7001142, mean difference -1.58  (group 1 minus group 2)
SLEEP_1 = [0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0]
SLEEP_2 = [1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4]


def test_matches_r_sleep_data():
    r = analyse(SLEEP_1, SLEEP_2)  # post - pre = group 2 - group 1
    assert r.df == 9
    assert r.t == pytest.approx(4.0621, abs=1e-4)
    assert r.p == pytest.approx(0.002833, abs=1e-6)
    assert r.mean_diff == pytest.approx(1.58)
    assert r.ci_low == pytest.approx(0.7001142, abs=1e-6)
    assert r.ci_high == pytest.approx(2.4598858, abs=1e-6)
    assert r.effect_sizes["dz"].value == pytest.approx(1.58 / np.std(np.subtract(SLEEP_2, SLEEP_1), ddof=1))


def test_sample_file_reference_values():
    d = pd.read_csv(Path(__file__).resolve().parents[1] / "sample_data.csv")
    r = analyse(d.Pre, d.Post)
    assert r.n == 30 and r.mean_diff == pytest.approx(7.0)
    assert r.t == pytest.approx(10.942906, abs=1e-6)
    es = r.effect_sizes["dz"]
    assert es.value == pytest.approx(1.997892, abs=1e-6)
    # exact noncentral-t interval for d_z
    assert es.ci_low == pytest.approx(1.367597, abs=1e-5)
    assert es.ci_high == pytest.approx(2.616256, abs=1e-5)
    assert r.shapiro_w == pytest.approx(0.985154, abs=1e-6)
    assert r.shapiro_p == pytest.approx(0.939779, abs=1e-5)
    assert es.size_label == "Large"


def test_hedges_g_is_smaller_than_dz():
    r = analyse(SLEEP_1, SLEEP_2)
    j = r.effect_sizes["gz"].value / r.effect_sizes["dz"].value
    assert j == pytest.approx(1 - 3 / (4 * 9 - 1), abs=0.01)  # Hedges' approximation
    assert j < 1


def test_one_tailed_is_half_of_two_tailed_when_gain_positive():
    two = analyse(SLEEP_1, SLEEP_2, tails=2)
    one = analyse(SLEEP_1, SLEEP_2, tails=1)
    assert one.p == pytest.approx(two.p / 2)


def test_listwise_deletion():
    pre, post, excluded = clean_pairs(["10", "12", "", "abs", "15", ""], ["14", "13", "11", "16", "", ""])
    assert list(pre) == [10, 12] and list(post) == [14, 13]
    assert excluded == 3  # last row is completely blank and is not counted


def test_constant_differences_rejected():
    with pytest.raises(ValueError):
        analyse([1, 2, 3, 4], [2, 3, 4, 5])


def test_apa_text_and_p_format():
    r = analyse(SLEEP_1, SLEEP_2)
    s = apa_text(r)
    assert "t(9) = 4.06" in s and "p = .003" in s and "95% CI [0.70, 2.46]" in s
    assert format_p(0.0004) == "< .001" and format_p(0.05) == ".050"
    assert not math.isnan(r.r)


@pytest.mark.parametrize("n", [604, 2000, 5000])
def test_large_samples_give_finite_effect_size_ci(n):
    # Regression test: SciPy's noncentral t returns NaN far from the answer for large samples.
    rng = np.random.default_rng(n)
    pre = rng.integers(0, 60, n).astype(float)
    post = np.clip(pre + rng.normal(4, 10, n).round(), 0, 100)
    r = analyse(pre, post, alpha=0.01)
    es = r.effect_sizes["dz"]
    assert np.isfinite(es.ci_low) and np.isfinite(es.ci_high)
    # for large n the exact interval is close to the normal approximation
    se = np.sqrt(1 / n + es.value ** 2 / (2 * n))
    assert es.ci_low == pytest.approx(es.value - 2.5758 * se, abs=0.01)
    assert es.ci_high == pytest.approx(es.value + 2.5758 * se, abs=0.01)

"""Statistics for pre-test / post-test (paired) designs.

All calculations use NumPy and SciPy, and follow the conventions of SPSS, JASP and R:
- Paired-samples t-test on the differences (post - pre).
- Cohen's d_z (SPSS/JASP default for paired data) with an exact noncentral-t confidence interval.
- Hedges' g_z (small-sample corrected d_z), Cohen's d_av and Cohen's d with pooled SD.
- Shapiro-Wilk test of normality on the differences (Royston, 1995).
- Wilcoxon signed-rank test as the non-parametric alternative (zero differences dropped,
  normal approximation with tie correction, no continuity correction), with r = |Z| / sqrt(N).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, special, stats


@dataclass
class Descriptives:
    n: int
    mean: float
    sd: float
    se: float
    min: float
    max: float


@dataclass
class EffectSize:
    key: str
    name: str
    value: float
    ci_low: float | None
    ci_high: float | None
    standardiser: str

    @property
    def size_label(self) -> str:
        return cohen_label(self.value)


@dataclass
class PairedResult:
    n: int
    alpha: float
    tails: int
    pre: Descriptives
    post: Descriptives
    r: float
    r_p: float
    mean_diff: float
    sd_diff: float
    se_diff: float
    ci_low: float
    ci_high: float
    t: float
    df: int
    p: float
    effect_sizes: dict[str, EffectSize] = field(default_factory=dict)
    shapiro_w: float | None = None
    shapiro_p: float | None = None
    wilcoxon_z: float | None = None
    wilcoxon_p: float | None = None
    wilcoxon_r: float | None = None
    wilcoxon_zeros: int = 0
    excluded_rows: int = 0

    @property
    def conf_level(self) -> int:
        return round((1 - self.alpha) * 100)

    @property
    def significant(self) -> bool:
        return self.p < self.alpha

    @property
    def normal(self) -> bool | None:
        return None if self.shapiro_p is None else self.shapiro_p >= 0.05


def cohen_label(d: float) -> str:
    a = abs(d)
    if a < 0.2:
        return "Negligible"
    if a < 0.5:
        return "Small"
    if a < 0.8:
        return "Medium"
    return "Large"


def _describe(x: np.ndarray) -> Descriptives:
    n = len(x)
    sd = float(np.std(x, ddof=1))
    return Descriptives(n, float(np.mean(x)), sd, sd / np.sqrt(n), float(np.min(x)), float(np.max(x)))


def _ncp_bound(t_obs: float, df: int, target: float) -> float:
    """Noncentrality parameter at which P(T <= t_obs) equals target."""
    def f(nc: float) -> float:
        v = stats.nct.cdf(t_obs, df, nc)
        if not np.isfinite(v):
            # SciPy can return NaN far out in the tails (large samples); the CDF there is ~1 or ~0.
            v = 1.0 if nc < t_obs else 0.0
        return v - target

    # The bound lies roughly z * sqrt(1 + t^2 / (2 df)) from t, so search close to t and widen only if needed.
    step = np.sqrt(1 + t_obs ** 2 / (2 * df))
    lo, hi = t_obs - 4 * step, t_obs + 4 * step
    for _ in range(60):
        if f(lo) > 0:
            break
        lo -= 2 * step
    for _ in range(60):
        if f(hi) < 0:
            break
        hi += 2 * step
    return optimize.brentq(f, lo, hi, xtol=1e-12)


def hedges_j(df: int) -> float:
    return float(np.exp(special.gammaln(df / 2) - np.log(np.sqrt(df / 2)) - special.gammaln((df - 1) / 2)))


def clean_pairs(pre, post) -> tuple[np.ndarray, np.ndarray, int]:
    """Convert to numbers and drop rows where either score is missing (listwise deletion)."""
    import pandas as pd

    raw_a, raw_b = pd.Series(pre).reset_index(drop=True), pd.Series(post).reset_index(drop=True)
    blank = lambda s: s.isna() | s.astype(str).str.strip().eq("")
    a = pd.to_numeric(raw_a, errors="coerce")
    b = pd.to_numeric(raw_b, errors="coerce")
    keep = a.notna() & b.notna()
    excluded = int((~keep & ~(blank(raw_a) & blank(raw_b))).sum())  # fully empty rows are not counted
    return a[keep].to_numpy(float), b[keep].to_numpy(float), excluded


def analyse(pre, post, alpha: float = 0.05, tails: int = 2) -> PairedResult:
    """Run the full pre/post analysis.

    tails = 2 for a two-tailed test (pre != post), 1 for one-tailed (post > pre).
    """
    pre = np.asarray(pre, float)
    post = np.asarray(post, float)
    if pre.shape != post.shape:
        raise ValueError("Pre and post must have the same number of scores.")
    n = len(pre)
    if n < 3:
        raise ValueError("At least 3 students with both scores are needed.")
    diff = post - pre
    if np.ptp(diff) == 0:
        raise ValueError("Every student changed by exactly the same amount, so the differences have no spread "
                         "and the t-test cannot be calculated.")

    df = n - 1
    md = float(diff.mean())
    sdd = float(diff.std(ddof=1))
    se = sdd / np.sqrt(n)
    t = md / se
    p = float(2 * stats.t.sf(abs(t), df)) if tails == 2 else float(stats.t.sf(t, df))
    tcrit = float(stats.t.ppf(1 - alpha / 2, df))

    d_pre, d_post = _describe(pre), _describe(post)
    if d_pre.sd > 0 and d_post.sd > 0:
        r, r_p = stats.pearsonr(pre, post)
        r, r_p = float(r), float(r_p)
    else:
        r, r_p = float("nan"), float("nan")

    # Effect sizes
    j = hedges_j(df)
    dz = md / sdd
    dz_lo = _ncp_bound(t, df, 1 - alpha / 2) / np.sqrt(n)
    dz_hi = _ncp_bound(t, df, alpha / 2) / np.sqrt(n)
    dav = md / ((d_pre.sd + d_post.sd) / 2)
    dpool = md / np.sqrt((d_pre.sd ** 2 + d_post.sd ** 2) / 2)
    es = {
        "dz": EffectSize("dz", "Cohen's d_z", dz, dz_lo, dz_hi, "SD of the differences"),
        "gz": EffectSize("gz", "Hedges' g_z", dz * j, dz_lo * j, dz_hi * j, "d_z × small-sample correction J"),
        "dav": EffectSize("dav", "Cohen's d_av", dav, None, None, "Average of pre and post SD"),
        "dpool": EffectSize("dpool", "Cohen's d (pooled SD)", dpool, None, None, "√((SD²pre + SD²post) / 2)"),
    }

    res = PairedResult(n=n, alpha=alpha, tails=tails, pre=d_pre, post=d_post, r=r, r_p=r_p,
                       mean_diff=md, sd_diff=sdd, se_diff=se, ci_low=md - tcrit * se, ci_high=md + tcrit * se,
                       t=t, df=df, p=p, effect_sizes=es)

    # Assumption check
    if n <= 5000:
        sw = stats.shapiro(diff)
        res.shapiro_w, res.shapiro_p = float(sw.statistic), float(sw.pvalue)

    nonzero = diff[diff != 0]
    res.wilcoxon_zeros = int(n - len(nonzero))
    if len(nonzero) > 0:
        ranks = stats.rankdata(np.abs(nonzero))
        w_plus = ranks[nonzero > 0].sum()
        m = len(nonzero)
        _, counts = np.unique(np.abs(nonzero), return_counts=True)
        var = m * (m + 1) * (2 * m + 1) / 24 - (counts ** 3 - counts).sum() / 48
        z = (w_plus - m * (m + 1) / 4) / np.sqrt(var) if var > 0 else 0.0
        res.wilcoxon_z = float(z)
        res.wilcoxon_p = float(2 * stats.norm.sf(abs(z))) if tails == 2 else float(stats.norm.sf(z))
        res.wilcoxon_r = float(abs(z) / np.sqrt(n))
    return res


def format_p(p: float) -> str:
    """APA style: no leading zero, '< .001' for very small values."""
    if p < 0.001:
        return "< .001"
    return f"{p:.3f}".lstrip("0")


def apa_text(res: PairedResult, es_key: str = "dz", pre_label: str = "pre-test", post_label: str = "post-test") -> str:
    es = res.effect_sizes[es_key]
    tail = "two-tailed" if res.tails == 2 else "one-tailed"
    p_txt = "p < .001" if res.p < 0.001 else f"p = {format_p(res.p)}"
    direction = "higher" if res.mean_diff > 0 else "lower"
    pre_l, post_l = pre_label.lower(), post_label.lower()
    pre_desc = f"(M = {res.pre.mean:.2f}, SD = {res.pre.sd:.2f})"
    post_desc = f"(M = {res.post.mean:.2f}, SD = {res.post.sd:.2f})"
    s = f"A paired-samples t-test ({tail}) was conducted to compare {pre_l} and {post_l} scores (N = {res.n}). "
    if res.significant:
        s += f"{post_label} scores {post_desc} were significantly {direction} than {pre_l} scores {pre_desc}, "
    else:
        s += f"There was no significant difference between {post_l} scores {post_desc} and {pre_l} scores {pre_desc}, "
    s += (f"t({res.df}) = {res.t:.2f}, {p_txt}, mean difference = {res.mean_diff:.2f}, "
          f"{res.conf_level}% CI [{res.ci_low:.2f}, {res.ci_high:.2f}]. ")
    sym = {"dz": "d", "gz": "g", "dav": "d_av", "dpool": "d"}[es_key]
    s += f"The effect size was {es.size_label.lower()}, {sym} = {es.value:.2f}"
    if es.ci_low is not None:
        s += f", {res.conf_level}% CI [{es.ci_low:.2f}, {es.ci_high:.2f}]"
    return s + "."

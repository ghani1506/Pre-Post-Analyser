"""Pre–Post Analyser: paired-samples t-test and Cohen's d for pre-test / post-test data.

Run locally:   streamlit run app.py
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from scipy import stats as sp_stats

from prepost_stats import analyse, apa_text, clean_pairs, format_p

st.set_page_config(page_title="Pre–Post Analyser", page_icon="📈", layout="wide")

SAMPLE_PATH = Path(__file__).parent / "sample_data.csv"
ES_OPTIONS = {
    "dz": "Cohen's d_z (SPSS / JASP default)",
    "gz": "Hedges' g_z (small samples)",
    "dav": "Cohen's d_av",
    "dpool": "Cohen's d (pooled SD)",
}


# ---------- data loading ----------
@st.cache_data(show_spinner=False)
def read_upload(name: str, data: bytes, sheet: str | None) -> pd.DataFrame:
    if name.lower().endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(io.BytesIO(data), sheet_name=sheet or 0)
    text = data.decode("utf-8-sig", errors="replace")
    sep = "\t" if "\t" in text.splitlines()[0] else None
    return pd.read_csv(io.StringIO(text), sep=sep, engine="python")


@st.cache_data(show_spinner=False)
def excel_sheets(data: bytes) -> list[str]:
    return pd.ExcelFile(io.BytesIO(data)).sheet_names


def numeric_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        s = df[c].dropna()
        if len(s) and pd.to_numeric(s, errors="coerce").notna().mean() >= 0.6:
            cols.append(c)
    return cols


def guess(cols: list[str], pattern: str, fallback: int) -> int:
    for i, c in enumerate(cols):
        if re.search(pattern, str(c), re.I):
            return i
    return min(fallback, len(cols) - 1)


# ---------- sidebar ----------
with st.sidebar:
    st.header("1 · Data")
    source = st.radio("Data source", ["Upload a file", "Type or paste", "Example data"], index=2,
                      help="Upload a CSV or Excel file with one row per student and a header row.")
    df: pd.DataFrame | None = None
    if source == "Upload a file":
        up = st.file_uploader("CSV or Excel file", type=["csv", "txt", "tsv", "xlsx", "xlsm", "xls"])
        if up is not None:
            raw = up.getvalue()
            sheet = None
            if up.name.lower().endswith((".xlsx", ".xlsm", ".xls")):
                sheets = excel_sheets(raw)
                sheet = st.selectbox("Sheet", sheets) if len(sheets) > 1 else sheets[0]
            try:
                df = read_upload(up.name, raw, sheet)
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read the file: {e}")
    elif source == "Type or paste":
        st.caption("Click a cell and paste straight from Excel. Add rows at the bottom of the table.")
        base = pd.DataFrame({"Student": [""] * 5, "Pre": [np.nan] * 5, "Post": [np.nan] * 5})
        df = st.data_editor(base, num_rows="dynamic", use_container_width=True, key="editor")
    else:
        df = pd.read_csv(SAMPLE_PATH)
        st.info("Showing example data for 30 students. Choose another data source to use your own scores.")

    pre_col = post_col = None
    if df is not None and len(df.columns):
        cols = numeric_columns(df)
        if len(cols) >= 2:
            pre_col = st.selectbox("Pre-test column", cols, index=guess(cols, r"pre|before|sebelum|test ?1|ujian ?1", 0))
            post_col = st.selectbox("Post-test column", cols, index=guess(cols, r"post|after|selepas|test ?2|ujian ?2", 1))

    st.header("2 · Settings")
    tails = st.radio("Hypothesis", [2, 1], format_func=lambda v: "Two-tailed (pre ≠ post)" if v == 2 else "One-tailed (post > pre)")
    alpha = st.selectbox("Significance level α", [0.05, 0.01, 0.10], format_func=lambda a: f"{a:.2f}  ({round((1 - a) * 100)}% CI)")
    es_key = st.selectbox("Effect size for the write-up", list(ES_OPTIONS), format_func=ES_OPTIONS.get)
    pre_label = st.text_input("Label for pre", "Pre-test")
    post_label = st.text_input("Label for post", "Post-test")

# ---------- header ----------
st.title("Pre–Post Analyser")
st.caption("Paired-samples t-test and Cohen's d for pre-test and post-test scores, with a normality check, "
           "a non-parametric backup and an APA write-up. Calculated with SciPy, matching SPSS, JASP and R.")

if df is None:
    st.info("Upload a CSV or Excel file in the sidebar to begin.")
    st.stop()
if pre_col is None and source == "Type or paste":
    st.info("Type or paste your pre-test and post-test scores into the table in the sidebar.")
    st.stop()
if pre_col is None:
    st.warning("I need at least two columns of numbers: one for the pre-test and one for the post-test.")
    st.stop()
if pre_col == post_col:
    st.warning("Choose two different columns for the pre-test and post-test.")
    st.stop()

pre, post, excluded = clean_pairs(df[pre_col], df[post_col])
try:
    res = analyse(pre, post, alpha=alpha, tails=tails)
except ValueError as e:
    st.error(str(e))
    st.stop()
res.excluded_rows = excluded
if excluded:
    st.warning(f"{excluded} row{'s were' if excluded > 1 else ' was'} left out because a pre or post score was "
               "blank or not a number (listwise deletion).")

ci = res.conf_level
es = res.effect_sizes[es_key]
tail_txt = "two-tailed" if tails == 2 else "one-tailed"
p_disp = format_p(res.p)

# ---------- headline ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Result", "Significant" if res.significant else "Not significant", f"α = {alpha}", delta_color="off")
c2.metric(f"t({res.df})", f"{res.t:.2f}", f"p {p_disp if p_disp.startswith('<') else '= ' + p_disp}", delta_color="off")
c3.metric("Mean gain", f"{res.mean_diff:+.2f}", f"{ci}% CI {res.ci_low:.2f} to {res.ci_high:.2f}", delta_color="off")
c4.metric(es.name, f"{es.value:.2f}", es.size_label, delta_color="off")

tab_res, tab_chart, tab_assume, tab_apa, tab_data, tab_method = st.tabs(
    ["Results", "Chart", "Assumptions", "Write-up", "Data used", "Method"])

# ---------- tables ----------
desc = pd.DataFrame(
    [[pre_label, res.n, res.pre.mean, res.pre.sd, res.pre.se, res.pre.min, res.pre.max],
     [post_label, res.n, res.post.mean, res.post.sd, res.post.se, res.post.min, res.post.max]],
    columns=["", "N", "Mean", "SD", "SE", "Min", "Max"]).set_index("")
ttest = pd.DataFrame([[f"{post_label} − {pre_label}", res.mean_diff, res.sd_diff, res.se_diff, res.ci_low, res.ci_high,
                       res.t, res.df, p_disp]],
                     columns=["", "Mean diff.", "SD diff.", "SE", f"{ci}% CI lower", f"{ci}% CI upper", "t", "df",
                              f"p ({tail_txt})"]).set_index("")
eff = pd.DataFrame([[e.name, e.value, e.ci_low, e.ci_high, e.size_label, e.standardiser] for e in res.effect_sizes.values()],
                   columns=["Measure", "Value", f"{ci}% CI lower", f"{ci}% CI upper", "Size", "Standardised by"]).set_index("Measure")
assume = pd.DataFrame(
    [["Shapiro–Wilk (normality of differences)", "W", res.shapiro_w, res.shapiro_p,
      "—" if res.normal is None else ("Normal enough" if res.normal else "Not normal")],
     ["Wilcoxon signed-rank (non-parametric)", "Z", res.wilcoxon_z, res.wilcoxon_p,
      "—" if res.wilcoxon_r is None else f"r = {res.wilcoxon_r:.2f}"]],
    columns=["Test", "Statistic", "Value", "p", "Reading"]).set_index("Test")

with tab_res:
    st.subheader("Descriptive statistics")
    st.dataframe(desc.style.format({"Mean": "{:.2f}", "SD": "{:.2f}", "SE": "{:.2f}", "Min": "{:g}", "Max": "{:g}"}),
                 use_container_width=True)
    if not np.isnan(res.r):
        st.caption(f"Correlation between {pre_label.lower()} and {post_label.lower()}: "
                   f"r = {res.r:.3f}, p {format_p(res.r_p) if res.r_p < 0.001 else '= ' + format_p(res.r_p)}")
    st.subheader(f"Paired-samples t-test ({post_label} − {pre_label})")
    st.dataframe(ttest.style.format({"Mean diff.": "{:.2f}", "SD diff.": "{:.2f}", "SE": "{:.2f}",
                                     f"{ci}% CI lower": "{:.2f}", f"{ci}% CI upper": "{:.2f}", "t": "{:.3f}"}),
                 use_container_width=True)
    st.subheader("Effect sizes")
    st.dataframe(eff.style.format({"Value": "{:.3f}", f"{ci}% CI lower": "{:.2f}", f"{ci}% CI upper": "{:.2f}"}, na_rep="—"),
                 use_container_width=True)
    st.caption("Cohen (1988) benchmarks: 0.2 small, 0.5 medium, 0.8 large. The d_z interval uses the exact "
               "noncentral t method. Always state which d you report, because they are not interchangeable.")

with tab_chart:
    long = pd.DataFrame({"Student": np.repeat(np.arange(res.n), 2),
                         "Time": [pre_label, post_label] * res.n,
                         "Score": np.column_stack([pre, post]).ravel()})
    order = [pre_label, post_label]
    lines = alt.Chart(long).mark_line(opacity=0.25 if res.n <= 60 else 0.1, color="#7a8794").encode(
        x=alt.X("Time:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Score:Q", scale=alt.Scale(zero=False)), detail="Student:N")
    tc = float(sp_stats.t.ppf(1 - alpha / 2, res.n - 1))
    means = pd.DataFrame({"Time": order, "Mean": [res.pre.mean, res.post.mean],
                          "Lo": [res.pre.mean - tc * res.pre.se, res.post.mean - tc * res.post.se],
                          "Hi": [res.pre.mean + tc * res.pre.se, res.post.mean + tc * res.post.se]})
    mline = alt.Chart(means).mark_line(strokeWidth=3, color="#0f5e7a").encode(x=alt.X("Time:N", sort=order), y="Mean:Q")
    mpts = alt.Chart(means).mark_point(size=90, filled=True, color="#0f5e7a").encode(
        x=alt.X("Time:N", sort=order), y="Mean:Q", tooltip=[alt.Tooltip("Mean:Q", format=".2f")])
    bars = alt.Chart(means).mark_errorbar(color="#0f5e7a", thickness=2).encode(
        x=alt.X("Time:N", sort=order), y=alt.Y("Lo:Q", title="Score"), y2="Hi:Q")
    st.altair_chart((lines + bars + mline + mpts).properties(height=380), use_container_width=True)
    st.caption(f"Grey lines are individual students. The blue line shows the means with {ci}% confidence intervals.")

    gains = pd.DataFrame({"Gain": post - pre})
    hist = alt.Chart(gains).mark_bar(color="#0f8a72").encode(
        x=alt.X("Gain:Q", bin=alt.Bin(maxbins=20), title=f"{post_label} − {pre_label}"), y=alt.Y("count()", title="Students"))
    st.altair_chart(hist.properties(height=220, title="Distribution of gains"), use_container_width=True)

with tab_assume:
    st.dataframe(assume.style.format({"Value": "{:.3f}", "p": "{:.3f}"}, na_rep="—"), use_container_width=True)
    if res.normal is False:
        if res.n >= 30:
            st.warning(f"The differences are not normally distributed, but with {res.n} students the t-test is usually "
                       "robust. Report the Wilcoxon result alongside it as a check.")
        else:
            st.warning(f"The differences are not normally distributed and the sample is small ({res.n}). Report the "
                       "Wilcoxon signed-rank test as your main result, or both.")
    else:
        st.success("The differences look normally distributed, so the paired t-test is appropriate.")
    if res.wilcoxon_zeros:
        st.caption(f"Wilcoxon left out {res.wilcoxon_zeros} student{'s' if res.wilcoxon_zeros > 1 else ''} with no change.")

text = apa_text(res, es_key, pre_label, post_label)
with tab_apa:
    st.subheader("APA-style write-up")
    st.write(text)
    st.code(text, language=None, wrap_lines=True)
    st.caption("Use the copy icon on the box above, then italicise the statistical symbols (t, p, M, SD, d) in Word.")

with tab_data:
    st.dataframe(pd.DataFrame({pre_label: pre, post_label: post, "Gain": post - pre}), use_container_width=True)

with tab_method:
    st.markdown(
        """
Differences are **post − pre** for each student, so a positive value means improvement.

| Statistic | Formula |
|---|---|
| Paired *t* | t = M_diff / (SD_diff / √n), df = n − 1 |
| Cohen's *d_z* | M_diff / SD_diff |
| Hedges' *g_z* | d_z × J, J = Γ(df/2) / (√(df/2) · Γ((df−1)/2)) |
| Cohen's *d_av* | M_diff / ((SD_pre + SD_post) / 2) |
| Cohen's *d* (pooled) | M_diff / √((SD²_pre + SD²_post) / 2) |

Shapiro–Wilk uses Royston's (1995) algorithm via `scipy.stats.shapiro`, the same algorithm used by SPSS and R.
Wilcoxon drops zero differences and uses the normal approximation with a tie correction; r = |Z| / √N.

**References**
- Cohen, J. (1988). *Statistical power analysis for the behavioral sciences* (2nd ed.). Erlbaum.
- Lakens, D. (2013). Calculating and reporting effect sizes to facilitate cumulative science. *Frontiers in Psychology, 4*, 863.
- Hedges, L. V., & Olkin, I. (1985). *Statistical methods for meta-analysis*. Academic Press.
- Royston, P. (1995). Remark AS R94. *Applied Statistics, 44*(4), 547–551.
"""
    )

# ---------- downloads ----------
def results_excel() -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        desc.to_excel(xw, sheet_name="Descriptives")
        ttest.to_excel(xw, sheet_name="Paired t-test")
        eff.to_excel(xw, sheet_name="Effect sizes")
        assume.to_excel(xw, sheet_name="Assumptions")
        pd.DataFrame({"APA write-up": [text]}).to_excel(xw, sheet_name="Write-up", index=False)
        pd.DataFrame({pre_label: pre, post_label: post, "Gain": post - pre}).to_excel(xw, sheet_name="Data used", index=False)
    return buf.getvalue()


st.divider()
d1, d2 = st.columns(2)
d1.download_button("Download results (Excel)", results_excel(), "prepost_results.xlsx",
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
d2.download_button("Download write-up (text)", text, "prepost_writeup.txt", "text/plain", use_container_width=True)

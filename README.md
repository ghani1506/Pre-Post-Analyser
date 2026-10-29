# Pre–Post Analyser

A Streamlit app for pre-test / post-test research designs. Upload your scores and it reports:

- **Descriptive statistics** for the pre-test and post-test (N, mean, SD, SE, min, max) and their correlation
- **Paired-samples *t*-test**: *t*, df, *p* (two- or one-tailed), mean gain and its confidence interval
- **Cohen's *d*** in four forms, each with Cohen's (1988) small / medium / large label:
  - *d<sub>z</sub>*, the SPSS and JASP default for paired data, with an exact noncentral-*t* confidence interval
  - Hedges' *g<sub>z</sub>*, corrected for small samples
  - *d<sub>av</sub>*, standardised by the average of the pre and post SDs
  - *d* with the pooled SD
- **Assumption check**: Shapiro–Wilk normality test on the differences, with the Wilcoxon signed-rank test (and effect size *r*) as the non-parametric alternative
- **Charts** of each student's change and of the distribution of gains
- **An APA-style write-up**, plus downloadable results as an Excel file

All statistics are calculated with SciPy and follow the same methods as SPSS, JASP and R. The test suite checks the results against R's published output for Student's sleep data.

## Data format

One row per student, with a header row. Extra columns (names, class, other tests) are fine, because you choose the pre-test and post-test columns in the app.

| Student | Pre | Post |
|---|---|---|
| S01 | 42 | 51 |
| S02 | 55 | 61 |

CSV and Excel (.xlsx) files both work, and you can also type or paste scores straight into the app. A row with a blank or non-number score in either column is left out (listwise deletion), and the app tells you how many rows were left out.

## Run it on your own computer

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy free on Streamlit Community Cloud

1. Put these files in a GitHub repository, keeping the same folder layout.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app**, choose the repository and branch, and set the main file to `app.py`.
4. Click **Deploy**. After a few minutes the app is live at a `*.streamlit.app` link you can share.

Each time you push changes to GitHub, the live app updates automatically.

## Check the statistics

```bash
pip install pytest
pytest
```

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit interface |
| `prepost_stats.py` | All statistical calculations |
| `sample_data.csv` | Example data for 30 students |
| `tests/` | Checks against reference values from R |
| `requirements.txt` | Python packages for Streamlit Cloud |
| `.streamlit/config.toml` | Colour theme and upload limit |

## References

- Cohen, J. (1988). *Statistical power analysis for the behavioral sciences* (2nd ed.). Erlbaum.
- Hedges, L. V., & Olkin, I. (1985). *Statistical methods for meta-analysis*. Academic Press.
- Lakens, D. (2013). Calculating and reporting effect sizes to facilitate cumulative science. *Frontiers in Psychology, 4*, 863.
- Royston, P. (1995). Remark AS R94: A remark on Algorithm AS 181. *Applied Statistics, 44*(4), 547–551.

# GrantLens: NLP-Based Research Grant Proposal Assessment

**Research prototype, not an operational grant-approval predictor.**

GrantLens analyzes a proposal draft for clarity, methodological completeness, and alignment with a funding-call description. It includes an experimental text-classification pipeline for award-outcome research.

## Critical validity warning

The included `data/sample_grants.csv` contains **synthetic demonstration records**, not real grant applications or verified outcomes. Results trained on it are for testing the software only and must not be reported as scientific findings.

Public repositories such as NIH RePORTER and NSF Award Search primarily expose funded projects/awards. A funded-only corpus cannot train a reliable funded-vs-not-funded classifier. For a defensible approval probability, obtain permissioned or open records with both positive and negative application outcomes, comparable application populations, and documented outcome definitions. Until then, use the quality-assessment outputs only and label any classifier result as a software demonstration.

## Features

- Streamlit web dashboard for proposal draft + funding call text.
- Section completeness, objective/methods/measurement/ethics indicators, word count, readability proxy.
- TF-IDF + Logistic Regression baseline classifier and evaluation metrics.
- Training script, held-out test metrics, confusion matrix, ROC-AUC (when defined), and model artifact.
- Dataset schema and reproducibility checklist.
- CSV upload for locally curated, permissioned records.
- No live API scraping is required to run the app.

## Run on Windows (CMD)

```bat
cd grant_nlp_project
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If `py -3.11` is unavailable, use `python -m venv .venv` with a supported Python version. Streamlit and Sentence Transformers support modern Python versions; check package documentation if your environment differs.

## Train a baseline model

```bat
python train_model.py --data data/sample_grants.csv --text-col abstract --label-col outcome
```

The script will mark metrics as **synthetic demonstration only** when the input is the bundled sample. For real research, replace it with a curated dataset containing verified outcomes and update `data/DATA_DICTIONARY.md`. Never combine unrelated schemes or years without documenting the differences.

## Realistic dataset strategy

1. **NSF Awards**: public award records and abstracts; useful for the funded-project corpus, not a complete rejected-application dataset.
   - API docs: https://resources.research.gov/common/webapi/awardapisearch-v1.htm
   - Search/download: https://www.nsf.gov/awardsearch/
2. **NIH RePORTER**: public projects, abstracts, funding metadata and APIs.
   - API: https://api.reporter.nih.gov/
   - Dataset overview: https://catalog.data.gov/dataset/nih-research-portfolio-online-reporting-tools-expenditures-and-results-reporter
3. **SBIR/STTR**: awards, topics and company metadata; useful for a defined innovation-funding domain.
   - Data resources: https://www.sbir.gov/data-resources
4. **Rejected / unsuccessful outcomes**: seek a research agreement with a funder or institution, or use a dataset whose licensing explicitly permits research. Do not infer rejection from absence in an award database.
5. **Call alignment**: collect the original call/RFP text, eligibility, objectives, evaluation criteria and call dates. Version each call because criteria change over time.
6. **Manual feature labels**: have at least two trained raters score a stratified subset for clarity, objectives, methods, evaluation plan, feasibility, ethics, and alignment. Report Cohen's kappa or Krippendorff's alpha and adjudication protocol.

## Suggested schema

See `data/DATA_DICTIONARY.md`. Minimum outcome modeling fields:
- stable `record_id`
- `abstract` or proposal text available at submission time
- `outcome` (`1` funded / `0` not funded), with source and definition
- `scheme_id`, `call_id`, `decision_date` or year
- duplicate / resubmission / project-family identifiers
- source and licensing metadata

Do not use post-award outputs, award amount, publication count, final project outcomes, or fields that only become available after decision as predictors. Those cause target leakage.

## Research evaluation protocol

- Freeze a data extraction date and a preregistered analysis plan.
- Deduplicate near-identical abstracts and link resubmissions/project families.
- Use **temporal holdout** (train on older decisions, test on newer decisions) as primary; also report group-held-out results by call or institution where feasible.
- Fit preprocessing and feature selection only within training folds.
- Compare majority-class baseline, TF-IDF + Logistic Regression, linear SVM, and a semantic embedding model. Keep a transformer model as an optional extension.
- Report class balance, precision, recall, F1, PR-AUC, ROC-AUC, Brier score, calibration curve, confidence intervals and threshold-specific confusion matrix.
- Calibrate on a validation set, never the test set. Scikit-learn explains why calibration should use data independent of the model's training data: https://scikit-learn.org/stable/modules/calibration.html
- Evaluate subgroup performance only when sample size and governance allow; check differences by scheme, discipline, institution type and year.
- Compare to blinded human reviewer ratings where possible. Report inter-rater reliability.
- Conduct ablations: text-only, engineered quality features, call-alignment features, and combined features.
- Use bootstrap confidence intervals and repeat across temporal cutoffs.
- Do not describe a probability as an individual applicant's true chance unless external calibration and population transportability have been demonstrated.

## Limitations and ethics

- Award abstracts are not full submitted proposals and often omit detailed methods, budget, ethics, and review comments.
- Public award-only datasets are selection-biased and cannot identify rejection outcomes.
- Funding decisions depend on budgets, panel composition, eligibility, strategic priorities, portfolio balance, and reviewer judgments that may not be represented in text.
- Writing style and institutional resources may act as proxies for privilege, language background or discipline.
- Quality scores are heuristic indicators, not validated psychometric measures.
- Synthetic examples are only for smoke-testing the app.
- Never upload confidential proposal material to a hosted app without authorization. The default app runs locally.
- The system is decision support, not a substitute for reviewers or a funding decision.

## Suggested paper contributions

1. A transparent benchmark comparing lexical, semantic and engineered proposal features.
2. Temporal and cross-scheme generalization tests.
3. Probability calibration and uncertainty analysis.
4. Human-annotated proposal-quality dimensions and inter-rater reliability.
5. Error analysis of discipline, call wording, and resubmission leakage.

Do not claim novelty or journal suitability until a literature review and empirical results support those claims.

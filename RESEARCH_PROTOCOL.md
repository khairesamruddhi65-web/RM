# Publication-oriented research protocol

## Research questions
RQ1. Do lexical features, semantic embeddings, and manually annotated proposal-quality features improve prediction over a majority-class baseline?
RQ2. How well do models generalize across decision years and funding schemes?
RQ3. Are predicted probabilities calibrated, and does calibration transfer across schemes?
RQ4. Which textual signals correlate with model decisions, and are those signals stable across disciplines?

## Primary outcome
Define the unit as an application submitted to a specific call. Define positive/negative labels from official decisions, including an explicit treatment of withdrawn, ineligible, triaged, wait-listed, and resubmitted applications. Do not treat missing award records as negatives.

## Data collection
- Record source URL/API, query, retrieval date, terms of use, and all transformations.
- Prefer full proposals only when access rights and consent permit. Otherwise state that the task concerns abstracts or public project summaries, not proposal drafts.
- Keep raw data immutable; store cleaned data separately and maintain a reproducible data dictionary.
- Exclude personal names and sensitive reviewer comments unless explicitly authorized and ethically approved.
- Deduplicate exact and near-duplicate text; group related proposals and resubmissions to prevent leakage.

## Feature families
A. Lexical: word/character n-grams, vocabulary diversity, sentence length, readability.
B. Semantic: sentence embeddings and similarity to call objectives/criteria.
C. Structure: objective, hypothesis, method, sampling, metrics, timeline, risk, ethics, impact, data management and dissemination indicators.
D. Context: call, scheme and year. Only use features available at decision time; report a text-only model separately.
E. Human annotations: blinded rubric scores by multiple raters, with inter-rater agreement.

## Baselines and ablations
1. Majority class.
2. TF-IDF + Logistic Regression.
3. TF-IDF + linear SVM.
4. Engineered feature-only logistic regression / gradient boosting.
5. Sentence embeddings + regularized classifier.
6. Combined features.
Do not compare models using different data splits.

## Splits and metrics
Primary: temporal holdout with all records from the latest decision period reserved for testing. Secondary: group split by call or institution. Use nested cross-validation for tuning within the training partition. Fit vocabulary, imputation, scaling, feature selection and calibration only inside training/validation folds.

Report prevalence, precision, recall, F1, balanced accuracy, PR-AUC, ROC-AUC, Brier score, calibration slope/intercept and reliability plots. Include bootstrap confidence intervals. Report a threshold selected on validation data, not test data. Compare calibration methods on validation data only.

## Human validation
Create a rubric with operational definitions. Two or more raters independently score a subset without seeing outcomes. Report Cohen's kappa for categorical labels or intraclass correlation / Krippendorff's alpha for ordinal dimensions. Resolve disagreements through documented adjudication.

## Governance
Document access permissions, retention, privacy, conflicts of interest and intended use. Do not deploy for automated funding decisions. Audit for disparate errors by discipline, language background, institution type and scheme where lawful and sufficiently powered.

## Paper outline
1. Introduction and related work (systematic search and novelty gap)
2. Data sources, cohort construction, outcome definitions and ethics
3. Feature engineering and model families
4. Temporal / grouped evaluation and calibration
5. Results with confidence intervals and ablations
6. Error analysis, subgroup limitations and human comparison
7. Discussion, reproducibility, limitations and responsible-use statement

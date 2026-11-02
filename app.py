from pathlib import Path
import json
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from features import quality_features, alignment_score

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "grant_outcome_model.joblib"
METRICS_PATH = ROOT / "reports" / "evaluation.json"

st.set_page_config(page_title="GrantLens | Proposal Research Analytics", page_icon="🔬", layout="wide")
st.title("🔬 GrantLens")
st.caption("NLP-based research proposal feature mining · Local research prototype")

with st.sidebar:
    st.header("About this prototype")
    st.info("Quality indicators are heuristic. A true approval probability requires verified funded and unsuccessful application records and external validation.")
    st.markdown("**Data sources**")
    st.markdown("- [NIH RePORTER API](https://api.reporter.nih.gov/)")
    st.markdown("- [NSF Award Search](https://www.nsf.gov/awardsearch/)")
    st.markdown("- [SBIR/STTR data](https://www.sbir.gov/data-resources)")

tab1, tab2, tab3 = st.tabs(["Proposal analyzer", "Dataset explorer", "Model evaluation"])

with tab1:
    left, right = st.columns(2)
    with left:
        title = st.text_input("Proposal title", placeholder="Enter a concise research title")
        proposal = st.text_area(
            "Proposal draft / abstract",
            height=320,
            placeholder="Paste the text you are authorized to analyze. Include objectives, methods, evaluation, timeline, ethics, and expected impact."
        )
    with right:
        call_text = st.text_area(
            "Funding call / programme criteria",
            height=320,
            placeholder="Paste the official funding call, eligibility criteria, objectives, and assessment criteria."
        )
        st.caption("The alignment score uses lexical TF-IDF similarity. It is a baseline, not a semantic or official eligibility assessment.")
    if st.button("Analyze proposal", type="primary", use_container_width=True):
        if len(proposal.strip()) < 40:
            st.error("Enter a longer draft (at least 40 characters) for a meaningful analysis.")
        else:
            f = quality_features(proposal)
            align = alignment_score(proposal, call_text) if call_text.strip() else None
            a, b, c, d = st.columns(4)
            a.metric("Word count", f["word_count"])
            b.metric("Clarity indicator", f'{f["clarity_heuristic"]}/100')
            c.metric("Methodology indicator", f'{f["methodology_completeness_heuristic"]}/100')
            d.metric("Call-text similarity", f"{align}/100" if align is not None else "Not supplied")
            st.subheader("Feature audit")
            names = {
                "section_objectives": "Objectives / research question",
                "section_methodology": "Methods / study design",
                "section_evaluation": "Evaluation / validation",
                "section_timeline": "Timeline / milestones",
                "section_ethics": "Ethics / privacy",
                "section_risk": "Risks / mitigation",
                "section_impact": "Impact / deliverables",
                "section_reproducibility": "Reproducibility / open science",
            }
            audit = pd.DataFrame([{"Feature": label, "Detected": "Detected" if f[key] else "Not detected"} for key,label in names.items()])
            st.dataframe(audit, use_container_width=True, hide_index=True)
            missing = [label for key,label in names.items() if not f[key]]
            if missing:
                st.warning("Consider explicitly addressing: " + ", ".join(missing) + ". Absence of keywords does not prove the section is missing.")
            st.subheader("Interpretation")
            st.write("These indicators are transparent screening aids. They do not evaluate scientific truth, feasibility, novelty, budget appropriateness, or reviewer judgment.")
            with st.expander("View raw feature values"):
                st.json(f)
            model_exists = MODEL_PATH.exists()
            if model_exists:
                try:
                    model = joblib.load(MODEL_PATH)
                    p = float(model.predict_proba([proposal])[0][1])
                    st.subheader("Experimental classifier output")
                    st.metric("Class-1 model score", f"{p:.1%}")
                    if METRICS_PATH.exists():
                        m = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
                        if m.get("synthetic_demo_only"):
                            st.error("This model was trained on synthetic demo data. This number is not an approval probability and must not be used in a proposal or publication.")
                        else:
                            st.warning("This is a model-estimated class probability, not a verified personal chance of funding. Check external calibration and dataset scope before interpretation.")
                except Exception as exc:
                    st.warning(f"Could not load classifier: {exc}")
            else:
                st.info("No classifier artifact found. Run `python train_model.py` to create a software smoke-test model. The bundled sample is synthetic and not scientifically valid.")
            st.download_button("Download feature audit (JSON)", json.dumps({"title":title,"features":f,"call_similarity":align}, indent=2), file_name="proposal_feature_audit.json", mime="application/json")

with tab2:
    st.subheader("Inspect a dataset")
    uploaded = st.file_uploader("Upload a CSV you have permission to use", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.write(f"Rows: {len(df):,} · Columns: {len(df.columns)}")
            st.dataframe(df.head(100), use_container_width=True)
            st.caption("Before modeling, verify outcome labels, licenses, duplicate/resubmission groups, decision dates, and target leakage.")
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
    else:
        st.markdown("The bundled `data/sample_grants.csv` is synthetic and intended only to test the training pipeline.")
        st.markdown("Read `data/DATA_DICTIONARY.md` for the suggested schema.")

with tab3:
    st.subheader("Training evaluation")
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        if metrics.get("synthetic_demo_only"):
            st.error("Synthetic demo metrics only — not research findings.")
        cols = st.columns(4)
        for col, key, label in zip(cols, ["n_total","f1","average_precision_pr_auc","brier_score"], ["Rows","F1","PR-AUC / AP","Brier score"]):
            val = metrics.get(key)
            col.metric(label, f"{val:.3f}" if isinstance(val, float) else (str(val) if val is not None else "N/A"))
        st.json(metrics)
        st.caption("Use a temporal or group-held-out test set for publication; this bundled random split is a demonstration.")
    else:
        st.info("No evaluation found. Run `python train_model.py` first.")

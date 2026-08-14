"""
Streamlit demo UI for the Automated Data Filler Agent.

A thin wrapper around agent.run_agent.run_pipeline() -- reads much better
live during judging than a terminal. Text box in, extracted Contact/
Account/Opportunity shown as cards, confirm button writes to the CRM.

Run from the project root:
    streamlit run app.py
"""

import json
import streamlit as st

from agent.extractor_ollama import extract_crm_data
from agent.validator import validate_extraction, ValidationError
from agent.mock_crm_client import MockCRMClient

st.set_page_config(page_title="Automated Data Filler Agent", page_icon="🤖", layout="centered")

st.title("🤖 Automated Data Filler Agent")
st.caption("Paste a sales meeting summary and I'll extract structured CRM data automatically.")

# Session state holds the extraction between the "Extract" and "Confirm" button clicks
if "extracted" not in st.session_state:
    st.session_state.extracted = None
if "warnings" not in st.session_state:
    st.session_state.warnings = []

EXAMPLE_SUMMARY = (
    "Had a call with Sarah Johnson, Marketing Director at GrowthTech. "
    "They need marketing automation for 50-person team. Budget: $30K annually. "
    "Evaluating HubSpot vs our solution. Next: Demo scheduled for Friday."
)

summary = st.text_area(
    "Meeting summary",
    height=140,
    placeholder=EXAMPLE_SUMMARY,
    help="Paste raw, unstructured sales meeting notes here.",
)

col1, col2 = st.columns([1, 1])
with col1:
    extract_clicked = st.button("Extract CRM Data", type="primary", use_container_width=True)
with col2:
    if st.button("Use example text", use_container_width=True):
        st.session_state["_prefill"] = EXAMPLE_SUMMARY
        st.rerun()

if "_prefill" in st.session_state:
    summary = st.session_state.pop("_prefill")
    st.info("Example text loaded above -- click 'Extract CRM Data' to run it.")

if extract_clicked:
    if not summary.strip():
        st.warning("Paste a meeting summary first.")
    else:
        with st.spinner("Extracting structured data (local model -- may take a bit)..."):
            try:
                extracted = extract_crm_data(summary)
                warnings = []
                try:
                    warnings = validate_extraction(extracted)
                except ValidationError as e:
                    st.error(f"Validation failed -- extraction looks malformed: {e}")
                    extracted = None
                st.session_state.extracted = extracted
                st.session_state.warnings = warnings
            except RuntimeError as e:
                st.error(f"Extraction failed: {e}")
                st.session_state.extracted = None

# --- Show extraction result as cards ---
if st.session_state.extracted:
    data = st.session_state.extracted
    c, a, o = data["Contact"], data["Account"], data["Opportunity"]

    st.subheader("Here's what I extracted")
    conf = data.get("confidence", "unknown")
    conf_color = {"high": "green", "medium": "orange", "low": "red"}.get(conf, "gray")
    st.markdown(f"Confidence: :{conf_color}[**{conf}**]")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 👤 Contact")
        name = f"{c.get('FirstName') or ''} {c.get('LastName') or ''}".strip()
        st.write(f"**Name:** {name or '—'}")
        st.write(f"**Title:** {c.get('Title') or '—'}")
        st.write(f"**Email:** {c.get('Email') or '—'}")
        st.write(f"**Phone:** {c.get('Phone') or '—'}")

    with col2:
        st.markdown("### 🏢 Account")
        st.write(f"**Company:** {a.get('Name') or '—'}")
        st.write(f"**Industry:** {a.get('Industry') or '—'}")
        emp = a.get("NumberOfEmployees")
        st.write(f"**Employees:** {emp if emp else '—'}")
        st.write(f"**Budget:** {a.get('BudgetRange__c') or '—'}")

    with col3:
        st.markdown("### 💼 Opportunity")
        st.write(f"**Deal:** {o.get('Name') or '—'}")
        amount = o.get("Amount")
        st.write(f"**Value:** {f'${amount:,.0f}' if amount else '—'}")
        st.write(f"**Stage:** {o.get('StageName') or 'Prospecting'}")
        st.write(f"**Competitor:** {o.get('Competitor__c') or '—'}")

    if o.get("NextStep"):
        st.info(f"**Next step:** {o['NextStep']}")

    if st.session_state.warnings:
        for w in st.session_state.warnings:
            st.warning(w)

    with st.expander("Raw extracted JSON"):
        st.json(data)

    st.divider()

    if st.button("✅ Confirm and write to CRM", type="primary"):
        crm = MockCRMClient()
        result = crm.write_crm_record(data)

        st.success("Done! CRM actions taken:")
        for key in ("account", "contact", "opportunity"):
            record_id = result.get(f"{key}_id")
            action = result.get(f"{key}_action")
            if record_id:
                icon = "🆕" if action == "created" else "🔁"
                st.write(f"{icon} **{key.capitalize()}**: `{action}` (`{record_id}`)")
            else:
                st.write(f"⏭️ **{key.capitalize()}**: not created (no data found)")

        st.session_state.extracted = None  # clear so the form resets for the next entry

st.divider()
with st.expander("View mock CRM contents"):
    crm = MockCRMClient()
    st.json(crm.db)
    if st.button("Reset mock CRM (clear all records)"):
        crm.reset()
        st.rerun()

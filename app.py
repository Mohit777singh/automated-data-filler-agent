"""
AI Sales Meeting Agent — Enterprise SaaS Dashboard
===================================================
Professional B2B Sales Intelligence Platform.

Workflow:
1. Meeting Setup Card (Google Meet / Zoom URL + Bot Name + Audio Source)
2. Live Transcript Card (Real-Time Audio Capture + Timestamps + Copy/Clear)
3. CRM Generation Card (Manual "Generate CRM Report" Trigger -> Structured Enterprise CRM Report)
"""

import time
import logging
import streamlit as st

# ---------------------------------------------------------------------------
# Backend Modules
# ---------------------------------------------------------------------------
from agent.sales_meeting_agent import SalesMeetingAgent, AgentResult
from agent.extractor_ollama import extract_crm_data, resolve_ollama_model
from agent.validator import validate_extraction, ValidationError
from agent.mock_crm_client import MockCRMClient
from agent.live_listener import LiveMeetingListener
from agent.meeting_joiner import MeetingJoiner
from agent.audio_devices import (
    logger as pipeline_logger,
    list_input_devices,
    find_loopback_device,
    is_macos,
    MACOS_BLACKHOLE_SETUP_MD,
)

# ---------------------------------------------------------------------------
# Streamlit Page Setup & Enterprise SaaS Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Sales Meeting Agent | Enterprise Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Professional Salesforce / HubSpot / Modern SaaS CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1300px;
    }

    /* Header Styling */
    .dashboard-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.2rem 1.6rem;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 12px;
        color: #ffffff;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.15);
    }
    .header-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .header-logo {
        font-size: 1.8rem;
        background: rgba(255, 255, 255, 0.1);
        padding: 8px 12px;
        border-radius: 8px;
    }
    .header-title {
        font-size: 1.45rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0;
        color: #ffffff;
    }
    .header-subtitle {
        font-size: 0.85rem;
        color: #94a3b8;
        margin: 0;
    }
    
    /* Status Badges */
    .badge-ready {
        background-color: #f1f5f9;
        color: #475569;
        border: 1px solid #cbd5e1;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .badge-recording {
        background-color: #fee2e2;
        color: #dc2626;
        border: 1px solid #fca5a5;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        animation: pulse 1.8s infinite;
    }
    .badge-processing {
        background-color: #eff6ff;
        color: #2563eb;
        border: 1px solid #bfdbfe;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .badge-completed {
        background-color: #ecfdf5;
        color: #059669;
        border: 1px solid #a7f3d0;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.55; }
        100% { opacity: 1; }
    }

    /* Enterprise Cards */
    .saas-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.4rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
    }
    .card-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Transcript Stream Viewer */
    .transcript-container {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.2rem;
        min-height: 200px;
        max-height: 360px;
        overflow-y: auto;
        font-family: 'JetBrains Mono', 'SFMono-Regular', Menlo, Monaco, Consolas, monospace;
        font-size: 0.88rem;
        line-height: 1.7;
        color: #1e293b;
        white-space: pre-wrap;
    }

    /* CRM Entity Items */
    .crm-item {
        margin-bottom: 0.9rem;
    }
    .crm-label {
        font-size: 0.76rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 2px;
    }
    .crm-value {
        font-size: 0.98rem;
        font-weight: 500;
        color: #0f172a;
    }
    .crm-tag {
        display: inline-block;
        background-color: #f1f5f9;
        color: #334155;
        font-size: 0.82rem;
        font-weight: 500;
        padding: 3px 8px;
        border-radius: 6px;
        margin-right: 4px;
        margin-bottom: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Pipeline Log Handler
# ---------------------------------------------------------------------------
if "ui_logs" not in st.session_state:
    st.session_state.ui_logs = []


class UIStreamLogHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            if "ui_logs" in st.session_state:
                st.session_state.ui_logs.append(msg)
                if len(st.session_state.ui_logs) > 50:
                    st.session_state.ui_logs.pop(0)
        except Exception:
            pass


_handler_attached = any(isinstance(h, UIStreamLogHandler) for h in pipeline_logger.handlers)
if not _handler_attached:
    ui_handler = UIStreamLogHandler()
    ui_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s", "%H:%M:%S"))
    pipeline_logger.addHandler(ui_handler)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "meeting_state" not in st.session_state:
    st.session_state.meeting_state = "idle"  # "idle", "recording", "processing", "completed", "error"

if "meeting_joiner" not in st.session_state:
    st.session_state.meeting_joiner = None

if "audio_listener" not in st.session_state:
    st.session_state.audio_listener = None

if "accumulated_transcript" not in st.session_state:
    st.session_state.accumulated_transcript = ""

if "crm_result" not in st.session_state:
    st.session_state.crm_result = None

if "error_message" not in st.session_state:
    st.session_state.error_message = None

if "ollama_model" not in st.session_state:
    st.session_state.ollama_model = resolve_ollama_model()

# ---------------------------------------------------------------------------
# Header & Dynamic Status Indicator
# ---------------------------------------------------------------------------
status_map = {
    "idle": ('<span class="badge-ready">⚪ Ready</span>', "Ready for meeting session"),
    "recording": ('<span class="badge-recording">🔴 [REC ●] Recording</span>', "Capturing live participant audio"),
    "processing": ('<span class="badge-processing">⚡ Processing</span>', "Extracting CRM intelligence"),
    "completed": ('<span class="badge-completed">✅ Completed</span>', "CRM Report ready"),
    "error": ('<span class="badge-recording">⚠️ Alert</span>', "Action required"),
}

badge_html, status_subtext = status_map.get(st.session_state.meeting_state, status_map["idle"])

st.markdown(
    f"""
    <div class="dashboard-header">
        <div class="header-brand">
            <div class="header-logo">⚡</div>
            <div>
                <h1 class="header-title">AI Sales Meeting Agent</h1>
                <p class="header-subtitle">Enterprise Sales Intelligence • Real-Time Whisper Transcription • Ollama CRM Sync</p>
            </div>
        </div>
        <div>
            {badge_html}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# SECTION 1: Meeting Setup Card
# ---------------------------------------------------------------------------
st.markdown('<div class="card-title">⚙️ Section 1: Meeting Setup</div>', unsafe_allow_html=True)

with st.container():
    col_url, col_bot, col_dev = st.columns([3.5, 2, 2.5])

    with col_url:
        meeting_url = st.text_input(
            "Meeting URL",
            placeholder="https://meet.google.com/xxx-yyyy-zzz or https://zoom.us/j/123456789",
            disabled=(st.session_state.meeting_state in ["recording", "processing"]),
            help="Paste a valid Google Meet or Zoom meeting invite link."
        )

    with col_bot:
        bot_name = st.text_input(
            "Bot Name",
            value="AI Sales Agent",
            disabled=(st.session_state.meeting_state in ["recording", "processing"]),
            help="Display name used by the bot when entering the call."
        )

    with col_dev:
        available_devices = list_input_devices()
        loopback_dev = find_loopback_device()
        device_options = {}
        default_index = 0

        if available_devices:
            for idx, d in enumerate(available_devices):
                label = f"{d['name']} (Ch: {d['max_input_channels']})"
                if d.get("is_loopback"):
                    label += " [Virtual Loopback]"
                    default_index = idx
                device_options[d["index"]] = label

            selected_dev_idx = st.selectbox(
                "Audio Source",
                options=list(device_options.keys()),
                format_func=lambda x: device_options[x],
                index=default_index,
                disabled=(st.session_state.meeting_state in ["recording", "processing"]),
                help="Select BlackHole (macOS) or loopback device to capture participant voices from the meeting."
            )
        else:
            st.selectbox("Audio Source", options=[None], format_func=lambda x: "System Default Mic", disabled=True)
            selected_dev_idx = None

    # macOS Setup Helper Guide
    if is_macos() and not loopback_dev:
        with st.expander("ℹ️ macOS Audio Routing (BlackHole Setup Guide)", expanded=False):
            st.markdown(MACOS_BLACKHOLE_SETUP_MD)

    # Action Buttons
    col_btn_join, col_btn_leave, _ = st.columns([2.5, 2.5, 3])

    if st.session_state.meeting_state in ["idle", "completed", "error"]:
        with col_btn_join:
            if st.button("🚀 Join Meeting", type="primary", use_container_width=True):
                if not meeting_url or not meeting_url.strip():
                    st.warning("Please enter a valid Google Meet or Zoom URL first.")
                else:
                    st.session_state.error_message = None
                    st.session_state.crm_result = None
                    st.session_state.accumulated_transcript = ""
                    st.session_state.ui_logs = []

                    # 1. Start Live Audio Capture
                    try:
                        listener = LiveMeetingListener(
                            model_size="small",
                            chunk_duration_sec=5.0,
                        )
                        listener.start(device=selected_dev_idx)
                        st.session_state.audio_listener = listener
                    except Exception as audio_err:
                        st.error(f"Could not open audio stream: {audio_err}")
                        st.stop()

                    # 2. Start Browser Meeting Joiner
                    try:
                        joiner = MeetingJoiner(participant_name=bot_name)
                        st.session_state.meeting_joiner = joiner
                        
                        with st.spinner("Bot launching browser and joining meeting muted..."):
                            joiner.join_meeting(meeting_url.strip(), headless=False)
                        
                        st.session_state.meeting_state = "recording"
                        st.rerun()

                    except Exception as join_err:
                        st.session_state.meeting_state = "error"
                        st.session_state.error_message = f"Failed to join meeting: {join_err}"
                        if st.session_state.audio_listener:
                            st.session_state.audio_listener.stop()
                        st.error(st.session_state.error_message)

    elif st.session_state.meeting_state == "recording":
        with col_btn_join:
            if st.button("⏹️ End Meeting & Store Transcript", type="primary", use_container_width=True):
                # 1. Teardown Joiner
                if st.session_state.meeting_joiner:
                    try:
                        st.session_state.meeting_joiner.leave()
                    except Exception as e:
                        pipeline_logger.warning(f"Browser leave error: {e}")
                    st.session_state.meeting_joiner = None

                # 2. Stop Audio Listener & Retrieve Complete Transcript
                final_transcript = ""
                if st.session_state.audio_listener:
                    try:
                        final_transcript = st.session_state.audio_listener.stop(wait_for_drain=True)
                    except Exception as e:
                        pipeline_logger.warning(f"Audio listener stop error: {e}")
                    st.session_state.audio_listener = None

                st.session_state.accumulated_transcript = final_transcript

                if not final_transcript.strip():
                    st.session_state.meeting_state = "error"
                    st.session_state.error_message = (
                        "Meeting joined successfully but no speech was captured. "
                        "On Windows: In 'Audio Source', select 'Stereo Mix' (to record meeting participants) "
                        "or 'Microphone Array' (to record your mic), and make sure participants speak audibly."
                    )
                else:
                    st.session_state.meeting_state = "idle"

                st.rerun()

        with col_btn_leave:
            if st.button("❌ Leave Without Saving", use_container_width=True):
                if st.session_state.meeting_joiner:
                    st.session_state.meeting_joiner.leave()
                if st.session_state.audio_listener:
                    st.session_state.audio_listener.stop()
                st.session_state.meeting_state = "idle"
                st.rerun()

st.divider()

# ---------------------------------------------------------------------------
# SECTION 2: Live Transcript Card
# ---------------------------------------------------------------------------
has_transcript = bool(st.session_state.accumulated_transcript and st.session_state.accumulated_transcript.strip())

col_t_title, col_t_actions = st.columns([5, 3])

with col_t_title:
    if st.session_state.meeting_state == "recording":
        st.markdown('<div class="card-title">🎙️ Section 2: Live Transcript <span class="badge-recording" style="font-size:0.75rem;">[REC ●] Meeting Recording</span></div>', unsafe_allow_html=True)
    elif has_transcript:
        st.markdown('<div class="card-title">🎙️ Section 2: Live Transcript <span class="badge-completed" style="font-size:0.75rem;">✅ Transcript Ready</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="card-title">🎙️ Section 2: Live Transcript</div>', unsafe_allow_html=True)

with col_t_actions:
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        if has_transcript:
            st.download_button(
                label="📋 Copy / Download",
                data=st.session_state.accumulated_transcript,
                file_name="meeting_transcript.txt",
                mime="text/plain",
                use_container_width=True,
            )
        else:
            st.button("📋 Copy / Download", disabled=True, use_container_width=True)

    with col_c2:
        if st.button("🗑️ Clear", disabled=(not has_transcript or st.session_state.meeting_state == "recording"), use_container_width=True):
            st.session_state.accumulated_transcript = ""
            st.session_state.crm_result = None
            st.session_state.meeting_state = "idle"
            st.rerun()

transcript_placeholder = st.empty()

# Real-time polling when actively recording in meeting
if st.session_state.meeting_state == "recording":
    if st.session_state.audio_listener:
        st.session_state.accumulated_transcript = st.session_state.audio_listener.get_live_transcript()

    current_transcript = st.session_state.accumulated_transcript or "Listening for participant voices... (speech will appear here with timestamps)"
    transcript_placeholder.markdown(
        f'<div class="transcript-container">{current_transcript}</div>',
        unsafe_allow_html=True
    )

    with st.expander("🪵 Live Audio & Pipeline Logs", expanded=True):
        recent_logs = "\n".join(st.session_state.ui_logs[-6:]) if st.session_state.ui_logs else "Audio stream connected. Listening..."
        st.code(recent_logs, language="bash")

    # Auto-refresh loop
    time.sleep(2.0)
    st.rerun()

else:
    current_transcript = st.session_state.accumulated_transcript or "No transcript captured yet. Click 'Join Meeting' to record live participant dialogue."
    transcript_placeholder.markdown(
        f'<div class="transcript-container">{current_transcript}</div>',
        unsafe_allow_html=True
    )

    with st.expander("✍️ Type / Paste Meeting Notes or Custom Dialogue", expanded=False):
        manual_text = st.text_area(
            "Meeting Conversation / Notes",
            placeholder="e.g. I am the COO of Catphy company and my budget for today's investment is $5,000...",
            value=st.session_state.accumulated_transcript,
            height=120,
            key="manual_transcript_input"
        )
        if st.button("📥 Use This Transcript", use_container_width=True):
            if manual_text.strip():
                st.session_state.accumulated_transcript = manual_text.strip()
                st.session_state.meeting_state = "idle"
                st.session_state.crm_result = None
                st.rerun()

    with st.expander("🪵 Live Audio & Pipeline Logs", expanded=False):
        recent_logs = "\n".join(st.session_state.ui_logs[-6:]) if st.session_state.ui_logs else "Ready."
        st.code(recent_logs, language="bash")

# ---------------------------------------------------------------------------
# Error Notification
# ---------------------------------------------------------------------------
if st.session_state.meeting_state == "error" and st.session_state.error_message:
    st.error(f"⚠️ {st.session_state.error_message}")

st.divider()

# ---------------------------------------------------------------------------
# SECTION 3: CRM Generation Card
# ---------------------------------------------------------------------------
st.markdown('<div class="card-title">📊 Section 3: CRM Report & Generation</div>', unsafe_allow_html=True)

col_crm_ctrl, col_crm_status = st.columns([3, 5])

with col_crm_ctrl:
    btn_disabled = not has_transcript
    generate_crm_clicked = st.button(
        "⚡ Generate CRM Report",
        type="primary",
        disabled=btn_disabled or (st.session_state.meeting_state in ["recording", "processing"]),
        use_container_width=True,
        help="Extracts structured Customer, Account, Opportunity, Requirements, and Action Items via Ollama."
    )

with col_crm_status:
    if has_transcript and st.session_state.meeting_state != "recording":
        st.success("Transcript Ready ✅ Click **Generate CRM Report** to extract structured sales records.")
    elif st.session_state.meeting_state == "recording":
        st.info("🔴 Live meeting recording in progress. Click **End Meeting & Store Transcript** above when finished.")
    else:
        st.caption("Waiting for transcript...")

# ---------------------------------------------------------------------------
# Manual CRM Extraction Execution
# ---------------------------------------------------------------------------
if generate_crm_clicked and has_transcript:
    st.session_state.meeting_state = "processing"
    
    progress_box = st.empty()
    with progress_box.container():
        st.info("🧠 **Step 1/3:** Analyzing conversation...")
        time.sleep(0.6)

    try:
        pipeline_logger.info("Sending transcript to CRM extractor")
        with progress_box.container():
            st.info("🔍 **Step 2/3:** Extracting customer insights...")
        
        agent = SalesMeetingAgent(model_name=st.session_state.ollama_model)
        result = agent.run_end_to_end_pipeline(st.session_state.accumulated_transcript, auto_confirm=True)
        
        with progress_box.container():
            st.info("⚡ **Step 3/3:** Building CRM record...")
            time.sleep(0.5)
            
        st.session_state.crm_result = result
        st.session_state.meeting_state = "completed"
        progress_box.empty()
        st.rerun()

    except Exception as proc_err:
        progress_box.empty()
        st.session_state.meeting_state = "error"
        st.session_state.error_message = f"CRM Generation failed: {proc_err}"
        st.rerun()

# ---------------------------------------------------------------------------
# CRM Report Display
# ---------------------------------------------------------------------------
if st.session_state.crm_result:
    res: AgentResult = st.session_state.crm_result
    crm_data = res.crm_data or {}
    summary_data = res.summary or {}
    actions_data = res.actions or {}

    contact = crm_data.get("Contact") or {}
    account = crm_data.get("Account") or {}
    opportunity = crm_data.get("Opportunity") or {}
    insights = crm_data.get("Insights") or {}

    # Executive Summary Card
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📋 Executive Summary</div>', unsafe_allow_html=True)
    concise = summary_data.get("concise_summary") or summary_data.get("purpose") or "Sales discovery meeting concluded."
    st.info(concise)
    st.markdown('</div>', unsafe_allow_html=True)

    # 4 Main Enterprise Cards
    col_card1, col_card2 = st.columns(2)

    # Card 1: Customer Overview Card
    with col_card1:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">👤 Customer Overview Card</div>', unsafe_allow_html=True)
        
        first_name = contact.get("FirstName") or ""
        last_name = contact.get("LastName") or ""
        full_name = f"{first_name} {last_name}".strip() or summary_data.get("participants") or "Not specified"
        
        col_c_sub1, col_c_sub2 = st.columns(2)
        with col_c_sub1:
            st.markdown(f'<div class="crm-item"><div class="crm-label">Customer Name</div><div class="crm-value">{full_name}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="crm-item"><div class="crm-label">Company</div><div class="crm-value">{account.get("Name") or "—"}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="crm-item"><div class="crm-label">Job Title</div><div class="crm-value">{contact.get("Title") or "—"}</div></div>', unsafe_allow_html=True)
        
        with col_c_sub2:
            contact_info = contact.get("Email") or contact.get("Phone") or "Pending follow-up"
            st.markdown(f'<div class="crm-item"><div class="crm-label">Contact Information</div><div class="crm-value">{contact_info}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="crm-item"><div class="crm-label">Department</div><div class="crm-value">{contact.get("Department") or "—"}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="crm-item"><div class="crm-label">Industry & Size</div><div class="crm-value">{account.get("Industry") or "SaaS / Tech"} ({account.get("NumberOfEmployees") or "—"} employees)</div></div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Card 2: Opportunity Card
    with col_card2:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">💼 Opportunity Card</div>', unsafe_allow_html=True)
        
        col_o_sub1, col_o_sub2 = st.columns(2)
        with col_o_sub1:
            st.markdown(f'<div class="crm-item"><div class="crm-label">Opportunity Name</div><div class="crm-value">{opportunity.get("Name") or "Sales Opportunity"}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="crm-item"><div class="crm-label">Opportunity Stage</div><div class="crm-value"><span class="crm-tag">{opportunity.get("StageName") or "Discovery & Qualification"}</span></div></div>', unsafe_allow_html=True)
            amount_val = f"${opportunity.get('Amount'):,}" if opportunity.get('Amount') else (summary_data.get('budget') or account.get('BudgetRange__c') or "Pending Review")
            st.markdown(f'<div class="crm-item"><div class="crm-label">Budget / Deal Value</div><div class="crm-value">{amount_val}</div></div>', unsafe_allow_html=True)
        
        with col_o_sub2:
            st.markdown(f'<div class="crm-item"><div class="crm-label">Timeline</div><div class="crm-value">{summary_data.get("timeline") or insights.get("ExpectedDecisionDate") or "Upcoming Milestone"}</div></div>', unsafe_allow_html=True)
            competitors = opportunity.get("Competitor__c") or (", ".join(insights.get("CompetitorsMentioned", []))) or "None explicitly raised"
            st.markdown(f'<div class="crm-item"><div class="crm-label">Current Solution / Competitor</div><div class="crm-value">{competitors}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="crm-item"><div class="crm-label">Next Step</div><div class="crm-value">{opportunity.get("NextStep") or "Schedule Technical Walkthrough"}</div></div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Card 3 & 4: Requirements & Action Items
    col_card3, col_card4 = st.columns(2)

    # Card 3: Requirements Card
    with col_card3:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🎯 Requirements Card</div>', unsafe_allow_html=True)
        
        needs_list = summary_data.get("customer_needs") or insights.get("Requirements") or []
        pains_list = summary_data.get("pain_points") or insights.get("PainPoints") or []

        if isinstance(needs_list, str):
            needs_list = [needs_list]
        if isinstance(pains_list, str):
            pains_list = [pains_list]

        if needs_list:
            st.markdown("**Business Requirements:**")
            for req in needs_list:
                st.markdown(f"- {req}")

        if pains_list:
            st.markdown("**Pain Points:**")
            for p in pains_list:
                st.markdown(f"- {p}")

        if not needs_list and not pains_list:
            st.write("—")

        st.markdown('</div>', unsafe_allow_html=True)

    # Card 4: Action Items Card
    with col_card4:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">⚡ Action Items Card</div>', unsafe_allow_html=True)
        
        nba = actions_data.get("next_best_action") or {}
        if nba.get("primary_action"):
            st.markdown(f"📌 **Next Best Action:** {nba.get('primary_action')}")
            if nba.get("rationale"):
                st.caption(f"Strategy: {nba.get('rationale')}")

        tasks = actions_data.get("tasks") or []
        if tasks:
            st.markdown("**Follow-up Tasks:**")
            for t in tasks:
                subj = t.get("Subject", "Follow-up")
                due = t.get("ActivityDate", "Upcoming")
                st.markdown(f"- **{subj}** (Due: {due})")

        next_steps = summary_data.get("next_steps") or []
        if isinstance(next_steps, str):
            next_steps = [next_steps]

        if next_steps:
            st.markdown("**Scheduled Next Steps:**")
            for step in next_steps:
                st.markdown(f"✅ {step}")

        st.markdown('</div>', unsafe_allow_html=True)

    # Raw CRM JSON
    with st.expander("🔍 View Raw Structured CRM JSON"):
        st.json(crm_data)
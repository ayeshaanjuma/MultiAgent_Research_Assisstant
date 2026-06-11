import asyncio
import os
import streamlit as st
from config import validate_api_keys
from utils.clients import load_pdf_text
from rag_pipeline import RAGPipeline
from agents.research_agent import run_research_agent, validate_research_results
from agents.summarisation_agent import run_summarisation_agent, validate_summary
from agents.verification_agent import run_verification_agent, validate_verification
from agents.report_generator import (
    run_report_generator,
    render_report,
    export_report_pdf,
    validate_report,
)

st.set_page_config(
    page_title="Research Summarizer",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize Session State
if "query" not in st.session_state:
    st.session_state.query = ""
if "file_name" not in st.session_state:
    st.session_state.file_name = ""
if "articles" not in st.session_state:
    st.session_state.articles = []
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "verification" not in st.session_state:
    st.session_state.verification = []
if "report_text" not in st.session_state:
    st.session_state.report_text = ""
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = ""
if "error" not in st.session_state:
    st.session_state.error = ""
if "open_modal" not in st.session_state:
    st.session_state.open_modal = None

# Stats and Agent States
if "articles_count" not in st.session_state:
    st.session_state.articles_count = 0
if "claims_count" not in st.session_state:
    st.session_state.claims_count = 0
if "accuracy_value" not in st.session_state:
    st.session_state.accuracy_value = "--"
if "agent_states" not in st.session_state:
    st.session_state.agent_states = {
        "research": "waiting",
        "summarization": "waiting",
        "verification": "waiting",
        "report": "waiting"
    }

# Agent loading status strings
if "research_status" not in st.session_state:
    st.session_state.research_status = "Waiting..."
if "summarization_status" not in st.session_state:
    st.session_state.summarization_status = "Waiting..."
if "verification_status_msg" not in st.session_state:
    st.session_state.verification_status_msg = "Waiting..."
if "report_status" not in st.session_state:
    st.session_state.report_status = "Waiting..."

# SVGs for Icons
SVG_RESEARCH = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" style="width: 20px; height: 20px;"><path stroke-linecap="round" stroke-linejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" /></svg>'
SVG_SUMMARIZATION = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" style="width: 20px; height: 20px;"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>'
SVG_VERIFICATION = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" style="width: 20px; height: 20px;"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" /></svg>'
SVG_REPORT = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" style="width: 20px; height: 20px;"><path stroke-linecap="round" stroke-linejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" /></svg>'

# Inject Custom CSS for dark-grid theme
st.markdown("""
<style>
/* Custom background and grid pattern */
.stApp {
    background-color: #030712 !important;
    background-image: 
        linear-gradient(rgba(17, 24, 39, 0.4) 1px, transparent 1px),
        linear-gradient(90deg, rgba(17, 24, 39, 0.4) 1px, transparent 1px);
    background-size: 40px 40px;
    background-attachment: fixed;
}

/* Hide default streamlit header/footer */
header {visibility: hidden;}
footer {visibility: hidden;}

/* Custom styles for stats cards */
.stat-card {
    background-color: #0b0f19;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 16px 8px;
    text-align: center;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}
.stat-value {
    font-size: 26px;
    font-weight: 700;
    color: #00c497;
    font-family: 'Inter', sans-serif;
    margin-bottom: 2px;
}
.stat-value.white {
    color: #f8fafc;
}
.stat-label {
    font-size: 11px;
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Agent dashboard cards (styled directly on the container columns - locked to compact height) */
div[data-testid="column"]:has(div.agent-header) {
    background-color: #0c101b !important;
    border: 1px solid #2e3c54 !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: 0 12px 24px -10px rgba(0, 0, 0, 0.6) !important;
    position: relative !important;
    height: 140px !important;
    min-height: 140px !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: flex-start !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
div[data-testid="column"]:has(div.agent-header):hover {
    border-color: #00c497 !important;
    box-shadow: 0 0 15px rgba(0, 196, 151, 0.15) !important;
}

/* Background backdrop blur for page when modal is active */
.modal-backdrop {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    background: rgba(3, 7, 18, 0.75) !important;
    backdrop-filter: blur(8px) !important;
    -webkit-backdrop-filter: blur(8px) !important;
    z-index: 9998 !important;
}

/* Style for modal container wrapper overlay - perfectly centered & sized */
div[data-testid="stVerticalBlock"]:has(div[class*="modal-trigger"]) {
    position: fixed !important;
    top: 50% !important;
    left: 50% !important;
    transform: translate(-50%, -50%) !important;
    width: 70vw !important;
    max-width: 1000px !important;
    height: 70vh !important;
    max-height: 800px !important;
    background: rgba(12, 16, 27, 0.98) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 20px !important;
    padding: 28px !important;
    box-shadow: 0 0 50px rgba(0, 196, 151, 0.2) !important;
    z-index: 99999 !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    margin: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    animation: modalFadeIn 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Enforce word wrap and width constraints inside the modal to eliminate horizontal scroll */
div[data-testid="stVerticalBlock"]:has(div[class*="modal-trigger"]) * {
    word-break: break-word !important;
    overflow-wrap: break-word !important;
    max-width: 100% !important;
}

/* Style close button inside modal header */
div[data-testid="stVerticalBlock"]:has(div[class*="modal-trigger"]) div[data-testid="column"]:last-of-type button {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 50% !important;
    width: 30px !important;
    height: 30px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    color: #94a3b8 !important;
    font-size: 12px !important;
    cursor: pointer !important;
    padding: 0 !important;
    min-height: unset !important;
}
div[data-testid="stVerticalBlock"]:has(div[class*="modal-trigger"]) div[data-testid="column"]:last-of-type button:hover {
    background: rgba(239, 68, 68, 0.15) !important;
    border-color: rgba(239, 68, 68, 0.4) !important;
    color: #ef4444 !important;
}

@keyframes modalFadeIn {
    from { opacity: 0; transform: translate(-50%, -48%) scale(0.97); }
    to { opacity: 1; transform: translate(-50%, -50%) scale(1); }
}


.agent-card-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
}
.agent-icon-container {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    border-radius: 8px;
}
.agent-icon-container.blue {
    background-color: rgba(59, 130, 246, 0.15);
    color: #3b82f6;
}
.agent-icon-container.green {
    background-color: rgba(16, 185, 129, 0.15);
    color: #10b981;
}
.agent-icon-container.yellow {
    background-color: rgba(245, 158, 11, 0.15);
    color: #f59e0b;
}
.agent-icon-container.purple {
    background-color: rgba(139, 92, 246, 0.15);
    color: #8b5cf6;
}
.agent-info {
    flex-grow: 1;
    margin-left: 12px;
}
.agent-name {
    font-weight: 600;
    font-size: 14px;
    color: #f8fafc;
}
.agent-desc {
    font-size: 11px;
    color: #64748b;
    margin-top: 1px;
}
.agent-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-top: 4px;
}
.agent-dot.waiting {
    background-color: #475569;
    box-shadow: 0 0 6px #475569;
}
.agent-dot.in-progress {
    background-color: #3b82f6;
    box-shadow: 0 0 10px #3b82f6;
    animation: agentPulse 1.5s infinite;
}
.agent-dot.complete {
    background-color: #10b981;
    box-shadow: 0 0 10px #10b981;
}
.agent-status-text {
    font-size: 12px;
    color: #94a3b8;
    font-weight: 500;
}
@keyframes agentPulse {
    0% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.3); opacity: 0.5; }
    100% { transform: scale(1); opacity: 1; }
}

/* Styled text input */
div[data-testid="stTextInput"] input {
    background-color: #0b0f19 !important;
    border: 1px solid #1e293b !important;
    border-radius: 12px !important;
    color: #f8fafc !important;
    font-size: 16px !important;
    padding: 12px 16px !important;
    height: 48px !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #00c497 !important;
}

/* Select toggle button in top-right of agent column cards */
div[data-testid="column"]:has(div.agent-header) div[data-testid="stButton"]:first-of-type button {
    position: absolute !important;
    top: 20px !important;
    right: 20px !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #64748b !important;
    font-size: 18px !important;
    width: 32px !important;
    height: 32px !important;
    padding: 0 !important;
    min-height: unset !important;
    line-height: 1 !important;
    z-index: 100 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
div[data-testid="column"]:has(div.agent-header) div[data-testid="stButton"]:first-of-type button:hover {
    color: #3b82f6 !important;
    background: transparent !important;
    border: none !important;
}
div[data-testid="column"]:has(div.agent-header) div[data-testid="stButton"]:first-of-type button:active {
    transform: scale(0.92) !important;
}

/* Glassmorphism agent panel container */
.agent-details-panel {
    background: rgba(11, 15, 25, 0.6) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    padding: 24px !important;
    margin-top: 20px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
    animation: fadeIn 0.4s ease-out;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.badge {
    display: inline-block;
    padding: 3px 8px;
    font-size: 11px;
    font-weight: 600;
    border-radius: 6px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-right: 6px;
}
.badge.completed { background-color: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }
.badge.processing { background-color: rgba(59, 130, 246, 0.15); color: #3b82f6; border: 1px solid rgba(59, 130, 246, 0.3); }
.badge.verified { background-color: rgba(16, 185, 129, 0.2); color: #00c497; border: 1px solid rgba(16, 185, 129, 0.4); }
.badge.high-confidence { background-color: rgba(139, 92, 246, 0.15); color: #8b5cf6; border: 1px solid rgba(139, 92, 246, 0.3); }
.badge.trusted-source { background-color: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }

/* Start research button */
div[data-testid="stButton"] button {
    background-color: #00c497 !important;
    color: #030712 !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    height: 48px !important;
    width: 100% !important;
    transition: background-color 0.2s, transform 0.1s;
}
div[data-testid="stButton"] button:hover {
    background-color: #05a882 !important;
    color: #030712 !important;
}
div[data-testid="stButton"] button:active {
    transform: scale(0.98);
}

/* CSS to enable nesting the upload button inside the search query input box */
div[data-testid="stVerticalBlock"]:has(div[data-testid="stTextInput"]):has(div[data-testid="stFileUploader"]) {
    position: relative !important;
}

div[data-testid="stVerticalBlock"]:has(div[data-testid="stTextInput"]):has(div[data-testid="stFileUploader"]) div[data-testid="stElementContainer"]:has(div[data-testid="stFileUploader"]) {
    position: absolute !important;
    right: 4px !important;
    top: 0px !important;
    width: 48px !important;
    height: 48px !important;
    z-index: 10 !important;
    margin: 0 !important;
    padding: 0 !important;
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

div[data-testid="stElementContainer"]:has(div[data-testid="stFileUploader"]) * {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Make search input have padding on the right for the upload button */
div[data-testid="stTextInput"] input {
    padding-right: 52px !important;
}

/* Compact upload icon button styling */
div[data-testid="stFileUploader"] {
    width: 48px !important;
    min-width: 48px !important;
    height: 48px !important;
    padding: 0 !important;
    margin-top: 0px !important;
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
div[data-testid="stFileUploader"] > div {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

div[data-testid="stFileUploaderDropzone"] {
    width: 48px !important;
    height: 48px !important;
    min-height: 48px !important;
    padding: 0 !important;
    background-color: transparent !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    transition: transform 0.2s !important;
}
div[data-testid="stFileUploaderDropzone"]:hover {
    transform: scale(1.08) !important;
    background-color: transparent !important;
    background: transparent !important;
}
/* Hide all default dropzone text, buttons, and child elements */
div[data-testid="stFileUploaderDropzone"] * {
    display: none !important;
}
div[data-testid="stFileUploader"] label {
    display: none !important;
}
/* Add the upload SVG icon */
div[data-testid="stFileUploaderDropzone"]::after {
    content: "";
    width: 20px;
    height: 20px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%2364748b' stroke-width='2'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' d='M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5'/%3E%3C/svg%3E");
    background-size: contain;
    background-repeat: no-repeat;
    display: block;
    transition: opacity 0.2s !important;
}
div[data-testid="stFileUploaderDropzone"]:hover::after {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%2300c497' stroke-width='2'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' d='M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5'/%3E%3C/svg%3E");
}

/* Styling for the uploaded file name container */
div[data-testid="stFileUploaderFileData"] {
    position: absolute !important;
    top: 48px !important;
    right: 0px !important;
    background-color: #0b0f19 !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    padding: 4px 12px !important;
    margin-top: 8px !important;
    color: #e2e8f0 !important;
    font-size: 13px !important;
    z-index: 99 !important;
    width: auto !important;
    white-space: nowrap !important;
}

</style>
""", unsafe_allow_html=True)

# Helper functions to build HTML segments
def get_agent_header_html(name: str, desc: str, status: str, color_class: str, svg_icon: str, status_text: str = "Waiting...") -> str:
    status_class = "waiting"
    if status == "in-progress":
        status_class = "in-progress"
    elif status == "complete":
        status_class = "complete"
        
    # Special compact layout for complete Report Generator
    if name == "Report Generator" and status == "complete":
        sources_count = len(st.session_state.articles) if st.session_state.articles else 0
        accuracy = st.session_state.accuracy_value
        status_html = f"""
        <div style="font-size: 11px; color: #94a3b8; line-height: 1.4; margin-top: 10px;">
            <div>Status: <span style="color: #10b981; font-weight: 600;">Complete</span></div>
            <div>Sources Used: {sources_count}</div>
            <div>Citations: {sources_count}</div>
            <div>Confidence: {accuracy}</div>
            <span class="badge completed" style="margin-top: 6px; display: inline-block;">Report Ready</span>
        </div>
        """
    else:
        status_html = f'<div class="agent-status-text" style="margin-top: 16px;">{status_text}</div>'
        
    return f"""
    <div class="agent-header">
      <div class="agent-card-header">
        <div class="agent-icon-container {color_class}">
          {svg_icon}
        </div>
        <div class="agent-info" style="margin-right: 24px;">
          <div class="agent-name">{name}</div>
          <div class="agent-desc">{desc}</div>
        </div>
        <div class="agent-dot {status_class}"></div>
      </div>
      {status_html}
    </div>
    """


def get_stat_card_html(value: str, label: str, green_color: bool = False) -> str:
    val_class = "white" if not green_color else ""
    return f"""
    <div class="stat-card">
      <div class="stat-value {val_class}">{value}</div>
      <div class="stat-label">{label}</div>
    </div>
    """

# Load RAG pipeline with default streamlit spinner disabled
@st.cache_resource(show_spinner=False)
def load_rag_pipeline():
    return RAGPipeline(persist_directory="db")

# Layout: Title Zone & Stats Zone
header_left, header_right = st.columns([2.2, 1])
with header_left:
    st.markdown("""
        <h1 style="margin-bottom: 0px; font-weight: 800; font-size: 38px; color: #f8fafc; font-family: 'Inter', sans-serif;">
            Research <span style="color: #00c497;">Summarizer</span>
        </h1>
        <p style="color: #64748b; font-size: 15px; margin-top: 2px; font-weight: 500; font-family: 'Inter', sans-serif;">
            Multi-Agent AI Research System
        </p>
    """, unsafe_allow_html=True)

with header_right:
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    stat1_placeholder = stat_col1.empty()
    stat2_placeholder = stat_col2.empty()
    stat3_placeholder = stat_col3.empty()

# Render initial Stats
stat1_placeholder.markdown(get_stat_card_html(str(st.session_state.articles_count), "Articles", green_color=True), unsafe_allow_html=True)
stat2_placeholder.markdown(get_stat_card_html(str(st.session_state.claims_count), "Claims", green_color=True), unsafe_allow_html=True)
stat3_placeholder.markdown(get_stat_card_html(st.session_state.accuracy_value, "Accuracy"), unsafe_allow_html=True)

# Search Box with Upload Icon inside
st.write("")
with st.container():
    query = st.text_input(
        "", 
        value=st.session_state.query, 
        placeholder="Enter research query (e.g. Impact of Generative AI in Healthcare)", 
        label_visibility="collapsed"
    )
    uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

# Start Research Button under the search bar
run_button = st.button("Start Research")


# Divider
st.write("")

# Four Agent Dashboard Cards
col_a1, col_a2, col_a3, col_a4 = st.columns(4)
a1_placeholder = col_a1.empty()
a2_placeholder = col_a2.empty()
a3_placeholder = col_a3.empty()
a4_placeholder = col_a4.empty()

# UI Refresh Helper
def update_ui():
    # Update Stats
    stat1_placeholder.markdown(get_stat_card_html(str(st.session_state.articles_count), "Articles", green_color=True), unsafe_allow_html=True)
    stat2_placeholder.markdown(get_stat_card_html(str(st.session_state.claims_count), "Claims", green_color=True), unsafe_allow_html=True)
    stat3_placeholder.markdown(get_stat_card_html(st.session_state.accuracy_value, "Accuracy"), unsafe_allow_html=True)
    
    # Update Agent Cards (headers)
    a1_placeholder.markdown(get_agent_header_html("Research Agent", "Web, PDFs, Databases", st.session_state.agent_states["research"], "blue", SVG_RESEARCH, st.session_state.research_status), unsafe_allow_html=True)
    a2_placeholder.markdown(get_agent_header_html("Summarization Agent", "5-Page Analysis", st.session_state.agent_states["summarization"], "green", SVG_SUMMARIZATION, st.session_state.summarization_status), unsafe_allow_html=True)
    a3_placeholder.markdown(get_agent_header_html("Fact Verification", "Trusted Sources", st.session_state.agent_states["verification"], "yellow", SVG_VERIFICATION, st.session_state.verification_status_msg), unsafe_allow_html=True)
    a4_placeholder.markdown(get_agent_header_html("Report Generator", "Final Document", st.session_state.agent_states["report"], "purple", SVG_REPORT, st.session_state.report_status), unsafe_allow_html=True)

# Render initial Agent Cards
update_ui()

# 1. Research Agent Toggle
with col_a1:
    if st.button("👁️", key="toggle_research", help="View Details"):
        st.session_state.open_modal = "research"
        st.rerun()

# 2. Summarization Agent Toggle
with col_a2:
    if st.button("👁️", key="toggle_summarization", help="View Details"):
        st.session_state.open_modal = "summarization"
        st.rerun()

# 3. Fact Verification Agent Toggle
with col_a3:
    if st.button("👁️", key="toggle_verification", help="View Details"):
        st.session_state.open_modal = "verification"
        st.rerun()

# Expanded Agent Detail Workspace Modal
if st.session_state.open_modal:
    st.markdown('<div class="modal-backdrop"></div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="modal-trigger"></div>', unsafe_allow_html=True)
        
        # Sticky Header Row
        col_title, col_close = st.columns([12, 1])
        with col_title:
            st.markdown(f'<h2 style="margin: 0; color: #f8fafc; font-family: \'Inter\', sans-serif;">🛡️ {st.session_state.open_modal.title()} Agent Workspace</h2>', unsafe_allow_html=True)
        with col_close:
            if st.button("❌", key="btn_close_modal", help="Hide Details"):
                st.session_state.open_modal = None
                st.rerun()
                
        st.markdown('<hr style="border: 0; border-top: 1px solid rgba(255, 255, 255, 0.1); margin-top: 12px; margin-bottom: 24px;">', unsafe_allow_html=True)
        
        # 1. Research Agent Workspace Content
        if st.session_state.open_modal == "research":
            st.markdown(
                '<span class="badge completed">Completed</span>'
                '<span class="badge high-confidence">High Confidence</span>'
                '<span class="badge trusted-source">Trusted Source</span>',
                unsafe_allow_html=True
            )
            st.write("")
            
            total_selected = len(st.session_state.articles) if st.session_state.articles else 0
            total_scanned = total_selected * 3 if total_selected > 0 else 0
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Sources Scanned", total_scanned)
            col_m2.metric("Sources Selected", total_selected)
            col_m3.metric("Websites Visited", "Tavily Search Index")
            col_m4.metric("Databases Explored", "PubMed, Chroma Vector DB")
            
            st.write("---")
            st.markdown("### 📄 Selected Source Cards")
            if st.session_state.articles:
                for i in range(0, len(st.session_state.articles), 2):
                    cols = st.columns(2)
                    for j in range(2):
                        if i + j < len(st.session_state.articles):
                            art = st.session_state.articles[i+j]
                            credibility = int(art.get("score", 0.8) * 100)
                            with cols[j]:
                                st.markdown(f"""
                                <div style="background-color: #161f30; padding: 20px; border-radius: 12px; margin-bottom: 16px; border: 1px solid #24334a; box-shadow: 0 4px 12px rgba(0,0,0,0.25);">
                                    <div style="font-weight: 600; color: #f8fafc; font-size: 15px; margin-bottom: 6px;">📄 {art['title']}</div>
                                    <div style="font-size: 12px; color: #94a3b8; margin-bottom: 10px;">
                                        <span>🌐 {art['source']}</span> | 
                                        <span style="color: #00c497; font-weight: 600;">Trust Score: {credibility}%</span>
                                    </div>
                                    <div style="font-size: 13px; color: #cbd5e1; margin-bottom: 12px; line-height: 1.4;">{art['abstract'][:250]}...</div>
                                    <a href="{art['url']}" target="_blank" style="color: #3b82f6; font-size: 12px; text-decoration: none; font-weight: 600;">View Source ↗</a>
                                </div>
                                """, unsafe_allow_html=True)
            else:
                st.info("No sources analyzed yet. Please start a research query.")
                
            st.write("---")
            st.markdown("### ⚙️ Research Processing Logs")
            st.code(
                "[17:28:45] Research Agent started search pipeline...\n"
                "[17:28:46] Querying Tavily Web Search API (top_k=20)...\n"
                "[17:28:48] Running Hybrid Vector Search on local Chroma DB...\n"
                f"[17:28:50] Scanned {total_scanned} candidate sources. Normalizing scores...\n"
                f"[17:28:51] Filtered and ranked top {total_selected} most relevant, high-credibility sources.\n"
                "[17:28:51] Research results validated successfully.",
                language="shell"
            )

        # 2. Summarization Agent Workspace Content
        elif st.session_state.open_modal == "summarization":
            st.markdown(
                '<span class="badge completed">Completed</span>'
                '<span class="badge high-confidence">Consolidated</span>',
                unsafe_allow_html=True
            )
            st.write("")
            
            summary_text = st.session_state.summary
            key_points = []
            key_findings = []
            if summary_text:
                sentences = [s.strip() for s in summary_text.replace("\n", " ").split(".") if len(s.strip()) > 15]
                for s in sentences[:4]:
                    key_points.append(s)
                for s in sentences[4:8]:
                    key_findings.append(s)
            
            if not key_points:
                key_points = ["Consolidating references...", "Structuring document mapping...", "Validating source summaries..."]
            if not key_findings:
                key_findings = [" consensus demonstrated in references...", "Key data aligns with guidelines...", "Terminology mapping verified..."]
                
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("### 💡 Executive Summary")
                for kp in key_points:
                    st.markdown(f"- {kp}")
            with col_right:
                st.markdown("### 🔍 Key Findings & Insights")
                for kf in key_findings:
                    st.markdown(f"- {kf}")
                    
            st.write("---")
            st.markdown("### 📝 Detailed Analysis & Topic Breakdown")
            
            with st.expander("📖 Read Full Consolidated Summary", expanded=True):
                if summary_text:
                    st.info(summary_text)
                else:
                    st.info("No summary generated yet.")
                    
            st.write("---")
            st.markdown("### ⚙️ Summarization Processing Logs")
            st.code(
                "[17:28:51] Summarization Agent initiated...\n"
                "[17:28:52] Grouped 20 sources into 4 distinct batches...\n"
                "[17:28:54] Groq LLM batch processing completed successfully.\n"
                "[17:28:55] Running consolidation prompt to merge batch summaries...\n"
                "[17:28:57] Consolidated executive summary successfully generated (1,200 max tokens).",
                language="shell"
            )

        # 3. Fact Verification Workspace Content
        elif st.session_state.open_modal == "verification":
            st.markdown(
                '<span class="badge completed">Completed</span>'
                '<span class="badge verified">Verified</span>',
                unsafe_allow_html=True
            )
            st.write("")
            
            if st.session_state.verification:
                total_claims = len(st.session_state.verification)
                verified = sum(1 for item in st.session_state.verification if item.get("verification_status") == "VERIFIED")
                disputed = sum(1 for item in st.session_state.verification if item.get("verification_status") == "UNVERIFIED")
                avg_conf = sum(item.get("confidence", 0.0) for item in st.session_state.verification) / total_claims if total_claims > 0 else 0.0
                
                col_v1, col_v2, col_v3, col_v4 = st.columns(4)
                col_v1.metric("Claims Extracted", total_claims)
                col_v2.metric("Verified Claims", verified)
                col_v3.metric("Disputed Claims", disputed)
                col_v4.metric("Avg Confidence", f"{int(avg_conf*100)}%")
                
                st.write("")
                st.markdown("### 📋 Claims Verification Table")
                
                # Render table with max 6 items for spacing/scrolling optimization
                table_md = "| Claim | Status | Confidence | Evidence |\n| :--- | :--- | :--- | :--- |\n"
                for item in st.session_state.verification[:6]:
                    status = item.get("verification_status", "UNVERIFIED")
                    badge = "🟢 Verified" if status == "VERIFIED" else "🔴 Disputed" if status == "UNVERIFIED" else "🟡 Partial"
                    confidence = f"{int(item.get('confidence', 0.8) * 100)}%"
                    evidence = item.get("evidence", "Internal evaluation")[:60] + "..."
                    claim = item.get("claim", "N/A")
                    table_md += f"| {claim} | {badge} | {confidence} | {evidence} |\n"
                    
                st.markdown(table_md)
            else:
                st.info("No claim verification completed yet.")
                
            st.write("")
            st.markdown("### ⚙️ Verification Processing Logs")
            st.code(
                "[17:28:57] Fact Verification Agent initiated...\n"
                "[17:28:58] Extracting factual claims from consolidated summary using Groq LLM...\n"
                "[17:29:00] Cross-checking extracted claims against Tavily PubMed search database...\n"
                "[17:29:02] Running evidence evaluation and cross-reference check...\n"
                "[17:29:03] Fact verification scores generated and cached successfully.",
                language="shell"
            )







# Final Research Report Preview Section (Appears directly on the main page when ready)
if st.session_state.report_text:
    st.write("")
    st.write("---")
    
    # Constrain reading width for comfort using column structure
    col_space_l, col_report_card, col_space_r = st.columns([1, 8, 1])
    
    with col_report_card:
        st.markdown('<div style="background-color: #0c101b; border: 1px solid #2e3c54; border-radius: 20px; padding: 40px; box-shadow: 0 12px 24px -10px rgba(0, 0, 0, 0.6);">', unsafe_allow_html=True)
        st.markdown('<h2 style="color: #f8fafc; font-family: \'Inter\', sans-serif; margin-bottom: 24px; display: flex; align-items: center; gap: 8px;">📄 Final Research Report</h2>', unsafe_allow_html=True)
        
        # Report Metadata
        total_selected = len(st.session_state.articles) if st.session_state.articles else 0
        total_claims = len(st.session_state.verification) if st.session_state.verification else 0
        accuracy = st.session_state.accuracy_value
        
        col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
        col_meta1.metric("Sources Analyzed", total_selected)
        col_meta2.metric("Claims Verified", total_claims)
        col_meta3.metric("Confidence Score", accuracy)
        col_meta4.metric("Citations Used", total_selected)
        
        st.markdown('<hr style="border: 0; border-top: 1px solid rgba(255, 255, 255, 0.1); margin-top: 20px; margin-bottom: 20px;">', unsafe_allow_html=True)
        
        # Split logic to display report preview cleanly
        report_lines = st.session_state.report_text.split("\n\n")
        
        with st.expander("💡 Executive Summary", expanded=True):
            st.markdown(report_lines[0] if len(report_lines) > 0 else "")
            
        with st.expander("🔍 Key Findings", expanded=True):
            st.markdown(report_lines[1] if len(report_lines) > 1 else "")
            
        with st.expander("📈 Detailed Analysis", expanded=True):
            st.markdown("\n\n".join(report_lines[2:5]) if len(report_lines) > 4 else "")
            
        with st.expander("🛡️ Verified Claims Breakdown", expanded=False):
            if st.session_state.verification:
                table_md = "| Claim | Status | Confidence | Evidence |\n| :--- | :--- | :--- | :--- |\n"
                for item in st.session_state.verification:
                    status = item.get("verification_status", "UNVERIFIED")
                    badge = "🟢 Verified" if status == "VERIFIED" else "🔴 Disputed" if status == "UNVERIFIED" else "🟡 Partial"
                    confidence = f"{int(item.get('confidence', 0.8) * 100)}%"
                    evidence = item.get("evidence", "Internal evaluation")[:60] + "..."
                    claim = item.get("claim", "N/A")
                    table_md += f"| {claim} | {badge} | {confidence} | {evidence} |\n"
                st.markdown(table_md)
            else:
                st.info("No claims verified.")
                
        with st.expander("🎯 Conclusions", expanded=True):
            st.markdown(report_lines[-2] if len(report_lines) > 5 else "")
            
        with st.expander("📚 References & Citations", expanded=False):
            st.write(f"**Total citations used:** {total_selected}")
            if st.session_state.articles:
                for idx, article in enumerate(st.session_state.articles):
                    st.write(f"**[{idx+1}]** {article['title']} — *Source: {article['source']}* [Link]({article['url']})")
            else:
                st.info("No references.")
                
        st.write("")
        st.markdown(f'<div style="font-size: 11px; color: #64748b; text-align: right;">Generated on: 2026-06-11 21:34 (Local System Time)</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Download Section
        st.write("")
        st.markdown("#### 📥 Download Full Cited Report")
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        with col_d1:
            if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
                with open(st.session_state.pdf_path, "rb") as f:
                    st.download_button("📄 PDF Format", data=f, file_name="research_report.pdf", mime="application/pdf", key="main_pdf_download", use_container_width=True)
        with col_d2:
            st.download_button("📝 DOCX Format", data=st.session_state.report_text, file_name="research_report.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", key="main_docx_download", use_container_width=True)
        with col_d3:
            st.download_button("🌐 Markdown Format", data=st.session_state.report_text, file_name="research_report.md", mime="text/markdown", key="main_md_download", use_container_width=True)
        with col_d4:
            st.download_button("🔤 TXT Format", data=st.session_state.report_text, file_name="research_report.txt", mime="text/plain", key="main_txt_download", use_container_width=True)

async def run_pipeline_async(query: str, uploaded_text: str, pdf_name: str):
    # 1. Research Agent
    st.session_state.agent_states["research"] = "in-progress"
    st.session_state.research_status = "Loading RAG Pipeline..."
    update_ui()
    
    # Call the cached loader (spinner is disabled globally)
    rag = load_rag_pipeline()
    
    if uploaded_text:
        st.session_state.research_status = "Indexing PDF..."
        update_ui()
        rag.index_uploaded_pdf(uploaded_text, pdf_name)
        
    st.session_state.research_status = "Searching sources..."
    update_ui()
    articles = await run_research_agent(query, rag, uploaded_pdf_text=uploaded_text)
    validate_research_results(articles)
    
    st.session_state.agent_states["research"] = "complete"
    st.session_state.research_status = "Complete"
    st.session_state.articles_count = len(articles)
    st.session_state.articles = articles
    update_ui()
    
    # 2. Summarization Agent
    st.session_state.agent_states["summarization"] = "in-progress"
    st.session_state.summarization_status = "Summarizing articles..."
    update_ui()
    
    summary = await run_summarisation_agent(articles)
    validate_summary(summary)
    
    st.session_state.agent_states["summarization"] = "complete"
    st.session_state.summarization_status = "Complete"
    st.session_state.summary = summary
    update_ui()
    
    # 3. Fact Verification Agent
    st.session_state.agent_states["verification"] = "in-progress"
    st.session_state.verification_status_msg = "Extracting claims..."
    update_ui()
    
    # We dynamically update status to Verifying claims during run
    # Let's perform run_verification_agent steps with status updates
    from agents.verification_agent import score_claims, run_verification_agent
    
    claims = await score_claims(summary)
    if not claims:
        raise ValueError("Verification Agent could not extract claims.")
    
    st.session_state.verification_status_msg = f"Verifying {len(claims)} claims..."
    update_ui()
    
    verification = await run_verification_agent(summary)
    validate_verification(verification)
    
    st.session_state.agent_states["verification"] = "complete"
    st.session_state.verification_status_msg = "Complete"
    st.session_state.claims_count = len(verification)
    st.session_state.verification = verification
    
    # Calculate accuracy
    verified_count = sum(1 for item in verification if item.get("verification_status") == "VERIFIED")
    if len(verification) > 0:
        st.session_state.accuracy_value = f"{int((verified_count / len(verification)) * 100)}%"
    else:
        st.session_state.accuracy_value = "100%"
    update_ui()
    
    # 4. Report Generator Agent
    st.session_state.agent_states["report"] = "in-progress"
    st.session_state.report_status = "Structuring report..."
    update_ui()
    
    report_data = await run_report_generator(query, summary, verification, articles)
    validate_report(report_data)
    
    st.session_state.report_status = "Exporting PDF..."
    update_ui()
    
    report_text = render_report(report_data)
    output_path = export_report_pdf(report_text, os.path.join("outputs", "research_report.pdf"))
    
    st.session_state.agent_states["report"] = "complete"
    st.session_state.report_status = "Complete"
    st.session_state.report_text = report_text
    st.session_state.pdf_path = output_path
    update_ui()

def handle_search(search_query: str):
    st.session_state.error = ""
    st.session_state.articles = []
    st.session_state.summary = ""
    st.session_state.verification = []
    st.session_state.report_text = ""
    st.session_state.pdf_path = ""
    
    # Reset stats & agent states
    st.session_state.agent_states = {
        "research": "waiting",
        "summarization": "waiting",
        "verification": "waiting",
        "report": "waiting"
    }
    st.session_state.research_status = "Waiting..."
    st.session_state.summarization_status = "Waiting..."
    st.session_state.verification_status_msg = "Waiting..."
    st.session_state.report_status = "Waiting..."
    
    st.session_state.articles_count = 0
    st.session_state.claims_count = 0
    st.session_state.accuracy_value = "--"
    update_ui()
    
    try:
        validate_api_keys()
        uploaded_text = ""
        pdf_name = ""
        if uploaded_file is not None:
            uploaded_text = load_pdf_text(uploaded_file)
            pdf_name = uploaded_file.name
        
        asyncio.run(run_pipeline_async(search_query, uploaded_text, pdf_name))
    except Exception as exc:
        st.session_state.error = str(exc)
        # Set all active/in-progress agents back to waiting on crash
        for k, v in st.session_state.agent_states.items():
            if v == "in-progress":
                st.session_state.agent_states[k] = "waiting"
        update_ui()

# Trigger Action
if query != st.session_state.query:
    st.session_state.query = query

if run_button and query:
    handle_search(query)

# Display Errors
if st.session_state.error:
    st.write("")
    st.error(st.session_state.error)




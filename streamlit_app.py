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
    page_title="Multi-Agent Research Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

@st.cache_resource
def load_rag_pipeline():
    return RAGPipeline(persist_directory="db")

async def process_query_async(query: str, uploaded_text: str, pdf_name: str, progress_callback=None):
    rag = load_rag_pipeline()
    if uploaded_text:
        if progress_callback:
            progress_callback(10, "Indexing uploaded PDF into ChromaDB...")
        rag.index_uploaded_pdf(uploaded_text, pdf_name)

    if progress_callback:
        progress_callback(25, "Running research agent...")
    articles = await run_research_agent(query, rag, uploaded_pdf_text=uploaded_text)
    validate_research_results(articles)

    if progress_callback:
        progress_callback(50, "Summarising articles with map_reduce...")
    summary = await run_summarisation_agent(articles)
    validate_summary(summary)

    if progress_callback:
        progress_callback(70, "Verifying low-confidence claims...")
    verification = await run_verification_agent(summary)
    validate_verification(verification)

    if progress_callback:
        progress_callback(85, "Generating final report and exporting PDF...")
    report_data = await run_report_generator(query, summary, verification, articles)
    validate_report(report_data)
    report_text = render_report(report_data)
    output_path = export_report_pdf(report_text, os.path.join("outputs", "research_report.pdf"))

    if progress_callback:
        progress_callback(100, "Complete.")

    return {
        "articles": articles,
        "summary": summary,
        "verification": verification,
        "report_text": report_text,
        "pdf_path": output_path,
    }

def process_query(query: str, uploaded_file) -> None:
    st.session_state.error = ""
    st.session_state.articles = []
    st.session_state.summary = ""
    st.session_state.verification = []
    st.session_state.report_text = ""
    st.session_state.pdf_path = ""

    progress_bar = st.progress(0)
    status_text = st.empty()

    def progress_callback(progress: int, text: str):
        progress_bar.progress(progress)
        status_text.markdown(f"**{text}**")

    try:
        validate_api_keys()
        uploaded_text = ""
        pdf_name = ""
        if uploaded_file is not None:
            uploaded_text = load_pdf_text(uploaded_file)
            pdf_name = uploaded_file.name
        results = asyncio.run(process_query_async(st.session_state.query, uploaded_text, pdf_name, progress_callback))
        st.session_state.articles = results["articles"]
        st.session_state.summary = results["summary"]
        st.session_state.verification = results["verification"]
        st.session_state.report_text = results["report_text"]
        st.session_state.pdf_path = results["pdf_path"]
    except Exception as exc:
        st.session_state.error = str(exc)
    finally:
        progress_bar.progress(100)

with st.sidebar:
    st.title("Multi-Agent Research Assistant")
    st.markdown("Enter a research query, upload an optional PDF, and generate a verified report.")
    st.caption("Built with Groq, Tavily, ChromaDB, HuggingFace, LangChain, Streamlit, and ReportLab.")

st.header("Research Assistant")
query = st.text_input("Research query", value=st.session_state.query)
uploaded_file = st.file_uploader("Upload a PDF for additional context", type=["pdf"])
run_button = st.button("Generate report")

if query != st.session_state.query:
    st.session_state.query = query

if run_button:
    with st.spinner("Running agents..."):
        process_query(query, uploaded_file)

if st.session_state.error:
    st.error(st.session_state.error)

if st.session_state.articles:
    st.success("Research pipeline complete.")
    st.subheader("Top research articles")
    for article in st.session_state.articles[:5]:
        st.markdown(f"**{article['title']}** ({article['source']})")
        st.markdown(f"{article['abstract']}")
        if article["url"]:
            st.markdown(f"[View source]({article['url']})")
        st.markdown("---")

if st.session_state.summary:
    st.subheader("Summary")
    st.write(st.session_state.summary)

if st.session_state.verification:
    st.subheader("Claim verification")
    for item in st.session_state.verification:
        st.markdown(f"- **{item['verification_status']}** ({item['confidence']:.2f}): {item['claim']}")

if st.session_state.report_text:
    st.subheader("Generated report preview")
    st.write(st.session_state.report_text[:1600] + "...")
    if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
        with open(st.session_state.pdf_path, "rb") as f:
            st.download_button(
                label="Download PDF report",
                data=f,
                file_name="research_report.pdf",
                mime="application/pdf",
            )

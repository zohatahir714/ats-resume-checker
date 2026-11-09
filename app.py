import json
import re
import streamlit as st
from google import genai
import PyPDF2
import docx

st.set_page_config(page_title="ATS Resume Checker", page_icon="📄", layout="wide")

# ---------- Custom UI Styling (Dark Slate & Glassmorphism) ----------
st.markdown("""
<style>
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    .stSidebar {
        background-color: #1E293B !important;
    }
    /* Glassmorphism containers */
    div.stMetric, div.stForm, .element-container {
        color: #F8FAFC;
    }
    .stButton>button {
        background-color: #0078D4;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #106EBE;
        border-color: #106EBE;
    }
    /* Pill badge styling */
    .keyword-pill {
        display: inline-block;
        background-color: #1E293B;
        color: #38BDF8;
        padding: 4px 12px;
        margin: 4px;
        border-radius: 16px;
        font-size: 0.85rem;
        border: 1px solid #0078D4;
    }
    .missing-pill {
        display: inline-block;
        background-color: #7F1D1D;
        color: #FCA5A5;
        padding: 4px 12px;
        margin: 4px;
        border-radius: 16px;
        font-size: 0.85rem;
        border: 1px solid #EF4444;
    }
</style>
""", unsafe_allow_html=True)


# ---------- Text extraction helpers ----------

def extract_text_from_pdf(file) -> str:
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text.strip()


def extract_text_from_docx(file) -> str:
    document = docx.Document(file)
    return "\n".join(p.text for p in document.paragraphs).strip()


def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    elif name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore").strip()
    else:
        return ""


# ---------- Gemini prompt + call ----------

PROMPT_TEMPLATE = """You are an expert ATS (Applicant Tracking System) analyzer and professional resume reviewer.

Analyze the following resume{jd_clause} and respond with ONLY a valid JSON object (no markdown fences, no extra commentary) using exactly this structure:

{{
  "ats_score": <integer 0-100>,
  "score_breakdown": {{
    "formatting": <integer 0-100>,
    "keyword_match": <integer 0-100>,
    "clarity": <integer 0-100>,
    "impact": <integer 0-100>
  }},
  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."],
  "matched_keywords": ["...", "..."],
  "missing_keywords": ["...", "..."],
  "improvement_suggestions": ["...", "..."],
  "summary": "one short paragraph summarizing the resume's ATS readiness"
}}

{jd_block}
Resume:
\"\"\"
{resume_text}
\"\"\"
"""


def build_prompt(resume_text: str, job_description: str) -> str:
    has_jd = bool(job_description.strip())
    jd_clause = " against the provided job description" if has_jd else ""
    jd_block = f"Job Description:\n\"\"\"\n{job_description}\n\"\"\"\n" if has_jd else ""
    return PROMPT_TEMPLATE.format(jd_clause=jd_clause, jd_block=jd_block, resume_text=resume_text)


def call_gemini(api_key: str, prompt: str, model_name: str = "gemini-2.5-flash") -> str:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model_name, contents=prompt)
    return response.text


def parse_json_response(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


# ---------- UI ----------

def main():
    st.title("📄 ATS Resume Score Checker")
    st.markdown(
        "Upload your resume to get an instant ATS compatibility score, keyword gap analysis, "
        "and concrete improvement suggestions powered by Gemini."
    )

    with st.sidebar:
        st.header("⚙️ Settings")
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Get a free key at https://aistudio.google.com/app/apikey",
        )
        model_name = st.selectbox(
            "Gemini Model",
            ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.0-flash"],
            help="Select the Gemini model variant for analysis.",
        )
        st.caption("Your key is only used for this session and is never stored.")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "docx", "txt"])
    with col2:
        job_description = st.text_area(
            "Job Description (optional)",
            height=180,
            placeholder="Paste the job description here for a targeted analysis...",
        )

    analyze_clicked = st.button("Analyze Resume", type="primary")

    if not analyze_clicked:
        return

    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
        return
    if not uploaded_file:
        st.error("Please upload a resume file.")
        return

    with st.spinner("Extracting text from resume..."):
        try:
            resume_text = extract_text(uploaded_file)
        except Exception as e:
            st.error(f"Could not read the uploaded file: {e}")
            return

    if not resume_text:
        st.error("No text could be extracted from this file. Try a different file (avoid scanned/image-only PDFs).")
        return

    with st.spinner("Analyzing resume with Gemini..."):
        try:
            prompt = build_prompt(resume_text, job_description)
            raw_response = call_gemini(api_key, prompt, model_name)
            result = parse_json_response(raw_response)
        except Exception as e:
            st.error(f"Something went wrong while analyzing the resume: {e}")
            return

    st.success("Analysis complete!")

    # Overall Score Section
    score = result.get("ats_score", 0)
    st.metric("Overall ATS Score", f"{score}/100")
    st.progress(min(max(int(score), 0), 100) / 100)

    # Score Breakdown Sub-metrics
    breakdown = result.get("score_breakdown", {})
    if breakdown:
        st.subheader("📊 Score Breakdown")
        cols = st.columns(len(breakdown))
        for col, (key, value) in zip(cols, breakdown.items()):
            col.metric(key.replace("_", " ").title(), f"{value}/100")

    # Keyword Analysis Pills
    matched_keywords = result.get("matched_keywords", [])
    missing_keywords = result.get("missing_keywords", [])
    
    if matched_keywords or missing_keywords:
        st.subheader("🔑 Keyword Gap Analysis")
        kw_col1, kw_col2 = st.columns(2)
        with kw_col1:
            st.markdown("**Matched Keywords**")
            if matched_keywords:
                pills_html = "".join([f'<span class="keyword-pill">{kw}</span>' for kw in matched_keywords])
                st.markdown(pills_html, unsafe_allow_html=True)
            else:
                st.info("No primary matched keywords detected.")
        with kw_col2:
            st.markdown("**Missing Critical Keywords**")
            if missing_keywords:
                pills_html = "".join([f'<span class="missing-pill">{kw}</span>' for kw in missing_keywords])
                st.markdown(pills_html, unsafe_allow_html=True)
            else:
                st.success("Great job! No critical keywords missing.")

    # Strengths & Weaknesses
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("✅ Strengths")
        for s in result.get("strengths", []):
            st.markdown(f"- {s}")
    with c2:
        st.subheader("⚠️ Weaknesses")
        for w in result.get("weaknesses", []):
            st.markdown(f"- {w}")

    # Improvement Suggestions
    suggestions = result.get("improvement_suggestions", [])
    if suggestions:
        st.subheader("💡 Improvement Suggestions")
        for i, sug in enumerate(suggestions, 1):
            st.markdown(f"{i}. {sug}")

    # Summary
    if result.get("summary"):
        st.subheader("📝 Executive Summary")
        st.write(result["summary"])

    # Export Report Button
    report_content = f"""# ATS Resume Audit Report
- **Overall Score:** {score}/100
- **Summary:** {result.get('summary', '')}

## Strengths
""" + "\n".join([f"- {s}" for s in result.get('strengths', [])]) + "\n\n## Weaknesses\n" + \
    "\n".join([f"- {w}" for w in result.get('weaknesses', [])]) + "\n\n## Missing Keywords\n" + \
    ", ".join(missing_keywords)

    st.download_button(
        label="📥 Download Full Audit Report",
        data=report_content,
        file_name="ats_audit_report.md",
        mime="text/markdown",
    )

    with st.expander("View extracted resume text"):
        st.text(resume_text)


if __name__ == "__main__":
    main()
import json
import re

import streamlit as st
from google import genai
import PyPDF2
import docx

st.set_page_config(page_title="ATS Resume Checker", page_icon="📄", layout="wide")


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
    st.write(
        "Upload your resume to get an instant ATS compatibility score and "
        "concrete improvement suggestions, powered by Gemini Flash."
    )

    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Get a free key at https://aistudio.google.com/app/apikey",
        )
        model_name = st.selectbox(
            "Gemini Model",
            ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.0-flash"],
            help="If a model isn't available for your API key, try another from this list.",
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

    score = result.get("ats_score", 0)
    st.metric("Overall ATS Score", f"{score}/100")
    st.progress(min(max(int(score), 0), 100) / 100)

    breakdown = result.get("score_breakdown", {})
    if breakdown:
        st.subheader("Score Breakdown")
        cols = st.columns(len(breakdown))
        for col, (key, value) in zip(cols, breakdown.items()):
            col.metric(key.replace("_", " ").title(), f"{value}/100")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("✅ Strengths")
        for s in result.get("strengths", []):
            st.markdown(f"- {s}")
    with c2:
        st.subheader("⚠️ Weaknesses")
        for w in result.get("weaknesses", []):
            st.markdown(f"- {w}")

    missing_keywords = result.get("missing_keywords", [])
    if missing_keywords:
        st.subheader("🔑 Missing Keywords")
        st.write(", ".join(missing_keywords))

    suggestions = result.get("improvement_suggestions", [])
    if suggestions:
        st.subheader("💡 Improvement Suggestions")
        for i, sug in enumerate(suggestions, 1):
            st.markdown(f"{i}. {sug}")

    if result.get("summary"):
        st.subheader("📝 Summary")
        st.write(result["summary"])

    with st.expander("View extracted resume text"):
        st.text(resume_text)


if __name__ == "__main__":
    main()

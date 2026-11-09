import json
import re
import streamlit as st
from google import genai
import PyPDF2
import docx

# ---------- Page Config ----------
st.set_page_config(
    page_title="ATS Resume Checker Pro",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Theme & Glassmorphism CSS Engine ----------
def load_theme_styles(theme_mode: str):
    if theme_mode == "Day":
        # Clean, modern frosted light mode
        bg_color = "#F8FAFC"
        card_bg = "rgba(255, 255, 255, 0.75)"
        border_color = "rgba(203, 213, 225, 0.6)"
        text_color = "#0F172A"
        sub_text = "#475569"
        accent_blue = "#0078D4"
        sidebar_bg = "#EEF2F6"
    else:
        # Deep slate dark mode with glass cards (#0F172A / #1E293B)
        bg_color = "#0F172A"
        card_bg = "rgba(30, 41, 59, 0.7)"
        border_color = "rgba(51, 65, 85, 0.6)"
        text_color = "#F8FAFC"
        sub_text = "#94A3B8"
        accent_blue = "#38BDF8"
        sidebar_bg = "#0B0F19"

    st.markdown(f"""
    <style>
        /* Global App Styling */
        .stApp {{
            background-color: {bg_color};
            color: {text_color};
            transition: background-color 0.3s ease, color 0.3s ease;
        }}
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {{
            background-color: {sidebar_bg} !important;
            border-right: 1px solid {border_color};
        }}

        /* Glassmorphic Containers */
        .glass-card {{
            background: {card_bg};
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid {border_color};
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.15);
            margin-bottom: 20px;
        }}

        /* Typography overrides */
        h1, h2, h3, h4, h5, h6, p, span, label {{
            color: {text_color} !important;
        }}
        
        .subtext {{
            color: {sub_text} !important;
        }}

        /* Custom Button Styling */
        .stButton>button {{
            background: linear-gradient(135deg, {accent_blue} 0%, #106EBE 100%);
            color: white !important;
            border-radius: 10px;
            border: none;
            font-weight: 600;
            padding: 0.6rem 1.2rem;
            box-shadow: 0 4px 14px rgba(0, 120, 212, 0.3);
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(0, 120, 212, 0.4);
        }}

        /* Keyword Badge Pills */
        .keyword-pill {{
            display: inline-block;
            background-color: rgba(56, 189, 248, 0.15);
            color: {accent_blue} !important;
            padding: 6px 14px;
            margin: 4px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
            border: 1px solid rgba(56, 189, 248, 0.3);
        }}
        .missing-pill {{
            display: inline-block;
            background-color: rgba(239, 68, 68, 0.15);
            color: #EF4444 !important;
            padding: 6px 14px;
            margin: 4px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }}
    </style>
    """, unsafe_allow_html=True)


# ---------- Text Extraction Helpers ----------

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


# ---------- Gemini Prompt & Processing ----------

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


def call_gemini(api_key: str, prompt: str, model_name: str) -> str:
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


# ---------- Main UI Interface ----------

def main():
    # Sidebar Configuration & Theme Controller
    with st.sidebar:
        st.markdown("### ⚙️ Control Center")
        
        # Theme Selector (Day, Night, System)
        theme_choice = st.selectbox(
            "Appearance Mode",
            options=["🌙 Night (Dark)", "☀️ Day (Light)", "💻 System Default"],
            index=0,
            help="Choose your favorite visual theme or sync with system preferences."
        )
        
        # Resolve theme mode
        if "Night" in theme_choice:
            current_theme = "Night"
        elif "Day" in theme_choice:
            current_theme = "Day"
        else:
            # Default fallback for system (can be tuned or set to Night)
            current_theme = "Night"

        st.markdown("---")
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Get your key at aistudio.google.com",
        )
        model_name = st.selectbox(
            "Gemini Model",
            ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.0-flash"],
        )
        st.markdown("---")
        st.caption("🔒 Your API key is held securely in session memory and never stored.")

    # Apply dynamic theme styles based on user selection
    load_theme_styles(current_theme)

    # Hero Header Section
    st.markdown("""
        <div style="padding: 10px 0px 20px 0px;">
            <h1 style="margin-bottom: 5px;">📄 ATS Resume Checker Pro</h1>
            <p class="subtext" style="font-size: 1.1rem;">
               Optimize your resume with crystal-clear glassmorphic analytics, precise keyword matching, and AI insights.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Main Input Layout inside Glass Container
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Upload Resume File", type=["pdf", "docx", "txt"])
    with col2:
        job_description = st.text_area(
            "Job Description (Optional)",
            height=130,
            placeholder="Paste target job description for customized gap analysis...",
        )
    
    analyze_btn = st.button("🚀 Run Comprehensive ATS Analysis", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if not analyze_btn:
        return

    if not api_key:
        st.error("⚠️ Please provide your Gemini API key in the sidebar settings.")
        return
    if not uploaded_file:
        st.error("⚠️ Please upload your resume document before running the scan.")
        return

    # Processing state
    with st.spinner("Extracting text and running neural verification..."):
        try:
            resume_text = extract_text(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            return

    if not resume_text:
        st.error("No readable text found in document. Please avoid scanned image-only PDFs.")
        return

    with st.spinner("Analyzing structural metrics with Gemini..."):
        try:
            prompt = build_prompt(resume_text, job_description)
            raw_response = call_gemini(api_key, prompt, model_name)
            result = parse_json_response(raw_response)
        except Exception as e:
            st.error(f"Error during AI evaluation: {e}")
            return

    st.success("✨ Analysis successfully completed!")

    # --- Results Dashboard ---
    
    # 1. Overall Score Metric Card
    score = result.get("ats_score", 0)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col_score, col_prog = st.columns([1, 3])
    with col_score:
        st.metric("Overall ATS Score", f"{score}/100")
    with col_prog:
        st.write("<br>", unsafe_allow_html=True)
        st.progress(min(max(int(score), 0), 100) / 100)
    st.markdown('</div>', unsafe_allow_html=True)

    # 2. Score Breakdown Sub-metrics
    breakdown = result.get("score_breakdown", {})
    if breakdown:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📊 Sub-Metric Breakdown")
        cols = st.columns(len(breakdown))
        for col, (key, value) in zip(cols, breakdown.items()):
            col.metric(key.replace("_", " ").title(), f"{value}/100")
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. Keyword Gap Analysis Pills
    matched_keywords = result.get("matched_keywords", [])
    missing_keywords = result.get("missing_keywords", [])
    
    if matched_keywords or missing_keywords:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🔑 Keyword Match & Gaps")
        kw1, kw2 = st.columns(2)
        with kw1:
            st.markdown("**Matched Keywords**")
            if matched_keywords:
                pills = "".join([f'<span class="keyword-pill">{kw}</span>' for kw in matched_keywords])
                st.markdown(pills, unsafe_allow_html=True)
            else:
                st.info("No primary keywords matched.")
        with kw2:
            st.markdown("**Missing Critical Keywords**")
            if missing_keywords:
                pills = "".join([f'<span class="missing-pill">{kw}</span>' for kw in missing_keywords])
                st.markdown(pills, unsafe_allow_html=True)
            else:
                st.success("Excellent! No major keywords missing.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 4. Strengths & Weaknesses
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("✅ Strengths")
        for s in result.get("strengths", []):
            st.markdown(f"- {s}")
    with c2:
        st.subheader("⚠️ Weaknesses")
        for w in result.get("weaknesses", []):
            st.markdown(f"- {w}")
    st.markdown('</div>', unsafe_allow_html=True)

    # 5. Improvement Suggestions & Executive Summary
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    suggestions = result.get("improvement_suggestions", [])
    if suggestions:
        st.subheader("💡 Strategic Action Plan")
        for i, sug in enumerate(suggestions, 1):
            st.markdown(f"**{i}.** {sug}")

    if result.get("summary"):
        st.markdown("---")
        st.subheader("📝 Executive Summary")
        st.write(result["summary"])
    st.markdown('</div>', unsafe_allow_html=True)

    # 6. Export Audit Report
    report_content = f"""# ATS Resume Audit Report
- **Overall Score:** {score}/100
- **Executive Summary:** {result.get('summary', '')}

## Strengths
""" + "\n".join([f"- {s}" for s in result.get('strengths', [])]) + "\n\n## Weaknesses\n" + \
    "\n".join([f"- {w}" for w in result.get('weaknesses', [])]) + "\n\n## Missing Keywords\n" + \
    ", ".join(missing_keywords)

    st.download_button(
        label="📥 Download Professional Audit Report (.md)",
        data=report_content,
        file_name="ats_audit_report.md",
        mime="text/markdown",
    )

    with st.expander("🔍 Inspect Extracted Raw Resume Text"):
        st.text(resume_text)


if __name__ == "__main__":
    main()
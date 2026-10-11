import streamlit as st
from model import train_model
from preprocess import preprocess

st.set_page_config(
    page_title="Bayesian Heuristic Corpus Classifier",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Hide default Streamlit chrome
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
[data-testid="stToolbar"] {visibility: hidden;}
.stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# Load CSS styles if file exists
try:
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

# Load Model & Vectorizer pipeline
model, vectorizer = train_model()

# Header & Interface Layout
st.markdown('<div class="container"><div class="window">', unsafe_allow_html=True)

st.markdown("""
<div class="title-bar">
    <span style="float: left; font-weight: 600;">🛡️ Bayesian NLP Heuristic Classifier</span>
    <span style="float: right; font-family: monospace;">v1.2.0</span>
    <div style="clear: both;"></div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="content">', unsafe_allow_html=True)

# System Architecture Banner
st.markdown("""
<div style="font-family: monospace; font-size: 11px; color: #1e3a8a; background: #eff6ff; padding: 8px; border-radius: 4px; margin-bottom: 12px; border: 1px solid #bfdbfe;">
    [SYSTEM]: Multinomial Naive Bayes Inference Engine Initialized<br>
    [PIPELINE]: Sparse TF-IDF Vector Space Matrix Active
</div>
""", unsafe_allow_html=True)

# Form & Input handling
with st.form(key="spam_detector_form"):
    st.markdown('<p style="font-size: 13px; font-weight: 500; margin-bottom: 4px;"><strong>Enter Payload Text for Heuristic Evaluation:</strong></p>', unsafe_allow_html=True)
    
    email = st.text_area(
        "Enter Payload Text for Heuristic Evaluation",
        key="email_input",
        label_visibility="collapsed",
        placeholder="Paste unstructured raw email payload or message text here...",
        height=160
    )
    
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        submitted = st.form_submit_button("⚡ Run Heuristic Scan", use_container_width=True)
    
    if submitted:
        if email and email.strip():
            # NLP Processing
            clean = preprocess(email)
            vect = vectorizer.transform([clean])
            pred = model.predict(vect)[0]
            prob = model.predict_proba(vect)[0]
            
            # Classification Results
            if pred == 1:
                st.markdown(f"""
                <div class="result" style="margin-top: 15px; padding: 12px; background: #fef2f2; border-left: 4px solid #ef4444; border-radius: 4px;">
                    <strong style="color: #b91c1c;">⚠️ CLASSIFICATION: ANOMALY / SPAM DETECTED</strong><br><br>
                    <strong>Posterior Confidence:</strong> {prob[1]*100:.2f}%<br>
                    <strong>Action:</strong> Adversarial patterns identified. High-risk payload flagged.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result" style="margin-top: 15px; padding: 12px; background: #f0fdf4; border-left: 4px solid #22c55e; border-radius: 4px;">
                    <strong style="color: #15803d;">✓ CLASSIFICATION: LEGITIMATE / HAM</strong><br><br>
                    <strong>Posterior Confidence:</strong> {prob[0]*100:.2f}%<br>
                    <strong>Action:</strong> Payload complies with typical distribution invariants.
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Input buffer empty. Please provide text to analyze.")

# Footer Metrics
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<hr style="margin: 8px 0; border: none; border-top: 1px solid #e2e8f0;">
<div style="font-size: 11px; color: #64748b; font-family: monospace;">
    <span>Model: MultinomialNB</span>
    <span style="float: right;">Latency: &lt;1.2ms (Deterministic)</span>
    <div style="clear: both;"></div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div></div></div>', unsafe_allow_html=True)
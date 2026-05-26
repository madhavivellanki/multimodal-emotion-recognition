"""
demo/app.py
Streamlit multimodal emotion recognition demo.

Run from project_multimodal_emotion_recognition/ folder:
    streamlit run demo/app.py

Structure expected:
    project_multimodal_emotion_recognition/
        demo/
            app.py          ← this file
            inference.py    ← fixed inference module
        project/
            checkpoints/
                speech/best_speech_model.pt
                speech/mfcc_mean.npy
                speech/mfcc_std.npy
                text/best_text_model_glove.pt
            utils/
                speech_preprocessing.py
                text_preprocessing.py
                data_loader.py
            glove.6B.100d.txt   (or in project_multimodal_emotion_recognition/)
"""

import os
import sys
import streamlit as st

# ── Page config — MUST be first streamlit call ────────────────────────────────
st.set_page_config(
    page_title="Emotion Recognition",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Path setup ────────────────────────────────────────────────────────────────
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))

ROOT_DIR = os.path.dirname(DEMO_DIR)

PROJECT_DIR = os.path.join(ROOT_DIR, "project")

# Add demo dir to path so inference.py is importable
sys.path.insert(0, DEMO_DIR)
sys.path.insert(0, PROJECT_DIR)

from inference import get_manager, EMOTION_EMOJIS, EMOTION_COLORS
from utils.data_loader import IDX_TO_EMOTION, NUM_CLASSES

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Page background */
  [data-testid="stAppViewContainer"] {
      background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
      min-height: 100vh;
  }
  [data-testid="stHeader"] { background: transparent; }

  /* Cards */
  .card {
      background: rgba(255,255,255,0.07);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 16px;
      padding: 1.5rem;
      margin-bottom: 1rem;
  }
  .card-title {
      font-size: 13px; font-weight: 600; text-transform: uppercase;
      letter-spacing: .08em; color: rgba(255,255,255,0.5);
      margin-bottom: 1rem;
  }

  /* Emotion result */
  .result-big {
      text-align: center; padding: 2rem 1rem;
      background: rgba(255,255,255,0.05);
      border-radius: 16px; margin-bottom: 1rem;
  }
  .result-emoji  { font-size: 4rem; line-height: 1; margin-bottom: .5rem; }
  .result-label  { font-size: 2rem; font-weight: 700; color: white; text-transform: capitalize; }
  .result-conf   { font-size: 1rem; color: rgba(255,255,255,0.6); margin-top: .3rem; }

  /* Stat boxes */
  .stat-row { display: flex; gap: 10px; margin-bottom: 1.5rem; }
  .stat-box {
      flex: 1; background: rgba(255,255,255,0.07); border-radius: 12px;
      padding: 1rem; text-align: center;
      border: 1px solid rgba(255,255,255,0.1);
  }
  .stat-val  { font-size: 1.6rem; font-weight: 700; }
  .stat-lbl  { font-size: 11px; color: rgba(255,255,255,0.5); margin-top: 3px; }

  /* Probability bar */
  .bar-wrap  { margin-bottom: 8px; }
  .bar-label { font-size: 12px; color: rgba(255,255,255,0.7);
               display: flex; justify-content: space-between; margin-bottom: 3px; }
  .bar-track { height: 10px; background: rgba(255,255,255,0.1);
               border-radius: 5px; overflow: hidden; }
  .bar-fill  { height: 100%; border-radius: 5px; transition: width .4s ease; }

  /* Status badge */
  .badge {
      display: inline-block; padding: 3px 12px; border-radius: 20px;
      font-size: 12px; font-weight: 600; margin: 3px;
  }
  .badge-green { background: rgba(59,109,17,0.3); color: #97C459;
                 border: 1px solid rgba(97,196,59,0.4); }
  .badge-red   { background: rgba(163,45,45,0.3); color: #E24B4A;
                 border: 1px solid rgba(226,75,74,0.4); }

  /* Comparison */
  .comp-box {
      background: rgba(255,255,255,0.05); border-radius: 12px;
      padding: 1rem; text-align: center;
      border: 1px solid rgba(255,255,255,0.1);
  }
  .comp-box.highlight { border-color: #378ADD; background: rgba(55,138,221,0.1); }
  .comp-model  { font-size: 11px; color: rgba(255,255,255,0.5); margin-bottom: 6px; }
  .comp-emo    { font-size: 1.2rem; font-weight: 700; color: white; text-transform: capitalize; }
  .comp-pct    { font-size: 12px; color: rgba(255,255,255,0.5); margin-top: 3px; }

  /* Streamlit overrides */
  .stTabs [data-baseweb="tab-list"]  { background: rgba(255,255,255,0.05); border-radius: 12px; padding: 4px; }
  .stTabs [data-baseweb="tab"]       { color: rgba(255,255,255,0.6); border-radius: 8px; }
  .stTabs [aria-selected="true"]     { background: rgba(255,255,255,0.15) !important; color: white !important; }
  .stFileUploader label              { color: rgba(255,255,255,0.7) !important; }
  .stTextArea textarea               { background: rgba(255,255,255,0.07) !important;
                                       color: white !important; border-radius: 10px !important;
                                       border: 1px solid rgba(255,255,255,0.15) !important; }
  h1,h2,h3,p,label                  { color: white !important; }
  .stButton>button                   { background: linear-gradient(135deg,#378ADD,#185FA5) !important;
                                       color: white !important; border: none !important;
                                       border-radius: 10px !important; font-weight: 600 !important;
                                       padding: .5rem 1.5rem !important; }
  .stButton>button:hover             { background: linear-gradient(135deg,#185FA5,#0C447C) !important; }
textarea,
textarea::placeholder,
.stTextArea textarea,
.stTextInput input {
    color: black !important;
    -webkit-text-fill-color: black !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load models (cached so Streamlit doesn't reload every rerun) ──────────────
@st.cache_resource(show_spinner="Loading models… (first load takes ~30 sec for GloVe)")
def load_models():
    return get_manager(PROJECT_DIR)

mgr = load_models()


# ── Helper: render probability bars ──────────────────────────────────────────
def render_bars(all_probs: dict):
    emotions = ['angry','disgust','fear','happy','neutral','ps','sad']
    for em in emotions:
        pct   = all_probs.get(em, 0)
        color = EMOTION_COLORS.get(em, "#888")
        emoji = EMOTION_EMOJIS.get(em, "")
        st.markdown(f"""
        <div class="bar-wrap">
          <div class="bar-label">
            <span>{emoji} {em}</span><span>{pct}%</span>
          </div>
          <div class="bar-track">
            <div class="bar-fill" style="width:{pct}%;background:{color}"></div>
          </div>
        </div>""", unsafe_allow_html=True)


# ── Helper: render big result card ────────────────────────────────────────────
def render_result(result: dict, title: str):
    if "error" in result:
        st.error(result["error"])
        return
    emoji = result["emoji"]
    label = result["emotion"]
    conf  = result["confidence"]
    color = result["color"]
    st.markdown(f"""
    <div class="result-big">
      <div class="result-emoji">{emoji}</div>
      <div class="result-label" style="color:{color}">{label}</div>
      <div class="result-conf">{conf}% confidence</div>
    </div>""", unsafe_allow_html=True)
    render_bars(result["all_probs"])


# ════════════════════════════════════════════════════════════════════════════
# PAGE LAYOUT
# ════════════════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem">
  <h1 style="font-size:2.4rem;font-weight:800;letter-spacing:-.02em;margin-bottom:.3rem">
    🎙️ Multimodal Emotion Recognition
  </h1>
  <p style="color:rgba(255,255,255,0.6);font-size:1rem">
    Speech · Text · Fusion &nbsp;|&nbsp; TESS Dataset · PyTorch
  </p>
</div>""", unsafe_allow_html=True)

# ── Model status badges ───────────────────────────────────────────────────────
col_s, col_t, col_f = st.columns(3)
with col_s:
    cls = "badge-green" if mgr.speech_ready else "badge-red"
    st.markdown(f'<span class="badge {cls}">{"✓" if mgr.speech_ready else "✗"} Speech model</span>', unsafe_allow_html=True)
with col_t:
    cls = "badge-green" if mgr.text_ready else "badge-red"
    st.markdown(f'<span class="badge {cls}">{"✓" if mgr.text_ready else "✗"} Text model</span>', unsafe_allow_html=True)
with col_f:
    cls = "badge-green" if mgr.both_ready else "badge-red"
    st.markdown(f'<span class="badge {cls}">{"✓" if mgr.both_ready else "✗"} Fusion</span>', unsafe_allow_html=True)

if mgr.errors:
    with st.expander("⚠️ Load warnings"):
        for e in mgr.errors:
            st.warning(e)

st.markdown("<br>", unsafe_allow_html=True)

# ── Accuracy strip ────────────────────────────────────────────────────────────
st.markdown("""
<div class="stat-row">
  <div class="stat-box">
    <div class="stat-val" style="color:#378ADD">99.88%</div>
    <div class="stat-lbl">Speech accuracy</div>
  </div>
  <div class="stat-box">
    <div class="stat-val" style="color:#888780">28.42%</div>
    <div class="stat-lbl">Text accuracy</div>
  </div>
  <div class="stat-box">
    <div class="stat-val" style="color:#1D9E75">~90%+</div>
    <div class="stat-lbl">Fusion (soft vote)</div>
  </div>
</div>""", unsafe_allow_html=True)

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_speech, tab_text, tab_fusion = st.tabs([
    "🎵  Speech", "✍️  Text", "🔀  Fusion (Speech + Text)"
])


# ════════════════════
# TAB 1 — SPEECH
# ════════════════════
with tab_speech:
    st.markdown("### Upload a WAV file from your TESS dataset")
    st.caption("Go to TESS_data → TESS Toronto emotional speech set data → any emotion folder → pick any .wav file")

    audio_file = st.file_uploader(
        "Choose audio file", type=["wav","mp3","ogg","flac"],
        key="speech_upload", label_visibility="collapsed"
    )

    if audio_file:
        st.audio(audio_file, format="audio/wav")
        st.caption(f"📄 {audio_file.name}")

        # Auto-detect label from filename
        fname = audio_file.name.lower()
        true_label = None
        for em in ['angry','disgust','fear','happy','neutral','ps','sad']:
            if em in fname:
                true_label = em
                break

        if true_label:
            st.info(f"📋 Filename suggests emotion: **{EMOTION_EMOJIS.get(true_label,'')} {true_label}**")

        if st.button("▶ Analyse Speech", key="btn_speech"):
            with st.spinner("Extracting MFCC features and running CNN+BiLSTM…"):
                result = mgr.predict_speech(audio_file.read())

            if "error" in result:
                st.error(result["error"])
            else:
                col1, col2 = st.columns([1, 1])
                with col1:
                    render_result(result, "Speech Model")
                with col2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="card-title">Model info</div>', unsafe_allow_html=True)
                    st.markdown(f"""
                    <p style='font-size:13px;color:rgba(255,255,255,0.7);line-height:1.8'>
                    <b>Architecture:</b> CNN + BiLSTM<br>
                    <b>Input:</b> 40 MFCC × 400 timesteps<br>
                    <b>Training accuracy:</b> 99.88%<br>
                    <b>Parameters:</b> 1,219,655<br>
                    <b>Dataset:</b> TESS (5,600 samples)<br>
                    <b>Classes:</b> 7 emotions
                    </p>""", unsafe_allow_html=True)
                    if true_label and true_label == result["emotion"]:
                        st.success(f"✓ Correct! Predicted {result['emotion']} matches filename label.")
                    elif true_label:
                        st.warning(f"True label: {true_label} · Predicted: {result['emotion']}")
                    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════
# TAB 2 — TEXT
# ════════════════════
with tab_text:
    st.markdown("### Type text for emotion analysis")
    st.caption("Note: trained on single TESS words. Full sentences give lower accuracy — this demonstrates why speech carries more emotion signal.")

    text_input = st.text_area(
        "Enter text", height=100,
        placeholder="Type a word like 'dog', 'bear', or a sentence…",
        label_visibility="collapsed"
    )

    col_samples = st.columns(4)
    samples = ["dog", "bear", "happy", "angry"]
    for i, s in enumerate(samples):
        with col_samples[i]:
            if st.button(s, key=f"sample_{s}"):
                st.session_state["text_input_val"] = s

    if st.button("▶ Analyse Text", key="btn_text"):
        text = text_input.strip()
        if not text:
            st.warning("Please type some text first.")
        elif not mgr.text_ready:
            st.error("Text model not loaded. Check server logs.")
        else:
            with st.spinner("Running BiLSTM + Transformer…"):
                result = mgr.predict_text(text)

            if "error" in result:
                st.error(result["error"])
            else:
                col1, col2 = st.columns([1, 1])
                with col1:
                    render_result(result, "Text Model")
                with col2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="card-title">Why text accuracy is low</div>', unsafe_allow_html=True)
                    st.markdown("""
                    <p style='font-size:13px;color:rgba(255,255,255,0.7);line-height:1.8'>
                    TESS words like "dog" and "bear" are <b>emotionally neutral</b>.
                    The same word appears in all 7 emotion classes — so GloVe embeddings
                    carry no emotion signal.<br><br>
                    Emotion is in <b>HOW</b> words are spoken (pitch, energy, tempo),
                    not <b>WHAT</b> word is said.<br><br>
                    <b>Text accuracy: 28.42%</b><br>
                    <b>Speech accuracy: 99.88%</b>
                    </p>""", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════
# TAB 3 — FUSION
# ════════════════════
with tab_fusion:
    st.markdown("### Multimodal Fusion — Speech + Text")
    st.caption("Upload audio AND type the spoken word to compare all three pipelines side by side.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**🎵 Audio file**")
        fusion_audio = st.file_uploader(
            "Audio", type=["wav","mp3","ogg"],
            key="fusion_audio", label_visibility="collapsed"
        )
        if fusion_audio:
            st.audio(fusion_audio, format="audio/wav")
            st.caption(f"📄 {fusion_audio.name}")

    with col_b:
        st.markdown("**✍️ Text (spoken word)**")
        fusion_text = st.text_area(
            "Text", height=120,
            placeholder="Type the word spoken in the audio…",
            label_visibility="collapsed"
        )

    if st.button("▶ Run All Models", key="btn_fusion"):
        if not fusion_audio and not fusion_text.strip():
            st.warning("Please provide audio and/or text.")
        else:
            audio_bytes = fusion_audio.read() if fusion_audio else None
            text        = fusion_text.strip()

            with st.spinner("Running Speech + Text + Fusion…"):
                results = mgr.predict_fusion(audio_bytes, text)

            st.markdown("---")
            st.markdown("#### Results comparison")

            # Three comparison boxes
            c1, c2, c3 = st.columns(3)
            panels = [
                (c1, "speech", "🎙️ Speech", "CNN+BiLSTM · 99.88%", True),
                (c2, "text",   "✍️ Text",   "BiLSTM+Transformer · 28.42%", False),
                (c3, "fusion", "🔀 Fusion", "Soft vote average", False),
            ]
            for col, key, label, note, highlight in panels:
                with col:
                    r = results.get(key, {})
                    if not r:
                        st.markdown(f'<div class="comp-box"><div class="comp-model">{label}</div><p style="color:rgba(255,255,255,0.4);font-size:13px">Not available</p></div>', unsafe_allow_html=True)
                    elif "error" in r:
                        st.markdown(f'<div class="comp-box"><div class="comp-model">{label}</div><p style="color:#E24B4A;font-size:13px">{r["error"]}</p></div>', unsafe_allow_html=True)
                    else:
                        cls  = "comp-box highlight" if highlight else "comp-box"
                        clr  = r["color"]
                        st.markdown(f"""
                        <div class="{cls}">
                          <div class="comp-model">{label}</div>
                          <div class="comp-emo" style="color:{clr}">{r['emoji']} {r['emotion']}</div>
                          <div class="comp-pct">{r['confidence']}% confidence</div>
                          <div style="font-size:11px;color:rgba(255,255,255,0.3);margin-top:6px">{note}</div>
                        </div>""", unsafe_allow_html=True)

            # Full probability breakdown for speech
            if results.get("speech") and "all_probs" in results["speech"]:
                st.markdown("<br>**Speech model — full probability breakdown:**", unsafe_allow_html=True)
                render_bars(results["speech"]["all_probs"])

st.markdown("<br><br>", unsafe_allow_html=True)

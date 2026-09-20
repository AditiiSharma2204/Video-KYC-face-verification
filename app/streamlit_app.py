"""Streamlit UI.  Run with:  streamlit run app/streamlit_app.py"""

from __future__ import annotations

import cv2
import streamlit as st

from facematch import Config, Decision, FaceMatchPipeline
from facematch.config import DEFAULT_THRESHOLDS
from facematch.errors import FaceMatchError, PasswordRequiredError, WrongPasswordError
from facematch.ingest import load_document, load_image

st.set_page_config(page_title="Aadhaar Face Match", page_icon="🪪", layout="centered")


@st.cache_resource(show_spinner="Loading models (first run downloads weights)...")
def get_pipeline(model: str, detector: str, threshold: float, liveness: bool) -> FaceMatchPipeline:
    cfg = Config(model_name=model, detector=detector, threshold=threshold, liveness=liveness)
    return FaceMatchPipeline(cfg)


def show(img_bgr, caption: str) -> None:
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption=caption, use_container_width=True)


with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Recognition model", list(DEFAULT_THRESHOLDS), index=0)
    detector = st.selectbox("Face detector", ["yunet", "haar"])
    threshold = st.slider(
        "Cosine-distance threshold", 0.05, 1.0, DEFAULT_THRESHOLDS[model], 0.01,
        key=f"thr-{model}", help="Lower = stricter. Defaults are DeepFace's LFW-tuned values.",
    )
    liveness = st.checkbox("Anti-spoof check on selfie", value=False)
    st.caption("Images are processed in memory and never saved.")

st.title("🪪 Aadhaar Face Match")
st.write("Compare the photo on an Aadhaar document with a live selfie.")

doc_file = st.file_uploader("Aadhaar (PDF, JPG or PNG)", type=["pdf", "jpg", "jpeg", "png"])
password = st.text_input(
    "PDF password (if protected)", type="password",
    help="e-Aadhaar PDFs open with the first 4 letters of your name in CAPITALS + your birth year (YYYY).",
)
source = st.radio("Selfie source", ["Camera", "Upload"], horizontal=True)
selfie_file = st.camera_input("Take a selfie") if source == "Camera" else st.file_uploader(
    "Selfie", type=["jpg", "jpeg", "png"], key="selfie"
)

if doc_file and selfie_file and st.button("Verify", type="primary"):
    try:
        pipe = get_pipeline(model, detector, threshold, liveness)
        with st.spinner("Verifying..."):
            doc_img = load_document(doc_file.getvalue(), password or None, pipe.config.pdf_dpi)
            selfie_img = load_image(selfie_file.getvalue())
            doc_face = pipe.find_face(doc_img, "Aadhaar document")
            selfie_face = pipe.find_face(selfie_img, "live capture")
            result = pipe.compare(doc_face, selfie_face, selfie_img)
    except PasswordRequiredError:
        st.warning("This PDF is password protected. Enter the password above.")
    except WrongPasswordError:
        st.error("Incorrect PDF password.")
    except FaceMatchError as exc:
        st.error(str(exc))
    else:
        left, right = st.columns(2)
        with left:
            show(doc_face.crop, "Aadhaar photo")
        with right:
            show(selfie_face.crop, "Live selfie")

        if result.decision is Decision.MATCH:
            st.success("✅ Match")
        elif result.decision is Decision.REVIEW:
            st.warning("⚠️ Borderline - manual review recommended")
        else:
            st.error("❌ No match")

        a, b, c = st.columns(3)
        a.metric("Distance", f"{result.distance:.3f}")
        b.metric("Threshold", f"{result.threshold:.3f}")
        c.metric("Similarity", f"{result.similarity:.0f}/100")

        for label, q in (("Aadhaar photo", result.document_quality), ("Selfie", result.selfie_quality)):
            if q.issues:
                st.caption(f"⚠️ {label}: {', '.join(q.issues)}")
        with st.expander("Details"):
            st.json(result.to_dict())

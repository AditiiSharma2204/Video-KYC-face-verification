"""Video KYC - Streamlit UI.   Run with:  streamlit run app/streamlit_app.py

Walks through the whole flow: upload ID -> live webcam (face + blink liveness + quality)
-> face verification -> voice KYC (10 questions) -> outcome.
"""

from __future__ import annotations

import json
import threading
import time

import av
import cv2
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from videokyc.document import DocumentProcessor, TesseractOcr
from videokyc.errors import (
    KycError,
    LivenessFailedError,
    OcrUnavailableError,
    PasswordRequiredError,
)
from videokyc.face import FaceConfig, FaceMatchPipeline
from videokyc.face.config import DEFAULT_THRESHOLDS
from videokyc.liveness.blink import BlinkConfig
from videokyc.liveness.landmarks import MediaPipeEyeTracker
from videokyc.session.engine import KycConfig, KycEngine
from videokyc.session.models import Outcome, Stage
from videokyc.storage.memory import MemoryStore
from videokyc.voice.questions import Verdict

st.set_page_config(page_title="Video KYC", page_icon="🪪", layout="wide")

STAGES = [Stage.DOCUMENT, Stage.LIVENESS, Stage.VOICE, Stage.COMPLETED]
STAGE_LABELS = {Stage.DOCUMENT: "1 · ID upload", Stage.LIVENESS: "2 · Live check", Stage.VOICE: "3 · Voice KYC",
                Stage.COMPLETED: "4 · Result"}


# -- engine ---------------------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading models (first run downloads weights)...")
def get_engine(model: str, threshold: float, blinks: int, stt_model: str) -> KycEngine:
    import os

    face_cfg = FaceConfig(model_name=model, threshold=threshold)
    face = FaceMatchPipeline(face_cfg)
    stt = None
    if stt_model != "off":
        from videokyc.voice.stt import FasterWhisperStt

        stt = FasterWhisperStt(stt_model)
    if os.environ.get("MONGO_URI"):
        from videokyc.factory import build_store

        store = build_store()
    else:
        store = MemoryStore()
    return KycEngine(
        DocumentProcessor(TesseractOcr(), face),
        face,
        lambda: MediaPipeEyeTracker(face_cfg.model_dir),
        store,
        stt,
        KycConfig(blink=BlinkConfig(required_blinks=blinks)),
    )


# -- webcam processor -----------------------------------------------------------------------------


class LiveProcessor(VideoProcessorBase):
    """Runs the LiveCapture on the newest frame in a worker thread and draws feedback.

    Frame decoding must not block the WebRTC stream, so ``recv`` only stores the latest
    frame and paints the last known status; a worker does the heavy landmark/face work.
    """

    def __init__(self) -> None:
        self.capture = None
        self.status = None
        self._latest = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self) -> None:
        last = None
        while not self._stop.is_set():
            with self._lock:
                frame, cap = self._latest, self.capture
            if frame is None or cap is None or frame is last:
                time.sleep(0.01)
                continue
            last = frame
            try:
                self.status = cap.process(frame, time.monotonic())
            except Exception:  # noqa: BLE001 - never let a bad frame kill the stream
                self.status = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        with self._lock:
            self._latest = img.copy()
        status, cap = self.status, self.capture
        if status is not None and cap is not None:
            if status.box is not None:
                b = status.box
                color = (0, 200, 0) if status.quality_ok else (0, 165, 255)
                cv2.rectangle(img, (b.x, b.y), (b.x + b.w, b.y + b.h), color, 2)
            label = f"blinks {cap.blink.blinks}/{cap.blink.config.required_blinks}"
            cv2.putText(img, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            if status.issues:
                cv2.putText(img, status.issues[0], (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def on_ended(self) -> None:
        self._stop.set()


# -- helpers ---------------------------------------------------------------------------------------


def speak(text: str) -> None:
    """Read a prompt aloud in the browser (Web Speech API; no audio leaves the machine)."""
    safe = json.dumps(text)
    st.components.v1.html(
        f"""<button onclick="s()" style="padding:6px 12px;border-radius:6px;border:1px solid #888;cursor:pointer">
        🔊 Read question aloud</button>
        <script>function s(){{speechSynthesis.cancel();speechSynthesis.speak(new SpeechSynthesisUtterance({safe}));}}
        try{{s()}}catch(e){{}}</script>""",
        height=46,
    )


def stepper(stage: Stage) -> None:
    current = Stage.COMPLETED if stage is Stage.REJECTED else stage
    idx = STAGES.index(current) if current in STAGES else 0
    cols = st.columns(len(STAGES))
    for i, s in enumerate(STAGES):
        mark = "✅" if i < idx or (stage is Stage.COMPLETED and i == idx) else ("▶️" if i == idx else "⚪")
        if stage is Stage.REJECTED and i == idx:
            mark = "❌"
        cols[i].markdown(f"**{mark} {STAGE_LABELS[s]}**")


def show_image(img_bgr, caption: str) -> None:
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption=caption, width=180)


def reset() -> None:
    for k in ("sid", "capture_ready", "last_answer"):
        st.session_state.pop(k, None)


# -- sidebar ----------------------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Face model", list(DEFAULT_THRESHOLDS), 0)
    threshold = st.slider("Face-distance threshold", 0.05, 1.0, DEFAULT_THRESHOLDS[model], 0.01, key=f"thr-{model}",
                          help="Lower is stricter. Calibrate with `python -m videokyc.face.evaluate`.")
    blinks = st.slider("Blinks required", 1, 4, 2)
    stt_model = st.selectbox("Speech-to-text (offline Whisper)", ["base.en", "tiny.en", "small.en", "off"], 0,
                             help="'off' = type answers instead of speaking them.")
    st.caption("Images and audio are processed in memory. Only masked results are stored.")
    if st.button("Start a new session"):
        reset()
        st.rerun()

try:
    engine = get_engine(model, threshold, blinks, stt_model)
except OcrUnavailableError as exc:
    st.error(str(exc))
    st.stop()

# Changing a setting builds a new engine, which has never heard of the old session: start fresh.
settings_key = (model, threshold, blinks, stt_model)
if st.session_state.get("settings_key") != settings_key:
    reset()
    st.session_state.settings_key = settings_key
if "sid" not in st.session_state:
    st.session_state.sid = engine.create_session().id
session = engine.get(st.session_state.sid)

st.title("🪪 Video KYC")
stepper(session.stage)
st.divider()

# -- stage 1: document ---------------------------------------------------------------------------------

if session.stage is Stage.DOCUMENT:
    st.subheader("Upload your government ID")
    st.caption("Aadhaar, PAN, Passport, Voter ID or Driving Licence · PDF, JPG or PNG")
    file = st.file_uploader("ID document", type=["pdf", "jpg", "jpeg", "png"], label_visibility="collapsed")
    password = st.text_input("PDF password (if protected)", type="password",
                             help="e-Aadhaar: first 4 letters of your name in CAPITALS + birth year (YYYY).")
    if file and st.button("Read my ID", type="primary"):
        try:
            with st.spinner("Running OCR and reading your ID..."):
                engine.submit_document(session.id, file.getvalue(), password or None)
            st.rerun()
        except PasswordRequiredError:
            st.warning("This PDF is password protected - enter the password above.")
        except KycError as exc:
            st.error(str(exc))
            session = engine.get(session.id)
            if session.terminal:
                st.rerun()

# -- stage 2: live check ---------------------------------------------------------------------------------

elif session.stage is Stage.LIVENESS:
    left, right = st.columns([1, 2])
    with left:
        st.success(f"Recognised **{session.doc_type.replace('_', ' ').title()}** "
                   f"(confidence {session.doc_confidence:.0%}) · ID {session.id_masked or 'n/a'}")
        if session.duplicate_id_seen:
            st.warning("This ID was used in an earlier approved session.")
        for w in session.doc_warnings:
            st.caption(f"⚠️ {w}")
        if session.fields is not None and session.doc_photo is not None:
            show_image(session.doc_photo.crop, "Photo taken from your ID")
    with right:
        st.subheader("Look at the camera and blink naturally")
        st.caption(f"Keep your face well lit and centred. Blink {blinks}× - a still photo cannot pass this check.")
        ctx = webrtc_streamer(
            key=f"live-{session.id}", mode=WebRtcMode.SENDRECV, video_processor_factory=LiveProcessor,
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True,
        )
        if ctx.video_processor is not None and ctx.video_processor.capture is None:
            ctx.video_processor.capture = engine.new_capture(session.id)

        cap = ctx.video_processor.capture if ctx.video_processor else None
        if cap is not None:
            st.progress(min(cap.blink.blinks / blinks, 1.0), text=f"Blinks detected: {cap.blink.blinks}/{blinks}")
            st.caption(f"Sharp frames captured: {len(cap.best_frames())}/{cap.keep}")
            ready = cap.blink_passed and cap.best_frames()
            if st.button("Verify my face", type="primary", disabled=not ready):
                try:
                    with st.spinner("Matching your face against the ID photo..."):
                        engine.complete_liveness(session.id, cap)
                    st.rerun()
                except LivenessFailedError as exc:
                    st.error(f"{exc} ({exc.attempts_left} attempts left)")
                    cap.reset()
                except KycError as exc:
                    st.error(str(exc))
                    st.rerun()
            elif ctx.state.playing:
                time.sleep(0.5)
                st.rerun()  # keep the progress bar live while the stream runs

        with st.expander("No webcam? Upload a short video instead"):
            vid = st.file_uploader("Video (mp4/webm)", type=["mp4", "webm", "mov", "avi"], key="vid")
            if vid and st.button("Analyse video"):
                try:
                    with st.spinner("Analysing video..."):
                        engine.submit_video(session.id, vid.getvalue())
                    st.rerun()
                except LivenessFailedError as exc:
                    st.error(f"{exc} ({exc.attempts_left} attempts left)")
                except KycError as exc:
                    st.error(str(exc))

# -- stage 3: voice -------------------------------------------------------------------------------------

elif session.stage is Stage.VOICE:
    flow = session.voice_flow
    q = flow.current
    f = session.face
    st.info(f"✅ Face matched · distance {f['distance']:.3f} (threshold {f['threshold']:.2f})"
            + (" · flagged for manual review" if session.needs_review else ""))
    st.progress(flow.index / len(flow.questions), text=f"Question {flow.index + 1} of {len(flow.questions)}")
    st.subheader(q.prompt)
    speak(q.prompt)
    if flow.retry_pending and (last := st.session_state.get("last_answer")):
        st.warning(f"{last} Please try again ({flow.attempts_left} attempt left).")

    outcome = None
    if engine.stt is not None:
        audio = st.audio_input("Record your answer", key=f"aud-{flow.index}-{flow.attempts}")
        if audio is not None:
            with st.spinner("Transcribing (offline)..."):
                outcome = engine.submit_answer(session.id, audio=audio.getvalue())
    typed = st.text_input("…or type your answer", key=f"txt-{flow.index}-{flow.attempts}")
    if typed and st.button("Submit answer"):
        outcome = engine.submit_answer(session.id, transcript=typed)

    if outcome is not None:
        st.session_state.last_answer = (
            outcome.result.detail or ("Could not understand that." if outcome.result.verdict is Verdict.UNCLEAR else "")
        )
        st.rerun()

# -- stage 4: result -------------------------------------------------------------------------------------

else:
    if session.outcome is Outcome.APPROVED:
        st.success("## ✅ KYC completed")
    elif session.outcome is Outcome.MANUAL_REVIEW:
        st.warning("## ⚠️ KYC submitted - a reviewer will take a final look")
    else:
        st.error(f"## ❌ KYC rejected\n{session.rejection_reason}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Document", (session.doc_type or "-").replace("_", " ").title())
    c2.metric("Blinks", session.blinks if session.blinks is not None else "-")
    if session.face:
        c3.metric("Face distance", f"{session.face['distance']:.3f}", help=f"threshold {session.face['threshold']}")
    if session.voice:
        st.subheader("Voice answers")
        icons = {"pass": "✅", "fail": "❌", "recorded": "📝", "unclear": "❔"}
        for v in session.voice:
            st.write(f"{icons.get(v['verdict'], '')} **{v['question'].replace('_', ' ')}** - {v['verdict']}"
                     + (f" ({v['detail']})" if v["detail"] and v["verdict"] == "fail" else ""))
    with st.expander("Audit trail & stored record (contains no name, DOB, address or images)"):
        st.json(session.public_view())
    if st.button("Start over"):
        reset()
        st.rerun()

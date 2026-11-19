import json
import queue
import threading

import streamlit as st
import streamlit.components.v1 as components

def is_english_voice(voice) -> bool:
    description = f"{voice.id} {voice.name}".lower()
    if "english" in description:
        return True
    for language in getattr(voice, "languages", []):
        if isinstance(language, bytes):
            language = language.decode("utf-8", errors="ignore")
        language = str(language).lower()
        if language.startswith("en") or "en-us" in language or "en-gb" in language:
            return True
    return False

class VoiceAlertService:
    def __init__(self):
        self._messages = queue.Queue()
        self._ready = threading.Event()
        self._engine_lock = threading.Lock()
        self._engine = None
        self._error = None
        self._worker = threading.Thread(
            target=self._run,
            name="voice-alert-worker",
            daemon=True,
        )
        self._worker.start()
        if not self._ready.wait(timeout=15):
            raise RuntimeError("Timed out while initializing English voice alerts.")
        if self._error is not None:
            raise RuntimeError(f"English voice alerts are unavailable: {self._error}")

    def _run(self):
        try:
            import pyttsx3

            engine = pyttsx3.init()
            english_voice = next(
                (voice for voice in engine.getProperty("voices") if is_english_voice(voice)),
                None,
            )
            if english_voice is None:
                raise RuntimeError("No English text-to-speech voice is installed.")
            engine.setProperty("voice", english_voice.id)
            engine.setProperty("rate", 175)
        except (ImportError, OSError, RuntimeError) as error:
            self._error = error
            self._ready.set()
            return

        with self._engine_lock:
            self._engine = engine
        self._ready.set()
        while True:
            message = self._messages.get()
            try:
                engine.say(message)
                engine.runAndWait()
            except (OSError, RuntimeError) as error:
                self._error = error
                return
            finally:
                self._messages.task_done()

    def stop(self):
        self.clear_pending()
        with self._engine_lock:
            engine = self._engine
        if engine is not None:
            try:
                engine.stop()
            except (OSError, RuntimeError) as error:
                raise RuntimeError(f"Could not stop voice playback: {error}") from error

    def speak(self, messages):
        if self._error is not None:
            raise RuntimeError(f"English voice alerts are unavailable: {self._error}")
        for message in messages:
            self._messages.put(message)

    def clear_pending(self):
        while True:
            try:
                self._messages.get_nowait()
            except queue.Empty:
                return
            self._messages.task_done()

@st.cache_resource
def create_voice_alert_service() -> VoiceAlertService:
    return VoiceAlertService()

def get_voice_alert_service() -> VoiceAlertService:
    service = create_voice_alert_service()
    st.session_state["voice_alert_service"] = service
    return service

def activity_voice_messages(result: dict) -> list[str]:
    activity_phrases = {
        "Sleeping": "is sleeping",
        "Using Phone": "is using phone",
        "Talking": "is talking",
    }
    announcements = []
    for student in result["students"]:
        activity_phrase = activity_phrases.get(student["Activity"])
        if activity_phrase is None:
            continue
        name = student.get("Student Name", "Unknown Student")
        speaker = (
            "An unknown student"
            if not name or name == "Unknown Student"
            else name
        )
        announcements.append(f"{speaker} {activity_phrase}")
    return [", ".join(announcements)] if announcements else []

def speak_in_browser(messages, cancel=False):
    speech_counter = int(st.session_state.get("browser_speech_counter", 0)) + 1
    st.session_state["browser_speech_counter"] = speech_counter
    if cancel:
        script = f"""
        <!-- voice request {speech_counter} -->
        <script>
        if ("speechSynthesis" in window) {{
            window.speechSynthesis.cancel();
        }}
        </script>
        """
    else:
        payload = json.dumps(messages, ensure_ascii=True)
        payload = (
            payload.replace("<", "\\u003c")
            .replace(">", "\\u003e")
            .replace("&", "\\u0026")
        )
        script = f"""
        <!-- voice request {speech_counter} -->
        <script>
        if ("speechSynthesis" in window) {{
            for (const message of {payload}) {{
                const utterance = new SpeechSynthesisUtterance(message);
                utterance.lang = "en-US";
                window.speechSynthesis.speak(utterance);
            }}
        }}
        </script>
        """
    components.html(
        script,
        height=0,
        scrolling=False,
    )

def queue_voice_alerts(messages, browser=False):
    if not messages:
        return
    messages = list(messages)
    st.session_state["last_voice_messages"] = messages
    if not st.session_state.get("voice_on", True):
        return
    if browser:
        speak_in_browser(messages)
        return
    try:
        get_voice_alert_service().speak(messages)
    except RuntimeError as error:
        st.warning(f"Voice alert unavailable: {error}")


def render_voice_sidebar():
    if "voice_on" not in st.session_state:
        st.session_state["voice_on"] = st.session_state.get(
            "voice_alert_enabled",
            True,
        )
    st.checkbox("Voice Alert ON", key="voice_on")
    voice_test_column, voice_stop_column = st.columns(2)
    if voice_test_column.button("Test Voice", key="test_voice"):
        if st.session_state.get("voice_on", True):
            test_message = ["Voice Alert is working"]
            st.session_state["last_voice_messages"] = test_message
            speak_in_browser(test_message)
        else:
            st.info("Turn on Voice Alert to test sound.")
    if voice_stop_column.button(
        "Stop Voice",
        key="stop_voice",
    ):
        speak_in_browser([], cancel=True)
        voice_service = st.session_state.get("voice_alert_service")
        if voice_service is None:
            st.success("Voice playback stopped.")
        else:
            try:
                voice_service.stop()
            except RuntimeError as error:
                st.error(str(error))
            else:
                st.success("Voice playback stopped.")
    if st.button("SPEAK AGAIN", key="speak_again"):
        if not st.session_state.get("voice_on", True):
            st.info("Turn on Voice Alert to speak the last analysis.")
        elif st.session_state.get("last_analysis") is not None:
            messages = activity_voice_messages(st.session_state["last_analysis"])
            if messages:
                speak_in_browser(messages)
            else:
                st.info("There is no voice message in the last analysis.")
        else:
            st.info("There is no analysis to speak yet.")
    st.caption("Click Test Voice once to enable sound in browser.")

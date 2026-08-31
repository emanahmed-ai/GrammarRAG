"""
streamlit_app.py

FINAL STAGE of the RAG pipeline: Streamlit UI.

Purpose
-------
Thin orchestration/UI layer for the English Grammar and Syntax Error
Detection RAG Assistant ("Verity AI"). Connects user input to the
existing, already-approved pipeline and displays the structured result
inside a premium, production-quality visual shell.

    User Input
    -> 07_prompting.analyze_sentence()
    ->   06_retrieve_context.retrieve_context()  (called internally)
    ->   Persistent ChromaDB                      (called internally)
    ->   OpenRouter LLM                            (called internally)
    -> Structured Grammar Analysis
    -> Streamlit Display (this file)

This file deliberately does NOT:
    - embed queries itself
    - query ChromaDB itself
    - construct prompts itself
    - call OpenRouter itself
    - parse the LLM's JSON response itself

All of that already exists in 06_retrieve_context.py and 07_prompting.py
and is reused as-is via analyze_sentence(). This redesign only changes
presentation (layout, CSS, structure of the visual shell) - it does not
change any functional/data-flow behavior from the original file.

Run with:
    streamlit run streamlit_app.py
"""

import base64
import importlib
import os
import random
from typing import Any, Dict, List, Optional

import streamlit as st

import config

# 07_prompting.py starts with a digit, so it is loaded via importlib
# rather than a normal `import` statement. Python caches this in
# sys.modules, so across Streamlit reruns within the same server process
# this is a cheap lookup, not a fresh reload - the module-level model and
# ChromaDB-collection caches already set up inside 06_retrieve_context.py
# (Phase 8) stay warm. No additional st.cache_resource wrapper is added
# around model loading here, to avoid a second, conflicting caching
# strategy on top of the one that already exists.
_prompting_module = importlib.import_module("07_prompting")

# Curated pool that the Analyze page's three example chips are randomly
# drawn from (see _new_example_set() / _refresh_examples() below). Each
# entry pairs a sentence with the grammar category it illustrates, so the
# UI can show a category badge beside the sentence. Covers a broad spread
# of common English error categories, plus a few correct sentences, so
# the examples shown feel fresh and varied rather than a fixed demo set.
EXAMPLE_SENTENCE_POOL = [
    # Subject-Verb Agreement
    {"sentence": "He go to school every day.", "error_type": "Subject-Verb Agreement"},
    {"sentence": "I has a new book.", "error_type": "Subject-Verb Agreement"},
    {"sentence": "The students is studying hard.", "error_type": "Subject-Verb Agreement"},
    {"sentence": "They enjoys playing football.", "error_type": "Subject-Verb Agreement"},
    {"sentence": "There is many problems in this sentence.", "error_type": "Subject-Verb Agreement"},
    # Verb Tense
    {"sentence": "She have finished her homework.", "error_type": "Verb Tense"},
    {"sentence": "They was playing football yesterday.", "error_type": "Verb Tense"},
    {"sentence": "He have been working here for five years.", "error_type": "Verb Tense"},
    {"sentence": "I have seen this movie yesterday.", "error_type": "Verb Tense"},
    {"sentence": "He did not went to school.", "error_type": "Verb Tense"},
    # Articles
    {"sentence": "I am interested on machine learning.", "error_type": "Articles"},
    {"sentence": "He is married with a doctor.", "error_type": "Articles"},
    {"sentence": "She is good in mathematics.", "error_type": "Articles"},
    {"sentence": "I bought new book yesterday.", "error_type": "Articles"},
    # Prepositions
    {"sentence": "I look forward to meet you.", "error_type": "Prepositions"},
    {"sentence": "The information are useful.", "error_type": "Prepositions"},
    {"sentence": "She is good in English.", "error_type": "Prepositions"},
    # Word Order
    {"sentence": "He can sings very well.", "error_type": "Word Order"},
    {"sentence": "I am agree with you.", "error_type": "Word Order"},
    {"sentence": "Always I go to school early.", "error_type": "Word Order"},
    # Singular/Plural
    {"sentence": "There is three childs in the room.", "error_type": "Singular/Plural"},
    {"sentence": "She has two cat at home.", "error_type": "Singular/Plural"},
    # Missing Words
    {"sentence": "She don't like coffee.", "error_type": "Missing Words"},
    {"sentence": "He going to the market now.", "error_type": "Missing Words"},
    # Unnecessary Words
    {"sentence": "I can to swim very well.", "error_type": "Unnecessary Words"},
    {"sentence": "She said to me that she is happy.", "error_type": "Unnecessary Words"},
    # Spelling
    {"sentence": "Their going to the store later.", "error_type": "Spelling"},
    {"sentence": "I recieved your message yesterday.", "error_type": "Spelling"},
    # Punctuation
    {"sentence": "Although it was raining we went outside.", "error_type": "Punctuation"},
    {"sentence": "Its a beautiful day outside.", "error_type": "Punctuation"},
    # Correct Sentence (used as control examples)
    {"sentence": "She went to the store yesterday.", "error_type": "Correct Sentence"},
    {"sentence": "She has a beautiful voice.", "error_type": "Correct Sentence"},
    {"sentence": "They are going to the market.", "error_type": "Correct Sentence"},
]

# Local project asset used as the decorative application background (a
# portrait-oriented illustration of a golden tree rising from an open
# book). Configurable so the file can be relocated without touching any
# other code. Resolved relative to this file's own directory (not the
# process's current working directory) so the background still loads
# correctly regardless of where `streamlit run` is invoked from, both
# locally and on Streamlit Cloud.
BACKGROUND_IMAGE_PATH = "background_image.png"
# Official BCreative Academy logo artwork (background already made
# transparent) - rendered as-is in the sidebar, never redrawn/recreated.
LOGO_IMAGE_PATH = "bcreative_logo.png"

BRAND_NAME = "BCreative Academy"
BRAND_SUBTITLE = "RAG-POWERED GRAMMAR ASSISTANT"
BRAND_TAGLINE = "Intelligent English."
BRAND_TAGLINE_ACCENT = "Clearer Expression."
BRAND_SUBTAGLINE = "AI-Powered RAG Grammar Assistant"
AUTHORS = ["Amira Salama", "Eman Ahmed"]

NAV_ITEMS = [
    ("\u2302", "Home"),
    ("+", "Analyze"),
    ("\u23F7", "History"),
    ("\u25A3", "Grammar Rules"),
    ("\u25A4", "Examples"),
    ("\u2605", "Learn"),
    ("\u25C8", "Insights"),
    ("\u24D8", "About Us"),
]


# ---------------------------------------------------------------------------
# Existing RAG integration / configuration / secrets handling
# (unchanged in behavior from the original file)
# ---------------------------------------------------------------------------

def _sync_streamlit_secrets_to_environment() -> None:
    """
    On Streamlit Cloud, secrets are provided via `st.secrets`, not a local
    `.env` file. 07_prompting.py and config.py read `OPENROUTER_API_KEY`
    and `OPENROUTER_MODEL` only from environment variables by design (the
    key is never stored as a module/config attribute), so this copies
    whichever of those two secrets are present into `os.environ` - once,
    and only if not already set - before the existing pipeline functions
    are called. This changes nothing about how 07_prompting.py resolves
    its configuration; it only makes the Streamlit-secrets deployment
    case reach the same environment-variable path local `.env` usage
    already goes through.

    Never logs, displays, or stores the secret value anywhere else.
    """
    try:
        if hasattr(st, "secrets"):
            if not os.environ.get("OPENROUTER_API_KEY") and "OPENROUTER_API_KEY" in st.secrets:
                os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]
            if "OPENROUTER_MODEL" in st.secrets:
                os.environ["OPENROUTER_MODEL"] = st.secrets["OPENROUTER_MODEL"]
    except Exception:
        # No secrets.toml locally (e.g. running via `.env` instead) is
        # expected and not an error - fall through silently.
        pass


def _resolved_model() -> str:
    """
    Resolve the OpenRouter model to use for this run: the environment
    variable if set (covers both local .env and synced Streamlit
    secrets), otherwise config.py's default.

    Returns:
        The OpenRouter model identifier to pass to analyze_sentence().
    """
    return os.environ.get("OPENROUTER_MODEL", config.OPENROUTER_MODEL)


def _api_key_available() -> bool:
    """
    Check whether an OpenRouter API key is available without exposing it.

    Returns:
        True if OPENROUTER_API_KEY is set in the environment, False
        otherwise.
    """
    return bool(_prompting_module.load_environment())


def _friendly_error_message(exc: Exception) -> str:
    """
    Map a raised exception to a short, non-technical message for the
    normal user. The full exception text is shown separately in an
    expandable technical-details section, never inline in the main flow.

    Args:
        exc: The exception raised by analyze_sentence().

    Returns:
        A short, user-facing message string.
    """
    text = str(exc)

    if isinstance(exc, ValueError):
        return "Please enter a valid sentence to analyze."
    if "OPENROUTER_API_KEY" in text:
        return (
            "The grammar assistant is not fully configured yet: no "
            "OpenRouter API key is set. Please contact the app "
            "administrator."
        )
    if "ChromaDB" in text or "chroma_db" in text.lower():
        return (
            "The grammar knowledge base is not available right now. "
            "Please try again later or contact the app administrator."
        )
    if "timed out" in text.lower():
        return "The request took too long and timed out. Please try again."
    if "OpenRouter returned HTTP" in text:
        return "The grammar assistant's language model service returned an error. Please try again."
    return "Something went wrong while analyzing your sentence. Please try again."


def _set_example_sentence(example: str) -> None:
    """
    Callback for an example button: populate the sentence input with the
    example text.

    Set via `on_click`, not inline after the widget renders: Streamlit
    forbids assigning to a widget-bound session_state key once that
    widget has already been instantiated in the current script run.
    on_click callbacks run before the next run's widgets are
    instantiated, which avoids that restriction. Selecting an example
    only populates the input - it does not call analyze_sentence(), so
    no API call happens just from clicking an example.

    Args:
        example: The example sentence to populate the input with.
    """
    st.session_state.sentence_input = example


def _new_example_set(avoid: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """
    Pick 3 distinct example entries at random from EXAMPLE_SENTENCE_POOL,
    preferring sentences that are not in `avoid` (typically the sentences
    from the previously-shown set) so consecutive sets differ whenever
    the pool is large enough to allow it.

    Args:
        avoid: Sentence strings to exclude if possible (e.g. the
            sentences from the current set).

    Returns:
        A list of exactly 3 unique example dicts, each shaped like
        {"sentence": ..., "error_type": ...}.
    """
    avoid_set = set(avoid or [])
    candidates = [e for e in EXAMPLE_SENTENCE_POOL if e["sentence"] not in avoid_set]
    if len(candidates) < 3:
        # Pool too small (or fully excluded) to avoid repeats - fall back
        # to the full pool rather than erroring.
        candidates = EXAMPLE_SENTENCE_POOL
    return random.sample(candidates, 3)


def _refresh_examples() -> None:
    """
    Helper/callback: replace the currently displayed example set with a
    new random set, avoiding immediate repeats of the previous set where
    possible. Safe to call both from plain application logic (page-entry
    refresh) and as a widget `on_click` callback (the "New Examples"
    button) - it only ever writes to `current_examples`, a plain (non
    widget-bound) session_state key.
    """
    previous = st.session_state.get("current_examples", [])
    previous_sentences = [e["sentence"] for e in previous]
    st.session_state.current_examples = _new_example_set(avoid=previous_sentences)


@st.cache_data(show_spinner=False)
def _load_background_image_data_uri(path: str) -> Optional[str]:
    """
    Read the local background image and return it as a base64 data URI
    so it can be embedded directly in injected CSS (works identically
    whether the app runs locally or on Streamlit Cloud, where relative
    file-serving of static assets is not guaranteed).

    Cached with `st.cache_data` so the file is only read from disk and
    base64-encoded once per server process, not on every rerun.

    Args:
        path: Path to the background image, resolved relative to this
            file's own directory.

    Returns:
        A `data:image/png;base64,...` string, or None if the file is
        missing or unreadable - callers must fall back to a plain
        gradient background in that case rather than erroring.
    """
    try:
        resolved = path
        if not os.path.isabs(resolved):
            resolved = os.path.join(os.path.dirname(os.path.abspath(__file__)), resolved)
        with open(resolved, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        # Missing/unreadable background image is non-fatal - the app
        # must keep working with the existing gradient-only background.
        return None


# ---------------------------------------------------------------------------
# Theme / session-state initialization
# ---------------------------------------------------------------------------

def initialize_theme() -> None:
    """Ensure app session state keys exist before any widget touches
    them. The application has a single, permanent dark cinematic theme -
    there is no light mode and no theme toggle."""
    if "sentence_input" not in st.session_state:
        st.session_state.sentence_input = ""


def initialize_navigation() -> None:
    """Ensure page-routing and history-related session state keys exist
    before any widget or page-render function touches them. Purely
    additive UI state - does not affect the backend pipeline."""
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []
    # Tracks the identity of the last output already appended to history,
    # so a Streamlit rerun (e.g. from switching pages) never appends the
    # same analysis twice.
    if "last_history_id" not in st.session_state:
        st.session_state.last_history_id = None
    # The 3 example sentences currently shown on the Analyze page. Chosen
    # once here (session start) and then only regenerated when the user
    # (re)enters the Analyze page or explicitly clicks "New Examples" -
    # never on a plain rerun (e.g. typing) - see main()/_refresh_examples().
    if "current_examples" not in st.session_state:
        st.session_state.current_examples = _new_example_set()
    if "_last_page" not in st.session_state:
        st.session_state._last_page = None


def _go_to_page(page_name: str) -> None:
    """Callback: switch the active page. Used by sidebar nav buttons and
    any in-page call-to-action buttons (e.g. Home's 'Start Analyzing',
    Examples' 'Analyze This Example')."""
    st.session_state.current_page = page_name


def _go_to_analyze_with_sentence(sentence: str) -> None:
    """Callback: populate the analysis input with `sentence` and jump to
    the Analyze page in one action. Setting `sentence_input` here is safe
    for the same reason `_set_example_sentence` is safe: callbacks run
    before the next run's widgets are instantiated."""
    st.session_state.sentence_input = sentence
    st.session_state.current_page = "Analyze"


def _clear_analysis() -> None:
    """
    Callback for the Clear / "Analyze Another Sentence" buttons: reset
    the sentence input and the current analysis result/error state so
    the Analyze page returns to a clean slate.

    Set via `on_click`, for the same widget-key-mutation reason as
    `_set_example_sentence` above. Deliberately does NOT touch
    `analysis_history`, `last_history_id`, `current_examples`, or
    `current_page` - only the current-attempt state is cleared.
    """
    st.session_state.sentence_input = ""
    st.session_state.pop("last_output", None)
    st.session_state.pop("last_error", None)
    st.session_state.pop("last_error_technical", None)


# ---------------------------------------------------------------------------
# CSS injection - design system
# ---------------------------------------------------------------------------

def inject_custom_css() -> None:
    """Inject the Verity AI design system (CSS variables + component
    styles) for the current theme. Pure presentation - no functional
    behavior lives here."""

    # Background image (Phase 1): loaded once per process and reused as
    # a data URI so it works both locally and on Streamlit Cloud. `None`
    # here (missing/unreadable file) is handled below by simply omitting
    # the image layer - the existing gradient-only background is used
    # instead and the app keeps working normally.
    _bg_data_uri = _load_background_image_data_uri(BACKGROUND_IMAGE_PATH)

    # Single, permanent dark cinematic palette - the application has no
    # light mode and no theme switching of any kind.
    palette = """
    --navy: #0A1128;
    --deep-navy: #060B1C;
    --panel: rgba(10, 25, 60, 0.62);
    --panel-light: rgba(10, 25, 60, 0.80);
    --panel-border: rgba(255, 122, 0, 0.35);
    --ivory: #FFFFFF;
    --muted-ivory: #CBD5E1;
    --gold: #FF6600;
    --gold-light: #FF9A3D;
    --royal-blue: #164D9B;
    --purple: #2F6EDB;
    --success: #34d399;
    --danger: #f16060;
    """

    st.markdown(
        f"""
        <style>
        :root {{
            {palette}
            --radius-lg: 20px;
            --radius-md: 14px;
            --radius-sm: 10px;
            --shadow-soft: 0 1px 2px rgba(0,0,0,0.10), 0 10px 24px rgba(0,0,0,0.16);
            --shadow-lift: 0 2px 4px rgba(0,0,0,0.12), 0 16px 32px rgba(0,0,0,0.20);
            --transition: all 0.2s ease;
        }}

        html, body, [data-testid="stAppViewContainer"] {{
            {(
                f'''background-image:
                    linear-gradient(rgba(6,11,28,0.55), rgba(6,11,28,0.72)),
                    url("{_bg_data_uri}") !important;
                background-repeat: no-repeat, no-repeat !important;
                background-position: center, top center !important;
                background-size: cover, cover !important;
                background-color: var(--navy);
                background-attachment: scroll, fixed;'''
                if _bg_data_uri else
                f'''background-image:
                    radial-gradient(circle at 88% 12%, rgba(255,200,120,0.55), rgba(255,150,60,0.22) 18%, transparent 40%),
                    linear-gradient(200deg, #1a2a52 0%, #16264a 25%, #2a3f6b 45%, #6b5a4a 65%, #a67c52 80%, #d9a05b 100%) !important;
                background-repeat: no-repeat, no-repeat;
                background-position: center, center;
                background-size: cover, cover;
                background-color: var(--navy);'''
            )}
            background-color: var(--navy) !important;
            color: var(--ivory);
        }}

        /* On narrower viewports "contain" keeps the full portrait image
           visible without cropping, distortion or stretching - it just
           occupies proportionally less width, which is intentional. */
        @media (max-width: 640px) {{
            html, body, [data-testid="stAppViewContainer"] {{
                background-position: {"center, top center" if _bg_data_uri else "center, center"};
            }}
        }}

        [data-testid="stHeader"] {{
            background: transparent;
        }}

        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0A1128 0%, #0B3A82 55%, #0A1128 100%) !important;
            border-right: 1px solid rgba(255,255,255,0.08);
            min-width: 300px !important;
            max-width: 320px !important;
        }}
        section[data-testid="stSidebar"] > div {{
            padding-top: 1.2rem;
        }}

        .block-container {{
            padding-top: 1.2rem;
            padding-bottom: 3rem;
            max-width: 1320px;
        }}

        * {{
            font-family: 'Georgia', 'Iowan Old Style', 'Palatino Linotype', serif;
        }}

        /* ---------- Sidebar brand ---------- */
        .lex-sidebar-brand {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 6px;
            padding: 6px 4px 18px 4px;
            border-bottom: 1px solid var(--panel-border);
            margin-bottom: 14px;
        }}
        .lex-sidebar-brand .brand-logo-img {{
            display: block;
            width: 100%;
            max-width: 190px;
            height: auto;
            margin: 0 auto;
        }}
        .lex-sidebar-brand .brand-text .sub {{
            font-size: 0.6rem;
            letter-spacing: 1px;
            color: var(--muted-ivory);
            text-transform: uppercase;
            margin-top: 2px;
            text-align: center;
        }}

        .lex-nav-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 12px;
            border-radius: var(--radius-sm);
            margin-bottom: 4px;
            color: var(--muted-ivory);
            font-size: 0.92rem;
            border: 1px solid transparent;
            transition: var(--transition);
        }}
        .lex-nav-item:hover {{
            background: var(--panel-light);
            border-color: var(--panel-border);
            color: var(--ivory);
        }}
        .lex-nav-item.active {{
            background: linear-gradient(90deg, rgba(0,90,224,0.22), rgba(0,90,224,0.05));
            border-color: rgba(0,90,224,0.45);
            color: var(--ivory);
            box-shadow: inset 0 0 12px rgba(0,90,224,0.15);
        }}
        .lex-nav-icon {{
            width: 18px;
            text-align: center;
            color: var(--gold);
        }}

        /* Sidebar nav is now made of real Streamlit buttons (for actual
           click behavior) styled to look like the original nav rows. */
        section[data-testid="stSidebar"] .stButton > button {{
            display: flex;
            justify-content: flex-start;
            align-items: center;
            gap: 10px;
            width: 100%;
            padding: 10px 12px !important;
            border-radius: var(--radius-sm) !important;
            margin-bottom: 4px;
            background: transparent !important;
            border: 1px solid transparent !important;
            color: var(--muted-ivory) !important;
            font-size: 0.92rem !important;
            font-weight: 400 !important;
            text-align: left !important;
            box-shadow: none !important;
            transform: none !important;
        }}
        section[data-testid="stSidebar"] .stButton > button:hover {{
            background: var(--panel-light) !important;
            border-color: var(--panel-border) !important;
            color: var(--ivory) !important;
        }}
        section[data-testid="stSidebar"] .stButton > button:focus {{
            box-shadow: none !important;
        }}
        section[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
            background: rgba(47,110,219,0.30) !important;
            border: 1.5px solid var(--gold) !important;
            color: #FFFFFF !important;
            font-weight: 700 !important;
            box-shadow: inset 0 0 12px rgba(47,110,219,0.25) !important;
        }}
        /* "Analyze" nav item always reads as the solid-orange primary
           call-to-action in the sidebar, active or not - matching the
           reference design's filled CTA nav pill. */
        section[data-testid="stSidebar"] .st-key-nav_Analyze .stButton > button {{
            background: linear-gradient(90deg, var(--gold), var(--gold-light)) !important;
            border: none !important;
            color: #FFFFFF !important;
            font-weight: 700 !important;
            box-shadow: 0 4px 14px rgba(255,102,0,0.4) !important;
        }}
        section[data-testid="stSidebar"] .st-key-nav_Analyze .stButton > button:hover {{
            box-shadow: 0 6px 18px rgba(255,102,0,0.55) !important;
            transform: translateY(-1px);
        }}

        /* ---------- Empty states ---------- */
        .lex-empty-state {{
            text-align: center;
            padding: 48px 24px;
            color: var(--muted-ivory);
            border: 1px solid var(--panel-border);
            background: var(--panel);
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-soft);
            margin-bottom: 18px;
        }}
        .lex-empty-state .icon {{
            width: 56px;
            height: 56px;
            line-height: 56px;
            margin: 0 auto 14px auto;
            border-radius: 50%;
            background: rgba(255,102,0,0.12);
            border: 1px solid var(--panel-border);
            font-size: 1.6rem;
            color: var(--gold);
        }}
        .lex-empty-state .title {{
            color: var(--ivory);
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 6px;
        }}
        .lex-empty-state .desc {{
            font-size: 0.92rem;
            line-height: 1.55;
            max-width: 420px;
            margin: 0 auto;
        }}
        .lex-empty-state-cta {{
            display: flex;
            justify-content: center;
            margin-top: 16px;
        }}
        .lex-empty-state-cta .stButton > button {{
            padding: 0.6rem 1.4rem !important;
        }}

        /* ---------- Badges ---------- */
        .lex-badge {{
            display: inline-block;
            border-radius: 999px;
            padding: 3px 14px;
            font-size: 0.78rem;
            letter-spacing: 0.3px;
            font-weight: 600;
        }}
        .lex-badge.gold {{
            background: rgba(255,102,0,0.15);
            color: var(--gold-light);
            border: 1px solid rgba(255,102,0,0.35);
        }}
        .lex-badge.success {{
            background: rgba(52,211,153,0.14);
            color: var(--success);
            border: 1px solid rgba(52,211,153,0.35);
        }}
        .lex-badge.danger {{
            background: rgba(241,96,96,0.14);
            color: var(--danger);
            border: 1px solid rgba(241,96,96,0.35);
        }}

        /* ---------- Stat cards (Insights) ---------- */
        .lex-stat-card {{
            border: 1px solid var(--panel-border);
            background: var(--panel);
            border-radius: var(--radius-md);
            padding: 20px;
            text-align: center;
            transition: var(--transition);
        }}
        .lex-stat-card:hover {{
            border-color: rgba(255,102,0,0.4);
            box-shadow: var(--shadow-soft);
        }}
        .lex-stat-card .stat-value {{
            font-size: 2.1rem;
            font-weight: 700;
            color: var(--gold-light);
            line-height: 1.1;
        }}
        .lex-stat-card .stat-label {{
            color: var(--muted-ivory);
            font-size: 0.85rem;
            margin-top: 6px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        /* ---------- History rows ---------- */
        .lex-history-item {{
            border: 1px solid var(--panel-border);
            border-left: 3px solid var(--royal-blue);
            background: var(--panel);
            border-radius: var(--radius-sm);
            padding: 14px 16px;
            margin-bottom: 10px;
            transition: var(--transition);
        }}
        .lex-history-item:hover {{
            border-color: rgba(255,102,0,0.35);
            transform: translateX(2px);
        }}
        .lex-history-item.correct {{ border-left-color: var(--success); }}
        .lex-history-item.incorrect {{ border-left-color: var(--danger); }}
        .lex-history-item .hist-sentence {{
            color: var(--ivory);
            font-size: 0.96rem;
            margin-bottom: 4px;
        }}
        .lex-history-item .hist-meta {{
            color: var(--muted-ivory);
            font-size: 0.78rem;
        }}

        /* ---------- Page title ---------- */
        .lex-page-title {{
            text-align: center;
            padding: 6px 10px 18px 10px;
        }}
        .lex-page-title h1 {{
            font-size: 2.1rem;
            color: var(--ivory);
            margin-bottom: 4px;
        }}
        .lex-page-title .accent {{
            font-style: italic;
            background: linear-gradient(90deg, var(--gold-light), var(--gold));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .lex-page-title .subtitle {{
            color: var(--muted-ivory);
            font-size: 0.98rem;
        }}

        .lex-quote-card {{
            margin-top: 22px;
            padding: 18px 16px;
            border: 1px solid rgba(47,110,219,0.45);
            border-radius: var(--radius-md);
            background: rgba(22,77,155,0.35);
            backdrop-filter: blur(6px);
            position: relative;
        }}
        .lex-quote-card .mark {{
            font-size: 1.8rem;
            color: #FFFFFF;
            line-height: 0.6;
        }}
        .lex-quote-card .quote-text {{
            font-style: italic;
            color: #FFFFFF;
            font-size: 0.9rem;
            line-height: 1.5;
            margin: 6px 0 10px 0;
        }}
        .lex-quote-card .quote-author {{
            font-size: 0.78rem;
            color: var(--muted-ivory);
            text-align: right;
        }}

        /* ---------- Header ---------- */
        .lex-header-row {{
            display: flex;
            justify-content: flex-end;
            align-items: center;
            gap: 14px;
            padding-bottom: 6px;
        }}
        .lex-author-card {{
            border: 1px solid var(--panel-border);
            background: var(--panel);
            border-radius: var(--radius-sm);
            padding: 8px 16px;
            text-align: center;
            font-size: 0.75rem;
            color: var(--muted-ivory);
        }}
        .lex-author-card .crafted {{
            letter-spacing: 0.5px;
            font-size: 0.68rem;
            text-transform: uppercase;
            color: var(--gold);
            margin-bottom: 2px;
        }}
        .lex-author-card .names {{
            color: var(--ivory);
            font-size: 0.82rem;
        }}
        .lex-bell {{
            font-size: 1.1rem;
            color: #FFFFFF;
            border: 1.5px solid var(--gold);
            border-radius: 50%;
            width: 42px;
            height: 42px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(255,102,0,0.35);
            backdrop-filter: blur(6px);
        }}

        /* ---------- Hero ---------- */
        .lex-hero {{
            text-align: center;
            padding: 22px 10px 8px 10px;
        }}
        .lex-hero h1 {{
            font-size: 3.1rem;
            font-weight: 800;
            background: linear-gradient(90deg, var(--gold), var(--gold-light));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
            letter-spacing: 0.3px;
            text-shadow: 0 0 30px rgba(255,102,0,0.25);
        }}
        .lex-hero h1 .refined {{
            font-style: normal;
        }}
        .lex-hero .subtag {{
            color: #FFFFFF;
            font-size: 1.15rem;
            margin-bottom: 14px;
        }}
        .lex-hero .workflow {{
            color: var(--gold-light);
            font-size: 0.95rem;
            letter-spacing: 1px;
        }}
        .lex-hero .workflow .sep {{
            color: var(--gold);
            margin: 0 10px;
        }}

        /* ---------- Cards / panels ---------- */
        .lex-panel {{
            border: 1px solid var(--panel-border);
            background: var(--panel);
            border-radius: var(--radius-lg);
            padding: 26px 28px;
            box-shadow: var(--shadow-soft);
            backdrop-filter: blur(6px);
            margin-bottom: 22px;
        }}
        .lex-description-card {{
            display: flex;
            align-items: center;
            gap: 18px;
            max-width: 880px;
            margin-left: auto;
            margin-right: auto;
        }}
        .lex-description-card .desc-icon {{
            flex: 0 0 auto;
            width: 42px;
            height: 42px;
            border-radius: var(--radius-sm);
            background: linear-gradient(135deg, var(--royal-blue), var(--purple));
            color: #FFFFFF;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1rem;
        }}
        .lex-description-card .desc-text {{
            color: #F5F5F5;
            font-size: 1.02rem;
            line-height: 1.65;
        }}
        @media (max-width: 640px) {{
            .lex-description-card {{
                flex-direction: column;
                text-align: center;
            }}
        }}

        .lex-panel-header {{
            text-align: center;
            color: var(--gold-light);
            letter-spacing: 1px;
            font-size: 1.05rem;
            margin-bottom: 16px;
        }}

        /* Streamlit text_area override - a dedicated light writing
           surface for maximum readability. Deliberately uses fixed
           colors (not the --ivory var) so typed text is always dark
           navy on a light input surface, and never inherits the
           gold/ivory body text color. Multiple selectors target
           Streamlit's actual textarea DOM across versions. */
        textarea,
        [data-testid="stTextArea"] textarea,
        [data-baseweb="textarea"] textarea,
        .stTextArea textarea {{
            background: #F8FAFC !important;
            border: 1.5px solid rgba(11, 19, 36, 0.16) !important;
            border-radius: var(--radius-md) !important;
            color: #0A1128 !important;
            caret-color: #0A1128 !important;
            -webkit-text-fill-color: #0A1128 !important;
            font-size: 1.05rem !important;
            padding: 16px !important;
            transition: var(--transition);
        }}
        textarea::placeholder,
        [data-testid="stTextArea"] textarea::placeholder,
        [data-baseweb="textarea"] textarea::placeholder,
        .stTextArea textarea::placeholder {{
            color: #94A3B8 !important;
            opacity: 1 !important;
        }}
        textarea::selection,
        [data-testid="stTextArea"] textarea::selection,
        [data-baseweb="textarea"] textarea::selection {{
            background: rgba(0,90,224, 0.28) !important;
            color: #0A1128 !important;
        }}
        .stTextArea textarea:focus {{
            border-color: var(--gold) !important;
            box-shadow: 0 0 0 2px rgba(255,102,0,0.28), 0 0 18px rgba(0,90,224,0.2) !important;
            outline: none !important;
        }}
        .stTextArea label {{
            color: var(--muted-ivory) !important;
        }}

        .lex-example-badge-row {{
            text-align: center;
            margin-bottom: 6px;
        }}

        .lex-char-counter {{
            text-align: right;
            color: var(--muted-ivory);
            font-size: 0.78rem;
            margin-top: -6px;
            margin-bottom: 6px;
        }}

        /* Example chip buttons + primary CTA share Streamlit's button element */
        .stButton > button {{
            border-radius: var(--radius-sm) !important;
            border: 1px solid var(--panel-border) !important;
            background: var(--panel-light) !important;
            color: var(--muted-ivory) !important;
            transition: var(--transition) !important;
        }}
        .stButton > button:hover {{
            border-color: var(--gold) !important;
            color: var(--ivory) !important;
            transform: translateY(-1px);
        }}
        .stButton > button[kind="primary"] {{
            background: linear-gradient(90deg, var(--gold), var(--gold-light)) !important;
            border: none !important;
            color: #FFFFFF !important;
            font-weight: 700 !important;
            letter-spacing: 0.5px;
            padding: 0.7rem 1rem !important;
            box-shadow: 0 6px 20px rgba(255,102,0,0.35);
        }}
        .stButton > button[kind="primary"]:hover {{
            box-shadow: 0 8px 26px rgba(255,102,0,0.5);
            transform: translateY(-2px);
        }}

        /* ---------- Result banners ---------- */
        .lex-banner {{
            border-radius: var(--radius-md);
            padding: 14px 18px;
            font-weight: 600;
            margin-bottom: 18px;
            border: 1px solid;
        }}
        .lex-banner.error {{
            background: rgba(241,96,96,0.08);
            border-color: rgba(241,96,96,0.35);
            color: var(--danger);
        }}
        .lex-banner.success {{
            background: rgba(52,211,153,0.08);
            border-color: rgba(52,211,153,0.35);
            color: var(--success);
        }}

        /* ---------- Status badge (Phase 3) ---------- */
        .lex-status-row {{
            display: flex;
            justify-content: center;
            margin-bottom: 16px;
        }}
        .lex-status-badge {{
            display: inline-flex;
            align-items: center;
            gap: 9px;
            padding: 9px 22px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.95rem;
            letter-spacing: 1px;
            text-transform: uppercase;
            border: 1.5px solid;
        }}
        .lex-status-badge .dot {{
            font-size: 1rem;
            line-height: 1;
        }}
        .lex-status-badge.correct {{
            background: rgba(52,211,153,0.14);
            border-color: var(--success);
            color: var(--success);
        }}
        .lex-status-badge.incorrect {{
            background: rgba(241,96,96,0.14);
            border-color: var(--danger);
            color: var(--danger);
        }}

        .lex-detected-count {{
            text-align: center;
            color: var(--muted-ivory);
            font-size: 0.88rem;
            margin: 2px 0 18px 0;
        }}
        .lex-detected-count b {{
            color: var(--ivory);
            font-size: 0.98rem;
        }}

        .lex-final-panel {{
            border: 1px solid var(--panel-border);
            border-top: 3px solid var(--gold);
            background: var(--panel-light);
            border-radius: var(--radius-md);
            padding: 16px 18px;
            margin-top: 18px;
        }}
        .lex-final-panel .label {{
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--gold-light);
            margin-bottom: 6px;
        }}
        .lex-final-panel .content {{
            color: var(--ivory);
            font-size: 1.05rem;
            font-weight: 600;
        }}

        .lex-result-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 18px;
            margin-bottom: 16px;
        }}
        @media (max-width: 700px) {{
            .lex-result-grid {{ grid-template-columns: 1fr; }}
        }}
        .lex-result-box {{
            border-radius: var(--radius-sm);
            padding: 14px 16px;
            border: 1px solid var(--panel-border);
            background: var(--panel-light);
        }}
        .lex-result-box .label {{
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 6px;
        }}
        .lex-result-box.original .label {{ color: var(--danger); }}
        .lex-result-box.corrected .label {{ color: var(--success); }}
        .lex-result-box .content {{
            color: var(--ivory);
            font-size: 1.02rem;
        }}

        .lex-explanation {{
            border-left: 3px solid var(--purple);
            padding: 10px 16px;
            background: var(--panel-light);
            border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
            color: var(--muted-ivory);
            margin-bottom: 16px;
        }}
        .lex-explanation .label {{
            color: var(--purple);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 4px;
        }}

        .lex-error-card {{
            border: 1px solid var(--panel-border);
            border-left: 3px solid var(--gold);
            background: var(--panel);
            border-radius: var(--radius-sm);
            padding: 14px 16px;
            margin-bottom: 12px;
            transition: var(--transition);
        }}
        .lex-error-card:hover {{
            border-left-color: var(--danger);
            transform: translateX(2px);
        }}
        .lex-error-card .etype {{
            display: inline-block;
            background: rgba(255,102,0,0.15);
            color: var(--gold-light);
            border-radius: 999px;
            padding: 2px 12px;
            font-size: 0.75rem;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}
        .lex-error-card .row {{
            font-size: 0.92rem;
            color: var(--muted-ivory);
            margin-bottom: 3px;
        }}
        .lex-error-card .row b {{ color: var(--ivory); }}

        /* ---------- Feature cards ---------- */
        .lex-feature-card {{
            border: 1.5px solid var(--panel-border);
            background: var(--panel);
            border-radius: var(--radius-md);
            padding: 18px;
            height: 100%;
            transition: var(--transition);
        }}
        .lex-feature-card:hover {{
            transform: translateY(-3px);
            border-color: var(--gold);
            box-shadow: var(--shadow-lift);
        }}
        .lex-feature-card .ficon {{
            width: 38px;
            height: 38px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 10px;
            border: 1.5px solid var(--gold);
            font-size: 1.1rem;
            margin-bottom: 10px;
        }}
        .lex-feature-card .ftitle {{
            font-weight: 700;
            color: var(--gold-light);
            margin-bottom: 6px;
        }}
        .lex-feature-card .fdesc {{
            font-size: 0.85rem;
            color: #F5F5F5;
            line-height: 1.4;
        }}
        .lex-feature-card.f1, .lex-feature-card.f2,
        .lex-feature-card.f3, .lex-feature-card.f4 {{ border-top: 1.5px solid var(--gold); }}

        /* ---------- Footer ---------- */
        .lex-footer {{
            text-align: center;
            padding-top: 24px;
            margin-top: 12px;
            border-top: 1px solid var(--panel-border);
            color: var(--muted-ivory);
            font-size: 0.82rem;
        }}
        .lex-footer .crafted {{
            color: var(--gold-light);
            margin-top: 4px;
        }}

        /* Expanders */
        .streamlit-expanderHeader, [data-testid="stExpander"] summary {{
            color: var(--gold-light) !important;
        }}
        [data-testid="stExpander"] {{
            border: 1px solid var(--panel-border) !important;
            border-radius: var(--radius-md) !important;
            background: var(--panel) !important;
        }}

        div[data-testid="stAlert"] {{
            border-radius: var(--radius-md) !important;
        }}

        /* ---------- Responsive design ---------- */
        html, body {{
            overflow-x: hidden;
        }}
        [data-testid="stAppViewContainer"] {{
            overflow-x: hidden;
        }}

        /* Tablet and below */
        @media (max-width: 900px) {{
            .block-container {{
                padding-left: 1rem;
                padding-right: 1rem;
                max-width: 100%;
            }}
            .lex-hero h1 {{
                font-size: 2.2rem;
            }}
            .lex-panel {{
                padding: 20px 18px;
            }}
        }}

        /* Phones */
        @media (max-width: 640px) {{
            .lex-hero h1 {{
                font-size: 1.7rem;
            }}
            .lex-hero .subtag {{
                font-size: 0.92rem;
            }}
            .lex-hero .workflow {{
                font-size: 0.8rem;
            }}
            .lex-hero .workflow .sep {{
                margin: 0 5px;
            }}
            .lex-panel {{
                padding: 16px 14px;
                border-radius: var(--radius-md);
            }}
            .lex-page-title h1 {{
                font-size: 1.5rem;
            }}
            .lex-header-row {{
                flex-wrap: wrap;
                justify-content: center;
                gap: 8px;
            }}
            .lex-author-card {{
                font-size: 0.7rem;
                padding: 6px 12px;
            }}
            /* Stack Streamlit's own column layout vertically wherever it
               is used for cards/buttons on narrow screens, so nothing is
               squeezed or clipped. */
            div[data-testid="stHorizontalBlock"] {{
                flex-wrap: wrap !important;
            }}
            div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {{
                min-width: 100% !important;
                flex: 1 1 100% !important;
            }}
            .stButton > button {{
                width: 100% !important;
                white-space: normal !important;
            }}
            section[data-testid="stSidebar"] .stButton > button {{
                min-height: 44px;
                font-size: 0.95rem !important;
                padding: 10px 14px !important;
            }}
            .lex-sidebar-brand .brand-logo-img {{
                max-width: 160px;
            }}
            .lex-result-box .content {{
                font-size: 0.95rem;
                word-break: break-word;
            }}
            .lex-error-card .row {{
                font-size: 0.88rem;
                word-break: break-word;
            }}
            .lex-stat-card .stat-value {{
                font-size: 1.7rem;
            }}
        }}

        /* Prevent long unbroken text (sentences, corrections) from
           forcing horizontal scroll on any screen size. */
        .lex-result-box .content,
        .lex-error-card .row,
        .hist-sentence,
        .lex-explanation,
        .lex-final-panel .content {{
            overflow-wrap: break-word;
            word-wrap: break-word;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# UI helper / render functions
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    """Elegant premium sidebar: brand identity, visual navigation, quote."""
    with st.sidebar:
        _logo_data_uri = _load_background_image_data_uri(LOGO_IMAGE_PATH)
        _logo_img_html = (
            f'<img class="brand-logo-img" src="{_logo_data_uri}" alt="{BRAND_NAME}">'
            if _logo_data_uri
            # Fallback so the sidebar still renders sensibly if the logo
            # asset file is ever missing - never a recreated/redrawn logo.
            else f'<div class="brand-text"><div class="name">{BRAND_NAME}</div></div>'
        )
        st.markdown(
            f"""
            <div class="lex-sidebar-brand">
                {_logo_img_html}
                <div class="brand-text">
                    <div class="sub">{BRAND_SUBTITLE}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        current_page = st.session_state.current_page
        for icon, label in NAV_ITEMS:
            is_active = label == current_page
            st.button(
                f"{icon}  {label}",
                key=f"nav_{label}",
                on_click=_go_to_page,
                args=(label,),
                use_container_width=True,
                type="primary" if is_active else "secondary",
            )

        st.markdown(
            """
            <div class="lex-quote-card">
                <div class="mark">&#8220;</div>
                <div class="quote-text">The limits of my language mean the
                limits of my world.</div>
                <div class="quote-author">&mdash; Ludwig Wittgenstein</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_header() -> None:
    """Top-right header control: the notification bell, matching the
    reference design's minimal glass-pill control. There is no theme
    toggle - the application has a single, permanent dark theme."""
    spacer, bell_col = st.columns([9, 0.6])
    with bell_col:
        st.markdown('<div class="lex-bell">&#128276;</div>', unsafe_allow_html=True)


def render_hero() -> None:
    """Luxury editorial hero / landing header for the app."""
    st.markdown(
        f"""
        <div class="lex-hero">
            <h1>{BRAND_TAGLINE} {BRAND_TAGLINE_ACCENT}</h1>
            <div class="subtag">{BRAND_SUBTAGLINE}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not _api_key_available():
        st.warning(
            "No OpenRouter API key is currently configured, so analysis "
            "requests will fail until one is set (OPENROUTER_API_KEY "
            "environment variable, or Streamlit secrets when deployed)."
        )


def render_analysis_input() -> bool:
    """
    Render the premium analysis input card: text area, example chips and
    the primary CTA. Returns True if the Analyze button was clicked this
    run (mirrors the original file's `analyze_clicked` flag).
    """
    st.markdown('<div class="lex-panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="lex-panel-header">&#10140; Enter an English sentence '
        'to analyze &#10141;</div>',
        unsafe_allow_html=True,
    )

    st.text_area(
        "Sentence",
        key="sentence_input",
        height=110,
        placeholder="Type or paste your sentence here...",
        label_visibility="collapsed",
    )

    char_count = len(st.session_state.sentence_input)
    st.markdown(
        f'<div class="lex-char-counter">{char_count}/500</div>',
        unsafe_allow_html=True,
    )

    label_col, refresh_col = st.columns([5, 1.6])
    with label_col:
        st.markdown(
            '<div style="color: var(--gold-light); font-size:0.85rem; '
            'margin-bottom:8px;">Or try an example:</div>',
            unsafe_allow_html=True,
        )
    with refresh_col:
        st.button(
            "\u21BB New Examples",
            key="refresh_examples_btn",
            on_click=_refresh_examples,
            use_container_width=True,
        )

    current_examples = st.session_state.current_examples
    example_cols = st.columns(len(current_examples))
    for i, (col, example) in enumerate(zip(example_cols, current_examples)):
        sentence = example["sentence"]
        error_type = example["error_type"]
        badge_cls = "success" if error_type == "Correct Sentence" else "gold"
        with col:
            st.markdown(
                f'<div class="lex-example-badge-row">'
                f'<span class="lex-badge {badge_cls}">{error_type}</span></div>',
                unsafe_allow_html=True,
            )
            st.button(
                f"\u2712\uFE0F {sentence}",
                use_container_width=True,
                key=f"example_{i}_{sentence}",
                on_click=_set_example_sentence,
                args=(sentence,),
            )

    st.write("")
    analyze_col, clear_col = st.columns([3, 1])
    with analyze_col:
        analyze_clicked = st.button(
            "+ Analyze Sentence", type="primary", use_container_width=True
        )
    with clear_col:
        st.button(
            "\u2716 Clear",
            use_container_width=True,
            key="clear_analysis_btn",
            on_click=_clear_analysis,
        )

    st.markdown("</div>", unsafe_allow_html=True)
    return analyze_clicked


def render_analysis_result(result: Dict[str, Any]) -> None:
    """
    Render the structured grammar-analysis result inside the premium
    result panel (see 07_prompting.parse_model_response() for its shape).
    Preserves the exact fields/keys used by the original implementation.

    Args:
        result: The "result" dict from analyze_sentence()'s output.
    """
    st.markdown('<div class="lex-panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="lex-panel-header">&#10140; Analysis Result &#10141;</div>',
        unsafe_allow_html=True,
    )

    if "parse_error" in result:
        st.warning(
            "The assistant's response could not be understood as "
            "structured data, so no analysis can be shown for this "
            "attempt. Please try again."
        )
        with st.expander("Technical details"):
            st.write(result["parse_error"])
            st.code(result.get("raw_response", ""), language="text")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    has_error = result.get("has_error", False)
    original_sentence = result.get("original_sentence", "")
    corrected_sentence = result.get("corrected_sentence", "")
    overall_explanation = result.get("overall_explanation", "")
    errors = result.get("errors", [])

    if not has_error:
        st.markdown(
            '<div class="lex-status-row">'
            '<div class="lex-status-badge correct">'
            '<span class="dot">&#10003;</span> Correct</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="lex-banner success">&#10003; Your sentence is '
            'grammatically correct.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="lex-result-box corrected">
                <div class="label">Original Sentence</div>
                <div class="content">{original_sentence}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if overall_explanation:
            st.markdown(
                f"""
                <div class="lex-explanation" style="margin-top:14px;">
                    <div class="label">Explanation</div>
                    <div>{overall_explanation}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    error_count = len(errors)
    error_word = "error" if error_count == 1 else "errors"

    st.markdown(
        '<div class="lex-status-row">'
        '<div class="lex-status-badge incorrect">'
        '<span class="dot">!</span> Needs Correction</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="lex-banner error">&#9888; This sentence contains '
        f'{error_count} grammar or syntax {error_word}.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="lex-result-grid">
            <div class="lex-result-box original">
                <div class="label">Original Sentence</div>
                <div class="content">{original_sentence}</div>
            </div>
            <div class="lex-result-box corrected">
                <div class="label">Corrected Sentence</div>
                <div class="content">{corrected_sentence}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="lex-detected-count">Detected Errors: '
        f'<b>{error_count}</b></div>',
        unsafe_allow_html=True,
    )

    if overall_explanation:
        st.markdown(
            f"""
            <div class="lex-explanation">
                <div class="label">Overall Explanation</div>
                <div>{overall_explanation}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if errors:
        st.markdown(
            '<div class="lex-panel-header" style="text-align:left; '
            'font-size:0.95rem; margin: 10px 0 10px 0;">Error Details</div>',
            unsafe_allow_html=True,
        )
        for index, error in enumerate(errors, start=1):
            etype = error.get("type", "unknown")
            eoriginal = error.get("original", "")
            ecorrection = error.get("correction", "")
            eexplanation = error.get("explanation", "")
            st.markdown(
                f"""
                <div class="lex-error-card">
                    <div class="etype">Error {index} &middot; {etype}</div>
                    <div class="row"><b>Original:</b> {eoriginal}</div>
                    <div class="row"><b>Correction:</b> {ecorrection}</div>
                    <div class="row"><b>Why this matters:</b> {eexplanation}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        f"""
        <div class="lex-final-panel">
            <div class="label">Final Corrected Text</div>
            <div class="content">{corrected_sentence}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


def render_retrieved_examples(examples: List[Dict[str, Any]]) -> None:
    """
    Render the retrieved reference examples in a refined expandable
    section, to demonstrate that retrieval actually informed the
    analysis. Preserves the original fields: distance, incorrect,
    correct.

    Args:
        examples: Retrieved example dicts from retrieve_context() (see
            06_retrieve_context.py's format_results()).
    """
    with st.expander(f"\U0001F4DA Retrieved Grammar Examples ({len(examples)})"):
        if not examples:
            st.write("No similar examples were retrieved.")
            return

        for rank, example in enumerate(examples, start=1):
            distance = example.get("distance")
            distance_str = f"{distance:.4f}" if isinstance(distance, (int, float)) else "n/a"
            st.markdown(
                f"**Rank {rank}** &nbsp;&middot;&nbsp; similarity distance: `{distance_str}`"
            )
            st.write(f"- Incorrect: {example.get('incorrect', '')}")
            st.write(f"- Correct: {example.get('correct', '')}")
            st.divider()


def render_page_title(title: str, accent: str = "", subtitle: str = "") -> None:
    """Small reusable page-header for the non-Home pages: plain title,
    optional italic gold `accent` word/phrase, optional subtitle line."""
    accent_html = f' <span class="accent">{accent}</span>' if accent else ""
    subtitle_html = f'<div class="subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="lex-page-title">
            <h1>{title}{accent_html}</h1>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_cards() -> None:
    """Four premium feature cards summarizing the product's value
    proposition. Purely presentational."""
    features = [
        ("\U0001F9E0", "AI-Powered Analysis",
         "Advanced AI models detect and analyze grammar errors with high accuracy.", "f1"),
        ("\U0001FAB6", "Smart Correction",
         "Get accurate corrections with clear explanations for better understanding.", "f2"),
        ("\U0001F393", "Learn & Improve",
         "Learn from your mistakes and improve your English grammar step by step.", "f3"),
        ("\U0001F4C4", "Detailed Insights",
         "Detailed explanations and error types to help you master grammar rules.", "f4"),
    ]
    cols = st.columns(4)
    for col, (icon, title, desc, cls) in zip(cols, features):
        with col:
            st.markdown(
                f"""
                <div class="lex-feature-card {cls}">
                    <div class="ficon">{icon}</div>
                    <div class="ftitle">{title}</div>
                    <div class="fdesc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_footer() -> None:
    """Premium footer with copyright line, matching the reference design."""
    st.markdown(
        """
        <div class="lex-footer">
            &copy; 2025 BCreative AI. rights reserved.
            <div class="crafted">Crafted with Intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_home_page() -> None:
    """The Home page: hero, short product explanation, CTA into Analyze,
    and the four feature cards. Purely presentational - no backend call
    happens on this page."""
    render_hero()

    st.markdown(
        """
        <div class="lex-panel lex-description-card">
            <div class="desc-icon">&#128269;</div>
            <div class="desc-text">
                Verify AI pairs semantic retrieval of real grammar correction
                examples with an AI language model to detect, correct, and
                explain English grammar and syntax errors&mdash;so every
                correction comes with a clear, learnable reason behind it.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cta_col = st.columns([1, 1, 1])[1]
    with cta_col:
        st.button(
            "+ Analyze Sentence",
            type="primary",
            use_container_width=True,
            on_click=_go_to_page,
            args=("Analyze",),
            key="home_start_analyzing",
        )

    st.write("")
    render_feature_cards()
    render_footer()


def _append_to_history(output: Dict[str, Any], sentence: str) -> None:
    """
    Append a successful analysis to `analysis_history` for the History
    and Insights pages, guarding against accidental duplicate entries if
    Streamlit reruns the same script pass more than once for a single
    click (e.g. due to a page switch immediately after analyzing).

    Args:
        output: The full dict returned by analyze_sentence().
        sentence: The sentence that was analyzed (already stripped).
    """
    result = output.get("result", {})
    record = {
        "original_sentence": result.get("original_sentence", sentence),
        "corrected_sentence": result.get("corrected_sentence", ""),
        "has_error": result.get("has_error", False),
        "errors": result.get("errors", []),
        "overall_explanation": result.get("overall_explanation", ""),
    }
    record_id = (record["original_sentence"], record["corrected_sentence"], record["has_error"])
    if st.session_state.get("last_history_id") == record_id:
        return
    st.session_state.analysis_history.append(record)
    st.session_state.last_history_id = record_id


def render_analyze_page() -> None:
    """
    The Analyze page: sentence input, example chips, Analyze CTA, and the
    result of the existing RAG/grammar-analysis pipeline. This is the
    original file's single-page workflow, relocated but functionally
    identical - the backend call itself is untouched.
    """
    render_page_title(
        "Analyze a Sentence",
        subtitle="Paste any English sentence and let Verity detect, correct, and explain the issue.",
    )

    if not _api_key_available():
        st.warning(
            "No OpenRouter API key is currently configured, so analysis "
            "requests will fail until one is set (OPENROUTER_API_KEY "
            "environment variable, or Streamlit secrets when deployed)."
        )

    analyze_clicked = render_analysis_input()

    if analyze_clicked:
        sentence = st.session_state.sentence_input.strip()

        if not sentence:
            st.session_state.pop("last_output", None)
            st.session_state["last_error"] = "Please enter a sentence to analyze."
        else:
            with st.spinner("\u2726 Verity AI is reviewing your grammar..."):
                try:
                    output = _prompting_module.analyze_sentence(
                        sentence,
                        model=_resolved_model(),
                    )
                    st.session_state["last_output"] = output
                    st.session_state.pop("last_error", None)
                    _append_to_history(output, sentence)
                except (ValueError, RuntimeError) as exc:
                    st.session_state.pop("last_output", None)
                    st.session_state["last_error"] = _friendly_error_message(exc)
                    st.session_state["last_error_technical"] = str(exc)

    if st.session_state.get("last_error"):
        st.error(st.session_state["last_error"])
        technical = st.session_state.get("last_error_technical")
        if technical:
            with st.expander("Technical details"):
                st.write(technical)

    if st.session_state.get("last_output"):
        output = st.session_state["last_output"]
        render_analysis_result(output["result"])
        render_retrieved_examples(output.get("retrieved_examples", []))

        again_col = st.columns([1, 1.4, 1])[1]
        with again_col:
            st.button(
                "\u21BA Analyze Another Sentence",
                type="primary",
                use_container_width=True,
                key="analyze_another_btn",
                on_click=_clear_analysis,
            )

    render_footer()


def render_history_page() -> None:
    """The History page: every past successful analysis from this
    session, most recent first. Reads only from `analysis_history` -
    never calls the backend."""
    render_page_title("Analysis", accent="History")

    history = st.session_state.get("analysis_history", [])

    if not history:
        st.markdown(
            """
            <div class="lex-empty-state">
                <div class="icon">&#8981;</div>
                <div class="title">No analyses yet</div>
                <div class="desc">Your grammar analysis history will appear
                here after you analyze your first sentence.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="lex-empty-state-cta">', unsafe_allow_html=True)
        cta_col = st.columns([1, 1.3, 1])[1]
        with cta_col:
            st.button(
                "\u2726 Analyze a Sentence",
                type="primary",
                use_container_width=True,
                on_click=_go_to_page,
                args=("Analyze",),
                key="history_empty_cta",
            )
        st.markdown("</div>", unsafe_allow_html=True)
        render_footer()
        return

    for i, record in enumerate(reversed(history)):
        status_cls = "incorrect" if record["has_error"] else "correct"
        status_label = "Needs correction" if record["has_error"] else "Correct"
        st.markdown(
            f"""
            <div class="lex-history-item {status_cls}">
                <div class="hist-sentence">{record['original_sentence']}</div>
                <div class="hist-meta">{status_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if record["has_error"]:
            with st.expander(f"View details \u2014 entry {len(history) - i}"):
                st.markdown(f"**Corrected:** {record['corrected_sentence']}")
                if record["overall_explanation"]:
                    st.markdown(f"**Explanation:** {record['overall_explanation']}")
                for err in record["errors"]:
                    st.markdown(f"- *{err.get('type', 'unknown')}*: {err.get('explanation', '')}")

    render_footer()


GRAMMAR_RULES = [
    ("Subject-Verb Agreement",
     "The verb must agree in number with its subject: singular subjects take singular verbs, plural subjects take plural verbs.",
     "He go to school every day.", "He goes to school every day."),
    ("Verb Tenses",
     "The verb form must match the intended time frame of the action (past, present, future).",
     "She go yesterday.", "She went yesterday."),
    ("Articles",
     "Use 'a'/'an' for a non-specific singular noun and 'the' for a specific one; singular countable nouns usually need an article.",
     "I bought new book.", "I bought a new book."),
    ("Prepositions",
     "Certain adjectives and verbs pair with specific prepositions by convention, not by direct translation.",
     "She is good in English.", "She is good at English."),
    ("Word Order",
     "Standard English order is Subject - Verb - Object, with adverbs of frequency placed before the main verb.",
     "Always I go to school early.", "I always go to school early."),
    ("Singular and Plural",
     "Nouns must match their determiners and verbs in number; irregular plurals don't take a trailing 's'.",
     "There is three childs in the room.", "There are three children in the room."),
    ("Punctuation",
     "Commas separate clauses and items; sentences end with correct terminal punctuation.",
     "Although it was raining we went outside", "Although it was raining, we went outside."),
    ("Spelling",
     "Commonly confused or misspelled words change meaning or readability even when pronunciation is similar.",
     "Their going to the store.", "They're going to the store."),
]


def render_grammar_rules_page() -> None:
    """Educational reference page: expandable cards for common English
    grammar rules, each with a short explanation and an example pair."""
    render_page_title("Grammar", accent="Rules")

    for title, explanation, incorrect, correct in GRAMMAR_RULES:
        with st.expander(title):
            st.markdown(explanation)
            st.markdown(
                f"""
                <div class="lex-result-grid">
                    <div class="lex-result-box original">
                        <div class="label">Incorrect</div>
                        <div class="content">{incorrect}</div>
                    </div>
                    <div class="lex-result-box corrected">
                        <div class="label">Correct</div>
                        <div class="content">{correct}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    render_footer()


EXAMPLES_LIBRARY = [
    ("Subject-Verb Agreement", "He go to school every day.", "He goes to school every day.",
     "The singular subject 'He' requires the singular verb form 'goes'."),
    ("Verb Tense", "She go yesterday.", "She went yesterday.",
     "'Yesterday' signals the past tense, so the verb must be 'went'."),
    ("Articles", "I bought new book.", "I bought a new book.",
     "Singular countable nouns like 'book' need an article such as 'a'."),
    ("Prepositions", "She is good in English.", "She is good at English.",
     "The adjective 'good' conventionally pairs with the preposition 'at'."),
    ("Word Order", "Always I go to school early.", "I always go to school early.",
     "Adverbs of frequency like 'always' go before the main verb, after the subject."),
]


def render_examples_page() -> None:
    """Curated example sentences by error category. Each card can send
    the incorrect sentence straight into the Analyze page."""
    render_page_title("Grammar", accent="Examples")

    for i, (etype, incorrect, correct, explanation) in enumerate(EXAMPLES_LIBRARY):
        st.markdown('<div class="lex-panel">', unsafe_allow_html=True)
        st.markdown(
            f'<span class="lex-badge gold">{etype}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="lex-result-grid" style="margin-top:12px;">
                <div class="lex-result-box original">
                    <div class="label">Incorrect</div>
                    <div class="content">{incorrect}</div>
                </div>
                <div class="lex-result-box corrected">
                    <div class="label">Correct</div>
                    <div class="content">{correct}</div>
                </div>
            </div>
            <div class="lex-explanation" style="margin-top:12px;">
                <div class="label">Why</div>
                <div>{explanation}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "\u2726 Analyze This Example",
            key=f"analyze_example_{i}",
            on_click=_go_to_analyze_with_sentence,
            args=(incorrect,),
        )
        st.markdown("</div>", unsafe_allow_html=True)

    render_footer()


def render_learn_page() -> None:
    """Short-form educational content about how grammar analysis works
    and how to improve. Purely static/informational."""
    render_page_title("Learn", accent="English")

    sections = [
        ("How grammar errors work",
         "Grammar errors happen when a sentence breaks an expected structural "
         "pattern - like verb agreement, tense, or word order - even if the "
         "meaning is still understandable. Verity looks for these patterns "
         "using examples retrieved from a curated grammar knowledge base."),
        ("How subject-verb agreement works",
         "The verb changes form depending on whether the subject is singular "
         "or plural, and depending on grammatical person (I, you, he/she/it, "
         "we, they). Mismatches, like 'He go' instead of 'He goes', are one "
         "of the most common English errors for learners."),
        ("Common English mistakes",
         "Frequent trouble spots include missing articles ('a'/'an'/'the'), "
         "incorrect prepositions, inconsistent verb tense within a sentence, "
         "and irregular plural or past-tense forms that don't follow the "
         "usual '-s' or '-ed' pattern."),
        ("How to improve sentence structure",
         "Reading sentences aloud, keeping sentences shorter while learning, "
         "and reviewing the 'why' behind each correction - not just the fix "
         "itself - helps the correct pattern stick for next time."),
    ]

    for title, body in sections:
        with st.expander(title):
            st.markdown(body)

    render_footer()


def render_insights_page() -> None:
    """Simple aggregate statistics computed from `analysis_history` -
    no external dependencies, no backend calls."""
    render_page_title("Your", accent="Insights")

    history = st.session_state.get("analysis_history", [])

    if not history:
        st.markdown(
            """
            <div class="lex-empty-state">
                <div class="icon">&#9733;</div>
                <div class="title">Insights will appear as you analyze more sentences</div>
                <div class="desc">Once you build analysis history, Verity AI
                will show useful patterns such as your most common error
                types, your correct-versus-incorrect ratio, and how your
                grammar changes over time.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="lex-empty-state-cta">', unsafe_allow_html=True)
        cta_col = st.columns([1, 1.3, 1])[1]
        with cta_col:
            st.button(
                "\u2726 Analyze a Sentence",
                type="primary",
                use_container_width=True,
                on_click=_go_to_page,
                args=("Analyze",),
                key="insights_empty_cta",
            )
        st.markdown("</div>", unsafe_allow_html=True)
        render_footer()
        return

    total = len(history)
    incorrect_count = sum(1 for r in history if r["has_error"])
    correct_count = total - incorrect_count

    error_type_counts: Dict[str, int] = {}
    for record in history:
        for err in record["errors"]:
            etype = err.get("type", "unknown")
            error_type_counts[etype] = error_type_counts.get(etype, 0) + 1

    stat_cols = st.columns(3)
    stats = [
        (str(total), "Total Analyses"),
        (str(correct_count), "Correct Sentences"),
        (str(incorrect_count), "Sentences With Errors"),
    ]
    for col, (value, label) in zip(stat_cols, stats):
        with col:
            st.markdown(
                f"""
                <div class="lex-stat-card">
                    <div class="stat-value">{value}</div>
                    <div class="stat-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    if error_type_counts:
        st.markdown(
            '<div class="lex-panel-header" style="margin-top:10px;">'
            '&#10140; Most Common Error Types &#10141;</div>',
            unsafe_allow_html=True,
        )
        ranked = sorted(error_type_counts.items(), key=lambda kv: kv[1], reverse=True)
        for etype, count in ranked:
            st.markdown(
                f'<span class="lex-badge gold" style="margin:4px 6px 4px 0;">'
                f'{etype} &middot; {count}</span>',
                unsafe_allow_html=True,
            )

    render_footer()


def render_about_page() -> None:
    """Static About Us page: project description and author credit."""
    render_page_title(BRAND_NAME, subtitle=BRAND_SUBTITLE)

    st.markdown(
        """
        <div class="lex-panel">
            <div style="color: var(--muted-ivory); font-size:1.02rem; line-height:1.7;">
            Verity AI combines semantic retrieval of real grammar-correction
            examples with AI-powered analysis to help learners detect,
            understand, and correct English grammar and syntax errors.
            Instead of a generic correction, every result is grounded in
            similar examples pulled from a curated grammar knowledge base,
            then explained in plain language so the underlying rule is easy
            to learn - not just fix.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="lex-panel" style="text-align:center;">
            <div class="lex-author-card" style="display:inline-block;">
                <div class="crafted">Crafted with Intelligence</div>
                <div class="names">{"<br>".join(AUTHORS)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_footer()


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:
    """Build and run the Streamlit UI."""
    st.set_page_config(
        page_title="Verity AI - Intelligent English Assistant",
        page_icon="\u2712\uFE0F",
        layout="wide",
    )

    _sync_streamlit_secrets_to_environment()
    initialize_theme()
    initialize_navigation()
    inject_custom_css()

    render_sidebar()
    render_header()

    page_renderers = {
        "Home": render_home_page,
        "Analyze": render_analyze_page,
        "History": render_history_page,
        "Grammar Rules": render_grammar_rules_page,
        "Examples": render_examples_page,
        "Learn": render_learn_page,
        "Insights": render_insights_page,
        "About Us": render_about_page,
    }
    current_page = st.session_state.current_page

    # A fresh example set is drawn once per visit to the Analyze page -
    # i.e. only on the rerun where current_page actually transitions into
    # "Analyze" (first load, or navigating back to it) - not on every
    # rerun caused by typing in the textarea or clicking Analyze while
    # already on the page.
    if current_page == "Analyze" and st.session_state._last_page != "Analyze":
        _refresh_examples()
    st.session_state._last_page = current_page

    render_page = page_renderers.get(current_page, render_home_page)
    render_page()


if __name__ == "__main__":
    main()

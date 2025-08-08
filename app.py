import json
import base64
from datetime import datetime
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(
    page_title="Notes → Structured JSON (Azure OpenAI Vision)",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Notes → Structured JSON Suite")
st.caption("Multi-tab app. Tab 1 does Notes → JSON via Azure OpenAI (Vision). Add your other flows in the next tabs.")

# ---------------------------
# Secrets / Config (st.secrets)
# ---------------------------
def get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets[name]
    except Exception:
        return default

AZURE_API_KEY     = get_secret("AZURE_API_KEY")
AZURE_ENDPOINT    = get_secret("AZURE_ENDPOINT")
AZURE_DEPLOYMENT  = get_secret("AZURE_DEPLOYMENT", "gpt-4")
AZURE_API_VERSION = get_secret("AZURE_API_VERSION", "2024-08-01-preview")

if not (AZURE_API_KEY and AZURE_ENDPOINT and AZURE_DEPLOYMENT and AZURE_API_VERSION):
    st.warning(
        "⚙️ Configure your Azure secrets in `.streamlit/secrets.toml` → "
        "AZURE_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT, AZURE_API_VERSION"
    )

# ---------------------------
# Prompts & helpers
# ---------------------------
SYSTEM_PROMPT = """
You are a teaching assistant. The student has uploaded a notes image.
1. Extract a catchy title → storytitle
2. Summarise into 5 slides (s2paragraph1 to s6paragraph1), each ≤ 400 characters
3. For each (including title), generate vivid, multi-color vector-style DALL·E image prompts (s1alt1 to s6alt1)
Respond in this exact JSON format (no extra text):

{
  "storytitle": "...",
  "s2paragraph1": "...",
  "s3paragraph1": "...",
  "s4paragraph1": "...",
  "s5paragraph1": "...",
  "s6paragraph1": "...",
  "s1alt1": "...",
  "s2alt1": "...",
  "s3alt1": "...",
  "s4alt1": "...",
  "s5alt1": "...",
  "s6alt1": "..."
}
"""

def make_vision_payload(b64_image: str):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                ]
            }
        ],
        "temperature": 0.7,
        "max_tokens": 1200
    }

def call_azure_chat(payload: dict) -> requests.Response:
    url = f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"
    headers = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}
    return requests.post(url, headers=headers, json=payload, timeout=60)

def try_parse_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        pass
    import re
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None

def generate_seo_metadata(storytitle: str, slides: list):
    seo_prompt = f"""
Generate SEO metadata for a web story:

Title: {storytitle}
Slides:
- {slides[0] if len(slides) > 0 else ""}
- {slides[1] if len(slides) > 1 else ""}
- {slides[2] if len(slides) > 2 else ""}
- {slides[3] if len(slides) > 3 else ""}
- {slides[4] if len(slides) > 4 else ""}

Respond in JSON:
{{"metadescription": "...", "metakeywords": "..."}}
"""
    payload = {
        "messages": [
            {"role": "system", "content": "You are an expert SEO assistant. Reply with JSON only."},
            {"role": "user", "content": seo_prompt}
        ],
        "temperature": 0.4,
        "max_tokens": 300
    }
    resp = call_azure_chat(payload)
    if resp.status_code == 200:
        raw = resp.json()["choices"][0]["message"]["content"]
        parsed = try_parse_json(raw)
        if isinstance(parsed, dict):
            return parsed.get("metadescription", ""), parsed.get("metakeywords", "")
    return "Explore this insightful story.", "web story, learning"

# =========================================================
# Tabs
# =========================================================
tab1, tab2, tab3 = st.tabs([
    "Tab 1 • Notes → JSON",
    "Tab 2 • (your next feature)",
    "Tab 3 • (your next feature)"
])

# ---------------------------
# TAB 1: Notes → JSON
# ---------------------------
with tab1:
    st.subheader("1) Upload your notes image")
    img_file = st.file_uploader("JPG/PNG only", type=["jpg", "jpeg", "png"], key="notes_img_uploader")

    st.subheader("2) Optional fields")
    coachingname = st.text_input("Coaching name (optional)", value="", key="coaching_name_tab1")
    add_seo = st.checkbox("Add SEO metadata", value=True, key="add_seo_tab1")

    run = st.button("Generate JSON", type="primary", disabled=not img_file or not AZURE_API_KEY, key="run_tab1")

    if run and img_file:
        try:
            # Read & preview
            image = Image.open(img_file).convert("RGB")
            st.image(image, caption="Uploaded", use_column_width=True)

            # Base64 encode
            buf = BytesIO()
            image.save(buf, format="JPEG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            # Vision call
            with st.spinner("Calling Azure OpenAI (Vision)…"):
                resp = call_azure_chat(make_vision_payload(b64))

            if resp.status_code != 200:
                st.error(f"Azure error {resp.status_code}: {resp.text[:400]}")
            else:
                content = resp.json()["choices"][0]["message"]["content"]
                parsed = try_parse_json(content)
                if not isinstance(parsed, dict):
                    st.error("Model did not return valid JSON. Check your deployment/model settings.")
                else:
                    # Enrich JSON
                    if coachingname.strip():
                        parsed["coachingname"] = coachingname.strip()

                    if add_seo:
                        slides = [
                            parsed.get("s2paragraph1", ""),
                            parsed.get("s3paragraph1", ""),
                            parsed.get("s4paragraph1", ""),
                            parsed.get("s5paragraph1", ""),
                            parsed.get("s6paragraph1", ""),
                        ]
                        with st.spinner("Generating SEO metadata…"):
                            meta_desc, meta_keywords = generate_seo_metadata(parsed.get("storytitle", ""), slides)
                        parsed["metadescription"] = meta_desc
                        parsed["metakeywords"] = meta_keywords

                    # Save to session for other tabs
                    st.session_state["notes_json"] = parsed

                    # Show + download
                    st.subheader("✅ Structured JSON")
                    st.json(parsed)

                    fname_slug = parsed.get("storytitle", "webstory").replace(" ", "_").replace(":", "").lower()
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    dl_name = f"{fname_slug}_{ts}.json"
                    st.download_button(
                        label="⬇️ Download JSON",
                        data=json.dumps(parsed, ensure_ascii=False, indent=2).encode("utf-8"),
                        file_name=dl_name,
                        mime="application/json",
                        key="download_json_tab1"
                    )

        except Exception as e:
            st.exception(e)

    # Helper display if JSON already exists from a previous run
    if "notes_json" in st.session_state and not run:
        with st.expander("Last generated JSON (from session)", expanded=False):
            st.json(st.session_state["notes_json"])

# ---------------------------
# TAB 2: Placeholder
# ---------------------------
with tab2:
    st.info("Add your next module here. You can read Tab 1 output via `st.session_state['notes_json']`.")
    if "notes_json" in st.session_state:
        st.write("Detected JSON from Tab 1:")
        st.json(st.session_state["notes_json"])

# ---------------------------
# TAB 3: Placeholder
# ---------------------------
with tab3:
    st.info("Another placeholder tab for future features.")
    if "notes_json" in st.session_state:
        st.write("You can reuse the same JSON here too:")
        st.json(st.session_state["notes_json"])

st.markdown("---")
st.caption("Tip: Add your Azure keys to `.streamlit/secrets.toml` when running locally.")

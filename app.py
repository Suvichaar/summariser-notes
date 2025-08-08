import json
import base64
from datetime import datetime
from io import BytesIO
from typing import Optional

import requests
import streamlit as st
from PIL import Image

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(
    page_title="Notes → AMP Web Story",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Notes → AMP Web Story")
st.caption("Tab 1: Notes → JSON (Azure OpenAI Vision). Tab 2: optional DALL·E images. Tab 3: your next feature.")

# ---------------------------
# Secrets / Config (st.secrets)
# ---------------------------
def get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets[name]
    except Exception:
        return default

# Chat/Vision
AZURE_API_KEY     = get_secret("AZURE_API_KEY")
AZURE_ENDPOINT    = get_secret("AZURE_ENDPOINT")  # e.g., https://<resource>.openai.azure.com
AZURE_DEPLOYMENT  = get_secret("AZURE_DEPLOYMENT", "gpt-4")
AZURE_API_VERSION = get_secret("AZURE_API_VERSION", "2024-08-01-preview")

# DALL·E 3
# If these are blank, we auto-fallback to the Chat/Vision endpoint/key (same resource).
# If you paste a FULL Images URL (including /images/generations and ?api-version=...), we'll use it as-is.
AZURE_DALLE_KEY         = get_secret("AZURE_DALLE_KEY", "")
AZURE_DALLE_ENDPOINT    = get_secret("AZURE_DALLE_ENDPOINT", "")  # can be a full URL or just the host
AZURE_DALLE_DEPLOYMENT  = get_secret("AZURE_DALLE_DEPLOYMENT", "dall-e-3")
AZURE_DALLE_API_VERSION = get_secret("AZURE_DALLE_API_VERSION", "2024-02-01")

if not (AZURE_API_KEY and AZURE_ENDPOINT and AZURE_DEPLOYMENT and AZURE_API_VERSION):
    st.warning(
        "⚙️ Configure your Azure Chat/Vision secrets in `.streamlit/secrets.toml` → "
        "AZURE_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT, AZURE_API_VERSION"
    )

# ---------------------------
# Prompts & helpers (Chat/Vision)
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
    url = f"{AZURE_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions"
    params = {"api-version": AZURE_API_VERSION}
    headers = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}
    return requests.post(url, headers=headers, params=params, json=payload, timeout=60)

def try_parse_json(text: str):
    # strict
    try:
        return json.loads(text)
    except Exception:
        pass
    # strip code fences
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if "\n" in t:
            t = t.split("\n", 1)[1]
    try:
        return json.loads(t)
    except Exception:
        pass
    # first {...}
    import re
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
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

# ---------------------------
# DALL·E helpers (Images)
# ---------------------------
def _effective_dalle_key() -> str:
    # Reuse Chat key if Images key not provided
    return AZURE_DALLE_KEY or AZURE_API_KEY or ""

def _effective_dalle_endpoint() -> str:
    # Use Images endpoint if provided; else fallback to Chat endpoint
    return (AZURE_DALLE_ENDPOINT or AZURE_ENDPOINT or "").rstrip("/")

def _is_full_images_url(url: str) -> bool:
    """True if the provided endpoint already includes /images/generations (possibly with ?api-version=...)."""
    if not url:
        return False
    u = url.lower()
    return "images/generations" in u

def _images_config_ok() -> bool:
    return bool(_effective_dalle_endpoint() and _effective_dalle_key())

def call_azure_dalle(prompt: str, size: str = "1024x1024") -> Optional[bytes]:
    """
    Azure OpenAI Images API (DALL·E 3) -> raw image bytes.

    Supports TWO modes for AZURE_DALLE_ENDPOINT:
      1) FULL URL pasted (contains '/images/generations' and maybe '?api-version=...') → use as-is.
      2) HOST only (e.g., https://<resource>.openai.azure.com) → we append the standard path and api-version.
    """
    endpoint = _effective_dalle_endpoint()
    key = _effective_dalle_key()

    if not _images_config_ok():
        st.error("Missing DALL·E config. Provide endpoint and key (or rely on Chat/Vision fallbacks).")
        return None

    headers = {"Content-Type": "application/json", "api-key": key}
    payload = {"prompt": prompt, "n": 1, "size": size}

    if _is_full_images_url(endpoint):
        # Use EXACT endpoint as given (no extra path/params)
        url = endpoint  # already includes /images/generations and maybe ?api-version=...
        r = requests.post(url, headers=headers, json=payload, timeout=90)
    else:
        # Build standard Images URL
        url = f"{endpoint}/openai/deployments/{AZURE_DALLE_DEPLOYMENT}/images/generations"
        params = {"api-version": AZURE_DALLE_API_VERSION}
        r = requests.post(url, headers=headers, params=params, json=payload, timeout=90)

    if r.status_code != 200:
        st.error(
            f"DALL·E error {r.status_code} from {url}\n"
            f"(If you pasted a full URL, we are using it verbatim.)\n"
            f"body: {r.text[:500]}"
        )
        return None

    try:
        img_url = r.json()["data"][0]["url"]
    except Exception:
        st.error(f"Unexpected Images response: {r.text[:500]}")
        return None

    try:
        return requests.get(img_url, timeout=60).content
    except Exception as e:
        st.error(f"Failed to download generated image: {e}")
        return None

# =========================================================
# Tabs
# =========================================================
tab1, tab2, tab3 = st.tabs([
    "Tab 1 • Notes → JSON",
    "Tab 2 • DALL·E images (optional)",
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

    if "notes_json" in st.session_state and not run:
        with st.expander("Last generated JSON (from session)", expanded=False):
            st.json(st.session_state["notes_json"])

# ---------------------------
# TAB 2: DALL·E images (optional)
# ---------------------------
with tab2:
    st.info("Generate images from prompts s1alt1..s6alt1 in the JSON from Tab 1. No S3 uploads — just preview and download.")
    # Debug: show what URL we'll call
    with st.expander("Images API config (debug)"):
        eff_ep = _effective_dalle_endpoint()
        st.write({
            "endpoint_provided": AZURE_DALLE_ENDPOINT,
            "endpoint_used": eff_ep,
            "treated_as_full_url": _is_full_images_url(eff_ep),
            "deployment": AZURE_DALLE_DEPLOYMENT,
            "api_version": AZURE_DALLE_API_VERSION,
            "key_present": bool(_effective_dalle_key()),
        })

    if "notes_json" not in st.session_state:
        st.warning("Run Tab 1 first to produce JSON with image prompts.")
    else:
        j = st.session_state["notes_json"]
        size = st.selectbox("Image size", ["1024x1024", "512x512", "256x256"], index=0)
        ready = _images_config_ok()
        go = st.button("Generate Images", disabled=not ready, type="primary")

        if go:
            images = []
            with st.spinner("Generating images with Azure DALL·E 3…"):
                for i in range(1, 7):
                    prompt = j.get(f"s{i}alt1", "").strip()
                    if not prompt:
                        images.append(None)
                        continue
                    img_bytes = call_azure_dalle(prompt, size=size)
                    images.append(img_bytes)

            st.divider()
            for i, img_bytes in enumerate(images, start=1):
                st.subheader(f"Slide {i}")
                if img_bytes:
                    img = Image.open(BytesIO(img_bytes)).convert("RGB")
                    st.image(img, use_column_width=True)
                    st.download_button(
                        "Download image",
                        data=img_bytes,
                        file_name=f"slide{i}.jpg",
                        mime="image/jpeg",
                        key=f"dl_img_{i}"
                    )
                else:
                    st.warning("No image for this slide.")

# ---------------------------
# TAB 3: Placeholder
# ---------------------------
with tab3:
    st.info("Placeholder tab for your next feature. You can reuse `st.session_state['notes_json']` here as well.")
    if "notes_json" in st.session_state:
        st.json(st.session_state["notes_json"])

st.markdown("---")
st.caption("Tip: never commit real keys. Use `.streamlit/secrets.toml` or Streamlit Cloud secrets.")

# app.py
# ------------------------------------------------------------
# Tab 1: Notes Image → GPT JSON → Safe DALL·E Images → S3 → JSON (CDN resized URLs)
# Tab 2: JSON → Azure Speech TTS MP3 → S3 → add audio fields to JSON
# ------------------------------------------------------------
import os
import io
import re
import json
import time
import base64
import requests
import boto3
import mimetypes
from io import BytesIO
from datetime import datetime
from PIL import Image
import streamlit as st

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Suvichaar Tools",
    page_icon="🧠",
    layout="centered"
)
st.title("🧠 Suvichaar Tools")
st.caption("Tab 1: Notes → JSON + Images | Tab 2: JSON → TTS MP3")

# ---------------------------
# Secrets / Config
# ---------------------------
def get_secret(key, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return default

# Azure OpenAI (vision)
AZURE_API_KEY     = get_secret("AZURE_API_KEY")
AZURE_ENDPOINT    = get_secret("AZURE_ENDPOINT")
AZURE_DEPLOYMENT  = get_secret("AZURE_DEPLOYMENT", "gpt-4o")  # vision-capable model
AZURE_API_VERSION = get_secret("AZURE_API_VERSION", "2024-08-01-preview")

# Azure DALL·E
DALE_ENDPOINT     = get_secret("DALE_ENDPOINT")  # full Azure DALL·E images/generations endpoint
DAALE_KEY         = get_secret("DAALE_KEY")

# AWS S3
AWS_ACCESS_KEY    = get_secret("AWS_ACCESS_KEY")
AWS_SECRET_KEY    = get_secret("AWS_SECRET_KEY")
AWS_REGION        = get_secret("AWS_REGION", "ap-south-1")
AWS_BUCKET        = get_secret("AWS_BUCKET")
S3_PREFIX         = get_secret("S3_PREFIX", "media")

# CDN image handler prefix (base64-encoded template)
CDN_PREFIX_MEDIA  = get_secret("CDN_PREFIX_MEDIA", "https://media.suvichaar.org/")

# Fallback image
DEFAULT_ERROR_IMAGE = get_secret("DEFAULT_ERROR_IMAGE", "https://media.suvichaar.org/default-error.jpg")

# Azure Speech (TTS)
AZURE_SPEECH_KEY     = get_secret("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION  = get_secret("AZURE_SPEECH_REGION", "eastus")
VOICE_NAME           = get_secret("VOICE_NAME", "hi-IN-AaravNeural")

# CDN for audio files (served via CloudFront)
CDN_BASE             = get_secret("CDN_BASE", "https://cdn.suvichaar.org/")

# Sanity checks
missing_core = []
for k in ["AZURE_API_KEY", "AZURE_ENDPOINT", "AZURE_DEPLOYMENT", "DALE_ENDPOINT", "DAALE_KEY",
          "AWS_ACCESS_KEY", "AWS_SECRET_KEY", "AWS_BUCKET"]:
    if not get_secret(k):
        missing_core.append(k)
if missing_core:
    st.warning("Add these secrets in `.streamlit/secrets.toml`: " + ", ".join(missing_core))

# ---------------------------
# Shared helpers (Tab 1)
# ---------------------------
def build_resized_cdn_url(bucket: str, key_path: str, width: int, height: int) -> str:
    """Return base64-encoded template URL for your Serverless Image Handler."""
    template = {
        "bucket": bucket,
        "key": key_path,
        "edits": {
            "resize": {
                "width": width,
                "height": height,
                "fit": "cover"
            }
        }
    }
    encoded = base64.urlsafe_b64encode(json.dumps(template).encode()).decode()
    return f"{CDN_PREFIX_MEDIA}{encoded}"

SAFE_FALLBACK = (
    "A joyful, abstract geometric illustration symbolizing unity and learning — "
    "soft shapes, harmonious gradients, friendly silhouettes, "
    "no text, no logos, no brands, no real persons, family-friendly, "
    "flat vector style, bright colors."
)

def sanitize_prompt(chat_url: str, headers: dict, original_prompt: str) -> str:
    """Rewrite any risky prompt into a safe, positive, family-friendly version using Azure Chat."""
    sanitize_payload = {
        "messages": [
            {"role": "system", "content": (
                "Rewrite image prompts to be safe, positive, inclusive, and family-friendly. "
                "Remove any hate/harassment/violence/adult/illegal/extremist content, slogans, logos, "
                "or real-person likenesses. Keep the core educational idea and flat vector art style. "
                "Return ONLY the rewritten prompt text."
            )},
            {"role": "user", "content": f"Original prompt:\n{original_prompt}\n\nRewritten safe prompt:"}
        ],
        "temperature": 0.2,
        "max_tokens": 300
    }
    try:
        sr = requests.post(chat_url, headers=headers, json=sanitize_payload, timeout=60)
        if sr.status_code == 200:
            return sr.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        st.info(f"Sanitizer call failed; using local guards: {e}")

    return (original_prompt +
            "\nFlat vector illustration, bright colors, no text, no logos, no brands, "
            "no real persons, family-friendly, inclusive, peaceful.")

def generate_and_upload_images(result_json: dict) -> dict:
    """Generate DALL·E images, upload originals to S3, return CDN resized URLs in JSON."""
    if not all([DALE_ENDPOINT, DAALE_KEY, AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_BUCKET]):
        st.error("Missing DALL·E and/or AWS S3 secrets.")
        return {**result_json}

    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION
    )

    slug = (
        result_json["storytitle"]
        .lower()
        .replace(" ", "-")
        .replace(":", "")
        .replace(".", "")
    )
    out = {k: result_json[k] for k in result_json}
    first_slide_key = None

    headers_dalle = {"Content-Type": "application/json", "api-key": DAALE_KEY}
    progress = st.progress(0, text="Generating images…")

    for i in range(1, 7):
        raw_prompt = result_json.get(f"s{i}alt1", "") or ""
        chat_headers = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}
        chat_url = f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"
        safe_prompt = sanitize_prompt(chat_url, chat_headers, raw_prompt)

        payload = {"prompt": safe_prompt, "n": 1, "size": "1024x1024"}
        image_url = None

        for attempt in range(3):
            r = requests.post(DALE_ENDPOINT, headers=headers_dalle, json=payload, timeout=120)
            if r.status_code == 200:
                try:
                    image_url = r.json()["data"][0]["url"]
                    break
                except Exception as e:
                    st.info(f"Slide {i}: unexpected DALL·E response format: {e}")
                    break
            elif r.status_code in (400, 403):
                st.info(f"Slide {i}: DALL·E blocked, retrying with fallback.")
                payload = {"prompt": SAFE_FALLBACK, "n": 1, "size": "1024x1024"}
                continue
            elif r.status_code == 429:
                st.info(f"Slide {i}: rate-limited, waiting 10s…")
                time.sleep(10)
            else:
                st.info(f"Slide {i}: DALL·E error {r.status_code} — {r.text[:200]}")
                break

        if image_url:
            try:
                img_data = requests.get(image_url, timeout=120).content
                buffer = BytesIO(img_data)  # upload original; no local resize
                key = f"{S3_PREFIX.rstrip('/')}/{slug}/slide{i}.jpg"
                s3.upload_fileobj(buffer, AWS_BUCKET, key, ExtraArgs={"ContentType": "image/jpeg"})
                if i == 1:
                    first_slide_key = key

                # build CDN resized URL (720x1200)
                final_url = build_resized_cdn_url(AWS_BUCKET, key, 720, 1200)
                out[f"s{i}image1"] = final_url
            except Exception as e:
                st.info(f"Slide {i}: upload/CDN URL build failed → {e}")
                out[f"s{i}image1"] = DEFAULT_ERROR_IMAGE
        else:
            out[f"s{i}image1"] = DEFAULT_ERROR_IMAGE

        progress.progress(i/6.0, text=f"Generating images… ({i}/6)")

    progress.empty()

    # portrait cover from slide 1 via CDN (640x853)
    try:
        if first_slide_key:
            out["potraitcoverurl"] = build_resized_cdn_url(AWS_BUCKET, first_slide_key, 640, 853)
        else:
            out["potraitcoverurl"] = DEFAULT_ERROR_IMAGE
    except Exception as e:
        st.info(f"Portrait cover URL build failed: {e}")
        out["potraitcoverurl"] = DEFAULT_ERROR_IMAGE

    return out

def generate_seo_metadata(chat_url: str, headers: dict, result_json: dict):
    seo_prompt = f"""
Generate SEO metadata for a web story with the following title and slide summaries.

Title: {result_json.get("storytitle","")}
Slides:
- {result_json.get("s2paragraph1","")}
- {result_json.get("s3paragraph1","")}
- {result_json.get("s4paragraph1","")}
- {result_json.get("s5paragraph1","")}
- {result_json.get("s6paragraph1","")}

Respond strictly in this JSON format:
{{
  "metadescription": "...",
  "metakeywords": "keyword1, keyword2, ..."
}}
"""
    payload_seo = {
        "messages": [
            {"role": "system", "content": "You are an expert SEO assistant."},
            {"role": "user", "content": seo_prompt}
        ],
        "temperature": 0.5,
        "max_tokens": 300
    }
    try:
        r = requests.post(chat_url, headers=headers, json=payload_seo, timeout=60)
        if r.status_code != 200:
            return "Explore this insightful story.", "web story, inspiration"
        content = r.json()["choices"][0]["message"]["content"]
        try:
            seo_data = json.loads(content)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", content)
            seo_data = json.loads(m.group(0)) if m else {}
        return seo_data.get("metadescription", "Explore this insightful story."), \
               seo_data.get("metakeywords", "web story, inspiration")
    except Exception:
        return "Explore this insightful story.", "web story, inspiration"

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2 = st.tabs(["Tab 1 — Notes → JSON", "Tab 2 — JSON → TTS (Azure)"])

# ===========================
# TAB 1
# ===========================
with tab1:
    uploaded_img = st.file_uploader("Upload a notes image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if st.button("Generate JSON", key="btn_tab1"):
        if not uploaded_img:
            st.error("Please upload an image first.")
            st.stop()

        # Read bytes + preview
        try:
            raw_bytes = uploaded_img.getvalue()
            if not raw_bytes:
                st.error("Uploaded file is empty.")
                st.stop()
            img = Image.open(BytesIO(raw_bytes)).convert("RGB")
            st.image(img, caption="Uploaded image", use_container_width=True)
        except Exception as e:
            st.error(f"Could not open image: {e}")
            st.stop()

        # Correct data URL with MIME
        mime = uploaded_img.type or "image/jpeg"
        if not (isinstance(mime, str) and mime.startswith("image/")):
            mime = "image/jpeg"
        base64_img = base64.b64encode(raw_bytes).decode("utf-8")
        user_content = [
            {"type": "text", "text": "Analyze this notes image and return the JSON."},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64_img}"}}
        ]

        # Azure chat (vision) to get JSON
        system_prompt = """
You are a teaching assistant. The student has uploaded a notes image.

Your job:
1) Extract a short and catchy title → storytitle
2) Summarise the whole content into 5 slides (s2paragraph1..s6paragraph1), each ≤ 400 characters.
3) For each paragraph (including the title), write a DALL·E prompt (s1alt1..s6alt1) for a 1024x1024 flat vector illustration: bright colors, clean lines, no text/captions/logos.

SAFETY & POSITIVITY RULES (MANDATORY):
- If the source includes hate, harassment, violence, adult content, self-harm, illegal acts, or extremist symbols, DO NOT reproduce them.
- Reinterpret into a positive, inclusive, family-friendly, educational scene (unity, learning, empathy, community, peace).
- Replace any hateful/violent symbol with abstract shapes, nature, or neutral motifs.
- Never include real people’s likeness or sensitive groups in a negative way.
- Avoid slogans, gestures, flags, trademarks, or captions. Absolutely NO TEXT in the image.

Respond strictly in this JSON format:
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
        chat_headers = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}
        chat_url = f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }

        with st.spinner("Generating structured JSON from the image…"):
            res = requests.post(chat_url, headers=chat_headers, json=payload, timeout=120)
            if res.status_code != 200:
                st.error(
                    f"Azure Chat error: {res.status_code} — {res.text[:300]}\n\n"
                    "Tip: ensure AZURE_DEPLOYMENT is a vision-capable model like 'gpt-4o' or 'gpt-4o-mini'."
                )
                st.stop()
            reply = res.json()["choices"][0]["message"]["content"]
            try:
                try:
                    result = json.loads(reply)
                except Exception:
                    m = re.search(r"\{[\s\S]*\}", reply)
                    result = json.loads(m.group(0)) if m else None
            except Exception as e:
                st.error(f"Model did not return valid JSON: {e}\n\nRaw:\n{reply[:500]}")
                st.stop()

        st.success("Structured JSON created.")
        st.json(result, expanded=False)

        # Generate DALL·E images → S3 → CDN URLs
        with st.spinner("Generating DALL·E images and uploading to S3…"):
            final_json = generate_and_upload_images(result)

        # SEO metadata
        with st.spinner("Generating SEO metadata…"):
            meta_desc, meta_keywords = generate_seo_metadata(chat_url, chat_headers, result)
            final_json["metadescription"] = meta_desc
            final_json["metakeywords"] = meta_keywords

        # Download JSON
        safe_title = result["storytitle"].replace(" ", "_").replace(":", "").lower()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"{safe_title}_{ts}.json"
        buf = io.StringIO()
        json.dump(final_json, buf, ensure_ascii=False, indent=2)
        content_str = buf.getvalue()

        st.success("✅ JSON ready (CDN resized URLs included)")
        st.json(final_json, expanded=False)
        st.download_button(
            "⬇️ Download JSON",
            data=content_str.encode("utf-8"),
            file_name=out_name,
            mime="application/json"
        )

# ===========================
# TAB 2 — JSON → Azure TTS
# ===========================
with tab2:
    st.subheader("Convert JSON to Audio (Azure Speech → S3)")
    st.caption("Uploads MP3 to S3 with UUID names and adds s#audio1 fields to your JSON.")

    # Quick sanity for TTS
    missing_tts = []
    for k in ["AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION", "VOICE_NAME", "AWS_ACCESS_KEY", "AWS_SECRET_KEY", "AWS_BUCKET", "S3_PREFIX", "CDN_BASE"]:
        if not get_secret(k):
            missing_tts.append(k)
    if missing_tts:
        st.info("(TTS) Missing in secrets: " + ", ".join(missing_tts))

    json_file = st.file_uploader("Upload your story JSON", type=["json"], key="json_uploader_tab2")
    run_tts = st.button("Generate Audio & Update JSON", key="btn_tab2")

    if run_tts:
        if not json_file:
            st.error("Please upload a JSON file first.")
            st.stop()

        try:
            json_data = json.load(json_file)
        except Exception as e:
            st.error(f"Could not parse JSON: {e}")
            st.stop()

        st.success("✅ JSON loaded.")
        st.json(json_data, expanded=False)

        # Import Azure Speech SDK
        try:
            import azure.cognitiveservices.speech as speechsdk
        except Exception as e:
            st.error("`azure-cognitiveservices-speech` is not installed. Add it to requirements.txt.\n"
                     f"Import error: {e}")
            st.stop()

        # Prepare AWS S3 client
        s3 = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION
        )

        # TTS setup
        speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
        speech_config.speech_synthesis_voice_name = VOICE_NAME

        # Mapping of JSON fields → audio field names
        field_mapping = {
            "storytitle":    "s1audio1",
            "s2paragraph1":  "s2audio1",
            "s3paragraph1":  "s3audio1",
            "s4paragraph1":  "s4audio1",
            "s5paragraph1":  "s5audio1",
            "s6paragraph1":  "s6audio1",
        }

        created = {}
        for field, audio_key in field_mapping.items():
            text = json_data.get(field)
            if not text:
                st.info(f"⚠️ Field missing: {field}")
                continue

            # Local output file
            uuid_name = f"{os.urandom(16).hex()}.mp3"
            audio_config = speechsdk.audio.AudioOutputConfig(filename=uuid_name)
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

            st.write(f"🎙️ Synthesizing: {field} → {uuid_name}")
            result = synthesizer.speak_text_async(text).get()

            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                # Upload to S3
                s3_key = f"{S3_PREFIX.rstrip('/')}/audio/{uuid_name}"
                try:
                    s3.upload_file(uuid_name, AWS_BUCKET, s3_key, ExtraArgs={"ContentType": "audio/mpeg"})
                    cdn_url = f"{CDN_BASE.rstrip('/')}/{s3_key}"
                    json_data[audio_key] = cdn_url
                    created[field] = cdn_url
                    st.write(f"✅ Uploaded to: {cdn_url}")
                except Exception as e:
                    st.error(f"Upload failed for {field}: {e}")
                finally:
                    try:
                        os.remove(uuid_name)
                    except Exception:
                        pass
            else:
                st.error(f"❌ Failed to synthesize: {field}")

        if created:
            st.success("✅ Audio URLs added to JSON")
            st.json(created, expanded=False)

        # Prepare updated JSON download
        safe_title = (json_data.get("storytitle","updated_story")
                      .replace(" ", "_").replace(":", "").lower())
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"{safe_title}_with_audio_{ts}.json"

        buf = io.StringIO()
        json.dump(json_data, buf, ensure_ascii=False, indent=2)
        content_str = buf.getvalue()

        st.download_button(
            "⬇️ Download Updated JSON (with audio fields)",
            data=content_str.encode("utf-8"),
            file_name=out_name,
            mime="application/json"
        )

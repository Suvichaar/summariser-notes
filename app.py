# app.py
# ------------------------------------------------------------
# Streamlit: Notes Image → GPT JSON → Safe DALL·E Images → S3 → JSON download
# Tab 1 only (ready for GitHub). Uses st.secrets for credentials.
# ------------------------------------------------------------
import os
import io
import re
import json
import time
import base64
import string
import random
import requests
import boto3
from io import BytesIO
from datetime import datetime, timezone
from PIL import Image
import streamlit as st

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Notes → JSON (GPT + DALL·E + S3)",
    page_icon="🧠",
    layout="centered"
)
st.title("🧠 Notes → JSON (GPT + DALL·E + S3)")
st.caption("Upload a notes image, get a structured JSON, safe image prompts, DALL·E images to S3, and download JSON.")

# ---------------------------
# Secrets / Config
# ---------------------------
def get_secret(key, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return default

AZURE_API_KEY     = get_secret("AZURE_API_KEY")
AZURE_ENDPOINT    = get_secret("AZURE_ENDPOINT")
AZURE_DEPLOYMENT  = get_secret("AZURE_DEPLOYMENT", "gpt-4o")   # ← vision-capable default
AZURE_API_VERSION = get_secret("AZURE_API_VERSION", "2024-08-01-preview")

DALE_ENDPOINT     = get_secret("DALE_ENDPOINT")  # full URL for images/generations endpoint
DAALE_KEY         = get_secret("DAALE_KEY")

AWS_ACCESS_KEY    = get_secret("AWS_ACCESS_KEY")
AWS_SECRET_KEY    = get_secret("AWS_SECRET_KEY")
AWS_REGION        = get_secret("AWS_REGION", "ap-south-1")
AWS_BUCKET        = get_secret("AWS_BUCKET")
S3_PREFIX         = get_secret("S3_PREFIX", "media")
DISPLAY_BASE      = get_secret("DISPLAY_BASE", "https://media.example.com")
DEFAULT_ERROR_IMAGE = get_secret("DEFAULT_ERROR_IMAGE", "https://media.example.com/default-error.jpg")

# Optional Azure Content Safety (text)
ACS_ENDPOINT           = get_secret("ACS_ENDPOINT")  # e.g., https://<your-cs>.cognitiveservices.azure.com
ACS_KEY                = get_secret("ACS_KEY")
ACS_API_VERSION        = get_secret("ACS_API_VERSION", "2023-10-01")
ACS_SEVERITY_THRESHOLD = int(get_secret("ACS_SEVERITY_THRESHOLD", 2))

# Fast validation
missing = []
for k in ["AZURE_API_KEY", "AZURE_ENDPOINT", "AZURE_DEPLOYMENT", "DAALE_KEY", "AWS_ACCESS_KEY", "AWS_SECRET_KEY", "AWS_BUCKET"]:
    if not get_secret(k):
        missing.append(k)
if not DALE_ENDPOINT:
    missing.append("DALE_ENDPOINT (full Azure DALL·E images/generations URL)")
if missing:
    st.warning("Add these secrets in `.streamlit/secrets.toml`: " + ", ".join(missing))

# ---------------------------
# Tabs (Tab 1 only)
# ---------------------------
tab1, = st.tabs(["Tab 1 — Notes → JSON"])

with tab1:
    uploaded_img = st.file_uploader("Upload a notes image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    run = st.button("Generate JSON")

    if run:
        if not uploaded_img:
            st.error("Please upload an image first.")
            st.stop()

        # --- Read bytes ONCE and preview ---
        try:
            raw_bytes = uploaded_img.getvalue()  # stable bytes in Streamlit
            if not raw_bytes:
                st.error("Uploaded file is empty.")
                st.stop()
            img = Image.open(BytesIO(raw_bytes)).convert("RGB")
            st.image(img, caption="Uploaded image", use_container_width=True)
        except Exception as e:
            st.error(f"Could not open image: {e}")
            st.stop()

        # --- Build correct data URL (mime-aware) ---
        mime = uploaded_img.type or "image/jpeg"  # e.g., "image/png" or "image/jpeg"
        if not (isinstance(mime, str) and mime.startswith("image/")):
            mime = "image/jpeg"
        base64_img = base64.b64encode(raw_bytes).decode("utf-8")
        if not base64_img:
            st.error("Base64 encoding failed (empty image data).")
            st.stop()

        user_content = [
            {"type": "text", "text": "Please analyze this notes image and return the structured JSON as requested."},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64_img}"}}
        ]

        # ------------ Azure Chat (vision) to get JSON ------------
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
        headers = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}
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
            res = requests.post(chat_url, headers=headers, json=payload, timeout=120)
            if res.status_code != 200:
                st.error(
                    "Azure Chat error: "
                    f"{res.status_code} — {res.text[:300]}\n\n"
                    "Tip: ensure AZURE_DEPLOYMENT is a vision-capable model like 'gpt-4o' or 'gpt-4o-mini'."
                )
                st.stop()
            reply = res.json()["choices"][0]["message"]["content"]
            # Parse JSON (with fallback extraction)
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

        # ------------ Optional: Azure Content Safety (text) ------------
        def content_safety_is_risky(text: str) -> bool:
            """Returns True if text crosses severity threshold on any category."""
            if not text or not ACS_ENDPOINT or not ACS_KEY:
                return False
            endpoint = f"{ACS_ENDPOINT}/contentsafety/text:analyze?api-version={ACS_API_VERSION}"
            headers_cs = {
                "Ocp-Apim-Subscription-Key": ACS_KEY,
                "Content-Type": "application/json"
            }
            payload_cs = {"text": text}
            try:
                r = requests.post(endpoint, headers=headers_cs, json=payload_cs, timeout=30)
                if r.status_code != 200:
                    st.info(f"Content Safety API non-200: {r.status_code}")
                    return False
                data = r.json()
                severities = []
                if isinstance(data, dict):
                    if "categoriesAnalysis" in data and isinstance(data["categoriesAnalysis"], list):
                        for cat in data["categoriesAnalysis"]:
                            sev = cat.get("severity")
                            if isinstance(sev, (int, float)):
                                severities.append(int(sev))
                    for key in ("hate", "violence", "sexual", "selfHarm"):
                        if key in data and isinstance(data[key], dict):
                            sev = data[key].get("severity")
                            if isinstance(sev, (int, float)):
                                severities.append(int(sev))
                max_sev = max(severities) if severities else 0
                return max_sev >= ACS_SEVERITY_THRESHOLD
            except Exception as e:
                st.info(f"Content Safety check failed: {e}")
                return False

        # ------------ Prompt Sanitizer ------------
        SAFE_FALLBACK = (
            "A joyful, abstract geometric illustration symbolizing unity and learning — "
            "soft shapes, harmonious gradients, friendly silhouettes, "
            "no text, no logos, no brands, no real persons, family-friendly, "
            "flat vector style, bright colors."
        )

        def sanitize_prompt(original_prompt: str) -> str:
            """Rewrite any risky prompt into a safe, positive, family-friendly version."""
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

            # Local guardrails if service fails
            return (original_prompt +
                    "\nFlat vector illustration, bright colors, no text, no logos, no brands, "
                    "no real persons, family-friendly, inclusive, peaceful.")

        # ------------ DALL·E + S3 upload ------------
        def generate_and_resize_images(result_json: dict) -> dict:
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
            final_json = {k: result_json[k] for k in result_json}

            progress = st.progress(0, text="Generating images…")
            for i in range(1, 7):
                raw_prompt = result_json.get(f"s{i}alt1", "")
                safe_prompt = raw_prompt

                # Pre-check with Content Safety (if configured)
                if content_safety_is_risky(safe_prompt):
                    safe_prompt = sanitize_prompt(safe_prompt)
                    if content_safety_is_risky(safe_prompt):
                        st.info(f"Slide {i}: sanitized prompt still risky → using fallback.")
                        safe_prompt = SAFE_FALLBACK

                payload = {"prompt": safe_prompt, "n": 1, "size": "1024x1024"}
                headers_dalle = {"Content-Type": "application/json", "api-key": DAALE_KEY}

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
                        im = Image.open(BytesIO(img_data)).convert("RGB")
                        im = im.resize((720, 1200))
                        buffer = BytesIO()
                        im.save(buffer, format="JPEG")
                        buffer.seek(0)
                        key = f"{S3_PREFIX.rstrip('/')}/{slug}/slide{i}.jpg"
                        s3.upload_fileobj(buffer, AWS_BUCKET, key, ExtraArgs={"ContentType": "image/jpeg"})
                        final_json[f"s{i}image1"] = f"{DISPLAY_BASE.rstrip('/')}/{key}"
                    except Exception as e:
                        st.info(f"Slide {i}: upload failed → {e}")
                        final_json[f"s{i}image1"] = DEFAULT_ERROR_IMAGE
                else:
                    final_json[f"s{i}image1"] = DEFAULT_ERROR_IMAGE

                progress.progress(i/6.0, text=f"Generating images… ({i}/6)")

            # Portrait cover from s1image1
            try:
                s1_url = final_json.get("s1image1")
                if s1_url and s1_url != DEFAULT_ERROR_IMAGE:
                    img_data = requests.get(s1_url, timeout=120).content
                    im = Image.open(BytesIO(img_data)).convert("RGB")
                    im = im.resize((640, 853))
                    buf = BytesIO()
                    im.save(buf, format="JPEG")
                    buf.seek(0)
                    portrait_key = f"{S3_PREFIX.rstrip('/')}/{slug}/portrait_cover.jpg"
                    s3.upload_fileobj(buf, AWS_BUCKET, portrait_key, ExtraArgs={"ContentType": "image/jpeg"})
                    final_json["potraitcoverurl"] = f"{DISPLAY_BASE.rstrip('/')}/{portrait_key}"
                else:
                    final_json["potraitcoverurl"] = DEFAULT_ERROR_IMAGE
            except Exception as e:
                st.info(f"Portrait cover generation failed: {e}")
                final_json["potraitcoverurl"] = DEFAULT_ERROR_IMAGE

            progress.empty()
            return final_json

        # ------------ SEO metadata ------------
        def generate_seo_metadata(result_json: dict):
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

        # ------------ Run images + SEO + save JSON ------------
        with st.spinner("Generating DALL·E images and uploading to S3…"):
            final_json = generate_and_resize_images(result)

        with st.spinner("Generating SEO metadata…"):
            meta_desc, meta_keywords = generate_seo_metadata(result)
            final_json["metadescription"] = meta_desc
            final_json["metakeywords"] = meta_keywords

        # Save JSON to memory, show + download
        safe_title = result["storytitle"].replace(" ", "_").replace(":", "").lower()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"{safe_title}_{ts}.json"
        buf = io.StringIO()
        json.dump(final_json, buf, ensure_ascii=False, indent=2)
        content_str = buf.getvalue()
        st.success("✅ JSON ready")
        st.json(final_json, expanded=False)
        st.download_button(
            "⬇️ Download JSON",
            data=content_str.encode("utf-8"),
            file_name=out_name,
            mime="application/json"
        )

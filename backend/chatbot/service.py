import io
import os

from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai

from chatbot.prompt_tempelate import build_image_context, format_user_prompt

_CHATBOT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_CHATBOT_DIR)
# Load in order: backend/.env first, then chatbot/.env (later overrides if both set)
load_dotenv(os.path.join(_BACKEND_DIR, ".env"))
load_dotenv(os.path.join(_CHATBOT_DIR, ".env"))


def _api_key() -> str:
    for name in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY2",
    ):
        key = (os.getenv(name) or "").strip().strip('"').strip("'")
        if key:
            return key
    return ""


def _model_name() -> str:
    # Prefer widely available models; override with GEMINI_MODEL in .env
    return (os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip().strip('"')


def _model_candidates() -> list[str]:
    """Try primary model first, then fallbacks if API rejects model id."""
    seen: set[str] = set()
    out: list[str] = []
    for m in (
        _model_name(),
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-3.5-flash",
        "gemini-flash-latest",
    ):
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _safety_settings():
    try:
        from google.generativeai.types import HarmCategory, HarmBlockThreshold

        th = getattr(HarmBlockThreshold, "BLOCK_ONLY_HIGH", None) or HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        return [
            {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": th},
            {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": th},
            {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": th},
            {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": th},
        ]
    except Exception:
        return None


def get_chatbot_config_public() -> dict:
    """Safe for API: no secrets. Used to verify Gemini is configured before chat."""
    return {
        "api_configured": bool(_api_key()),
        "model": _model_name(),
    }


def run_medical_chat(
    user_text: str,
    image_bytes: bytes | None = None,
    image_mime: str | None = None,
    context_prefix: str | None = None,
) -> str:
    """
    One assistant turn: text only, or text + image (multimodal).
    Optional context_prefix is prepended (e.g. linked patient lab / history).
    """
    text = (user_text or "").strip()
    if not text:
        raise ValueError("Message cannot be empty.")

    if context_prefix and context_prefix.strip():
        combined = (
            "=== SELECTED PRIOR REPORTS / LABS (full context) ===\n"
            + context_prefix.strip()
            + "\n\n=== CURRENT DOCTOR QUERY (answer using BOTH sections above and below) ===\n"
            + text.strip()
        )
    else:
        combined = text.strip()

    has_image = bool(image_bytes)
    image_context = build_image_context(has_image)
    prompt_body = format_user_prompt(combined, image_context)

    api_key = _api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    genai.configure(api_key=api_key)
    safety = _safety_settings()
    last_err: Exception | None = None

    for model_id in _model_candidates():
        try:
            try:
                model = (
                    genai.GenerativeModel(model_id, safety_settings=safety)
                    if safety
                    else genai.GenerativeModel(model_id)
                )
            except TypeError:
                model = genai.GenerativeModel(model_id)

            if has_image and image_bytes:
                try:
                    img = Image.open(io.BytesIO(image_bytes))
                    if img.mode not in ("RGB", "L"):
                        img = img.convert("RGB")
                except Exception as e:
                    raise ValueError("Could not read the image. Use PNG, JPG, or WebP.") from e
                response = model.generate_content([prompt_body, img])
            else:
                response = model.generate_content(prompt_body)

            if not response.candidates:
                block = None
                try:
                    pf = getattr(response, "prompt_feedback", None)
                    if pf is not None:
                        block = getattr(pf, "block_reason", None)
                except Exception:
                    pass
                raise RuntimeError(
                    "No response (blocked or empty)."
                    + (f" Block reason: {block}" if block else "")
                )

            parts = []
            for cand in response.candidates:
                if not cand.content or not cand.content.parts:
                    continue
                for part in cand.content.parts:
                    if hasattr(part, "text") and part.text:
                        parts.append(part.text)
            out = "\n".join(parts).strip()
            if not out and hasattr(response, "text") and response.text:
                out = response.text.strip()
            if not out:
                raise RuntimeError("The model returned no text.")
            return out
        except ValueError:
            raise
        except RuntimeError as e:
            print(f"[CHATBOT] Model {model_id} failed: {e}")
            last_err = e
            if "API_KEY_INVALID" in str(e):
                raise RuntimeError("Invalid API Key Error: Your API key is rejected by Google. Please check your .env file and ensure it is valid.")
            continue
        except Exception as e:
            print(f"[CHATBOT] Model {model_id} failed: {e}")
            last_err = e
            if "API_KEY_INVALID" in str(e):
                raise RuntimeError("Invalid API Key Error: Your API key is rejected by Google. Please check your .env file and ensure it is valid.")
            continue

    msg = str(last_err) if last_err else "Unknown error"
    raise RuntimeError(f"Gemini failed after trying alternate models. Last error: {msg}")


def extract_structured_with_ai(report_text: str) -> dict:
    """Uses Gemini API to strictly extract structured report sections into JSON."""
    if not report_text.strip():
        return {}
    api_key = _api_key()
    if not api_key:
        return {}
        
    prompt = f"""
    Extract the medical information strictly from the following doctor's clinical note.
    Return ONLY a raw JSON dictionary without any markdown blocks.
    
    Required keys:
    "case_description": (string) The patient's presentation and symptoms
    "primary_diagnosis": (string) Diagnosis name and percentage exactly as written
    "severity": (string) Just the severity word (e.g. MILD, MODERATE, RISK/SEVERE)
    "treatment": (string) Any general advice or treatment
    "medications": (list of dicts) Extract medications. Each dict MUST have keys: "name", "dosage", "frequency", "duration". If any missing, use empty string.
    "other_diagnoses": (string) Any other diagnoses listed
    "disclaimer": (string) The disclaimer text
    
    If a section is completely missing, return an empty string for it. 
    
    Clinical Note:
    {report_text}
    """
    
    genai.configure(api_key=api_key)
    for model_id in _model_candidates():
        try:
            model = genai.GenerativeModel(model_id)
            res = model.generate_content(prompt)
            txt = res.text.strip()
            if txt.startswith("```json"):
                txt = txt.split("```json")[-1]
            if txt.startswith("```"):
                txt = txt.split("```")[-1]
            if txt.endswith("```"):
                txt = txt.rsplit("```", 1)[0]
                
            import json
            return json.loads(txt.strip())
        except Exception as e:
            print(f"[EXTRACT_AI] Model {model_id} failed: {e}")
            continue
    return {}


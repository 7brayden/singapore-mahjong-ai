"""Diagnose the coach's LLM configuration.

The coach degrades silently to its deterministic template on any
failure — correct during play, unhelpful when setting up, because a
wrong deployment name and no credentials at all look identical from
the UI. This prints exactly what the server sees, then makes one real
(tiny) call so a misconfiguration surfaces as its actual error.

    PYTHONPATH=src python3 -m mahjong.coach.check

Costs a fraction of a cent — it sends a handful of tokens.
"""

import asyncio
import os
import sys

from mahjong.coach.explain import (
    DEFAULT_ANTHROPIC_MODEL, DEFAULT_AZURE_API_VERSION, _provider,
)

AZURE_VARS = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY",
              "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_API_VERSION"]
ANTHROPIC_VARS = ["ANTHROPIC_API_KEY", "COACH_MODEL"]

PROBE = ("Reply with exactly: OK")


def _show(name: str) -> str:
    """Report a variable without leaking a secret into the terminal."""
    value = os.environ.get(name, "")
    if not value:
        return f"  {name:<26} (not set)"
    if "KEY" in name:
        return f"  {name:<26} set, {len(value)} chars, ends …{value[-4:]}"
    return f"  {name:<26} {value}"


async def _probe_azure() -> str:
    from openai import AsyncAzureOpenAI

    missing = [v for v in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY",
                           "AZURE_OPENAI_DEPLOYMENT")
               if not os.environ.get(v, "").strip()]
    if missing:
        raise RuntimeError(f"missing required variables: {', '.join(missing)}")

    client = AsyncAzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"].strip(),
        api_key=os.environ["AZURE_OPENAI_API_KEY"].strip(),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION",
                                   DEFAULT_AZURE_API_VERSION).strip(),
        timeout=25.0,
    )
    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"].strip()
    try:
        resp = await client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": PROBE}],
            max_tokens=16)
    except Exception as exc:
        if "max_completion_tokens" not in str(exc):
            raise
        resp = await client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": PROBE}],
            max_completion_tokens=16)
    return (resp.choices[0].message.content or "").strip()


async def _probe_anthropic() -> str:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(timeout=25.0)
    resp = await client.messages.create(
        model=os.environ.get("COACH_MODEL", DEFAULT_ANTHROPIC_MODEL),
        max_tokens=1000,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": PROBE}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def _explain_failure(provider: str, exc: Exception) -> str:
    """Translate the usual setup errors into the fix."""
    text = str(exc)
    lowered = text.lower()
    if isinstance(exc, ModuleNotFoundError):
        extra = "coach" if provider == "azure" else "coach-anthropic"
        return (f"The SDK isn't installed. Run:\n"
                f'    pip install -e ".[server,{extra}]"')
    if provider == "azure":
        if "404" in text or "not found" in lowered:
            return ("404 — almost always AZURE_OPENAI_DEPLOYMENT holding a "
                    "model id instead of the DEPLOYMENT NAME you typed when "
                    "creating the deployment. Check the Deployments list in "
                    "Azure AI Foundry and copy the Name column exactly.\n"
                    "Less often: a wrong AZURE_OPENAI_ENDPOINT, or an "
                    "AZURE_OPENAI_API_VERSION your model doesn't support.")
        if "401" in text or "unauthorized" in lowered or "access denied" in lowered:
            return ("401 — the key doesn't match the resource. Copy KEY 1 "
                    "from the same resource whose endpoint you used "
                    "(Azure portal → your resource → Keys and Endpoint).")
        if "429" in text:
            return ("429 — rate/quota limit. Raise the deployment's "
                    "tokens-per-minute in Azure, or wait and retry.")
        if "connect" in lowered or "timeout" in lowered:
            return ("Network — check AZURE_OPENAI_ENDPOINT is the full "
                    "https://<resource>.openai.azure.com URL and reachable.")
    if provider == "anthropic" and ("401" in text or "authentication" in lowered):
        return "401 — check ANTHROPIC_API_KEY."
    return "See the error above."


def main() -> int:
    provider = _provider()
    pinned = os.environ.get("COACH_PROVIDER", "").strip()

    print("Coach configuration")
    print("=" * 60)
    print(f"  selected provider          {provider}"
          f"{'  (pinned by COACH_PROVIDER)' if pinned else ''}")
    print()
    print("Azure OpenAI")
    for v in AZURE_VARS:
        print(_show(v))
    if not os.environ.get("AZURE_OPENAI_API_VERSION", "").strip():
        print(f"  {'(api-version default)':<26} {DEFAULT_AZURE_API_VERSION}")
    print()
    print("Anthropic")
    for v in ANTHROPIC_VARS:
        print(_show(v))
    print()

    if provider == "template":
        print("No LLM backend configured — the coach will render its "
              "deterministic\ntemplate. That works, it is just less fluent. "
              "Set the variables above\nto enable a live backend.")
        return 0

    print(f"Calling {provider}…")
    try:
        reply = asyncio.run(
            _probe_azure() if provider == "azure" else _probe_anthropic())
    except Exception as exc:
        print(f"\n  FAILED: {type(exc).__name__}: {exc}\n")
        print(_explain_failure(provider, exc))
        return 1

    print(f"  OK — model replied: {reply!r}\n")
    print("The coach will use this backend. Restart the API container so it "
          "picks up\nthe same environment:  docker compose up -d")
    return 0


if __name__ == "__main__":
    sys.exit(main())

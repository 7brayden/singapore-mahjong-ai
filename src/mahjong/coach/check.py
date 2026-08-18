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
    azure_client, openai_compatible_client,
)

GENERIC_VARS = ["COACH_BASE_URL", "COACH_API_KEY", "COACH_MODEL"]
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


async def _probe_openai() -> str:
    model = os.environ.get("COACH_MODEL", "").strip()
    if not model:
        raise RuntimeError("COACH_MODEL is not set (the model this endpoint serves)")
    client = openai_compatible_client()
    resp = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": PROBE}], max_tokens=16)
    return (resp.choices[0].message.content or "").strip()


async def _probe_azure() -> str:
    missing = [v for v in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY",
                           "AZURE_OPENAI_DEPLOYMENT")
               if not os.environ.get(v, "").strip()]
    if missing:
        raise RuntimeError(f"missing required variables: {', '.join(missing)}")

    client, surface = azure_client()
    print(f"  endpoint surface           {surface}")
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
    if provider == "openai":
        if "404" in text or "not found" in lowered:
            return ("404 — COACH_BASE_URL or COACH_MODEL is wrong. The base "
                    "URL usually ends in /v1, and the model must be one this "
                    "endpoint actually serves.")
        if "401" in text or "unauthorized" in lowered:
            return "401 — check COACH_API_KEY for this endpoint."
        if "connect" in lowered or "timeout" in lowered:
            return ("Could not reach COACH_BASE_URL. If it is a local server "
                    "(Ollama, LM Studio), make sure it is running — and note "
                    "that from inside Docker, localhost is the container: use "
                    "http://host.docker.internal:11434/v1 instead.")
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
        if ("connect" in lowered or "timeout" in lowered
                or "name or service not known" in lowered
                or "nodename nor servname" in lowered):
            return (
                "The endpoint host did not resolve or could not be reached.\n"
                "Check in this order:\n"
                "  1. Does the resource still exist? Azure lab sandboxes\n"
                "     (hostnames like odl-user-1234567-...) are deleted when\n"
                "     the lab ends, taking their DNS record with them. A key\n"
                "     from an expired lab looks valid and resolves to nothing.\n"
                "  2. Is AZURE_OPENAI_ENDPOINT copied whole, including https://?\n"
                "  3. Resolve it yourself:  nslookup <host>\n"
                "     NXDOMAIN means the resource is gone, not misconfigured.")
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
    print("OpenAI-compatible endpoint (COACH_BASE_URL)")
    for v in GENERIC_VARS:
        print(_show(v))
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
        probe = {"openai": _probe_openai, "azure": _probe_azure,
                 "anthropic": _probe_anthropic}.get(provider)
        if probe is None:
            raise RuntimeError(
                f"unknown COACH_PROVIDER {provider!r} — "
                "use openai, azure, anthropic, or template")
        reply = asyncio.run(probe())
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

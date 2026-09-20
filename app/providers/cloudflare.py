"""
Cloudflare Workers AI provider, behind the AIProvider interface.

Cloudflare's request/response shape is meaningfully different from the
OpenAI-compatible vendors (app/providers/_openai_compatible.py): the URL
is per-account (.../accounts/{account_id}/ai/run/{model}), and a response
can come back as HTTP 200 with `"success": false` and an `errors` list
instead of a non-2xx status - so this adapter is implemented directly
against AIProvider rather than sharing that base class.

Provider name (see app/providers/factory.py): "cloudflare"
API base URL: https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}
Required env: JARVIS_CLOUDFLARE_API_KEY, JARVIS_CLOUDFLARE_ACCOUNT_ID
Model config: JARVIS_CLOUDFLARE_MODEL (see app/config.py for the default)
"""

import httpx

from app.config import settings
from app.providers.base import AIProvider, ProviderError

CLOUDFLARE_URL = (
    "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
)


class CloudflareError(ProviderError):
    """Raised when Cloudflare Workers AI can't be reached or returns
    something unusable."""


class CloudflareProvider(AIProvider):
    """AIProvider implementation backed by Cloudflare Workers AI."""

    def generate_reply(self, message: str) -> str:
        if not settings.cloudflare_api_key:
            raise CloudflareError("JARVIS_CLOUDFLARE_API_KEY is not set.")
        if not settings.cloudflare_account_id:
            raise CloudflareError("JARVIS_CLOUDFLARE_ACCOUNT_ID is not set.")

        url = CLOUDFLARE_URL.format(
            account_id=settings.cloudflare_account_id,
            model=settings.cloudflare_model,
        )

        try:
            response = httpx.post(
                url,
                headers={"Authorization": f"Bearer {settings.cloudflare_api_key}"},
                json={"messages": [{"role": "user", "content": message}]},
                timeout=120,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            # Built from status/body, never str(exc) or exc.request - same
            # discipline as GeminiProvider/OpenAICompatibleProvider so the
            # account id embedded in the URL, and the key in the
            # Authorization header, never leak into a CloudflareError.
            raise CloudflareError(
                f"Cloudflare returned {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise CloudflareError(f"Failed to reach Cloudflare: {exc}") from exc

        data = response.json()
        if not data.get("success", False):
            raise CloudflareError(f"Cloudflare reported failure: {data.get('errors')}")

        try:
            return data["result"]["response"]
        except (KeyError, TypeError) as exc:
            raise CloudflareError(f"Unexpected Cloudflare response: {data}") from exc

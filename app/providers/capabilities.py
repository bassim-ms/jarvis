"""
Provider-side capability metadata for M13 hybrid provider routing.

Static, non-secret metadata about what each registered provider (by the
same stable name used in app/providers/factory.py: "gemini", "groq",
"openrouter", "cloudflare") can be relied on for - used only by
ProviderRouter (app/providers/router.py) to filter/score candidates.
Deliberately imports nothing from app.config and never touches
settings/API keys/tokens - capability metadata lives in its own module,
entirely separate from credentials, so it's always safe to import, log, or
expose regardless of what's configured in .env.

Audited against the actual current model configuration (verified against
app/config.py's defaults AND the live .env, which match exactly): Gemini
`gemini-3.8-flash`, Groq `openai/gpt-oss-120b`, OpenRouter
`openrouter/auto`, Cloudflare `@cf/meta/llama-3.1-8b-instruct`, FreeLLMAPI
always `auto` (its own internal routing choice, not a fixed model - see
app/providers/freellmapi.py).

IMPORTANT distinction this audit is built on: whether a *vendor model*
can technically do something (e.g. gpt-oss and Llama 3.1 both document
native tool-calling/JSON-mode support at the vendor API level) is a
DIFFERENT question from whether *this project's adapter code* actually
invokes that vendor feature. It does not, for any provider:
- GeminiProvider.generate_reply sends only `{"contents": [...]}` -
  no `generationConfig.response_mime_type`/`response_schema`.
- OpenAICompatibleProvider (Groq/OpenRouter's shared base) sends only
  `{"model", "messages"}` - no `tools`/`tool_choice`/`response_format`.
- CloudflareProvider sends only `{"messages": [...]}`.
All four adapters are architecturally identical plain single-turn text
completions - "send text, get text back." Nothing here ever asks any
vendor's API for structured/tool-calling output through a dedicated
parameter, on any provider, including Gemini.

Given that, every capability assigned below is justified by demonstrated,
validated USE in this codebase - not by a vendor API feature, and not by
a benchmark:

- Gemini backs three specific call sites elsewhere in this project that
  no other provider has ever been engineered or tested for:
  GeminiCapabilityDecisionProvider (app/capability.py) prompts it for
  strict JSON and rigorously validates the response; the Gemini-backed
  ToolCallProvider (app/tools/decision.py) does the same for tool-call
  JSON; GeminiSearchAnswerGenerator (app/search/answer.py) prompts it to
  synthesize a grounded, cited answer from search evidence. STRUCTURED_
  OUTPUT and TOOL_USE record the first two facts - both require a
  response that reliably parses as strict, schema-conforming JSON, which
  only Gemini has an actual prompt-and-validator pair built for here.
  SEARCH_SYNTHESIS records the third, but is treated differently by
  ProviderRouter - see below.
- Groq, OpenRouter, Cloudflare, and FreeLLMAPI are all plain single-turn
  chat-completion adapters (see _openai_compatible.py, cloudflare.py,
  freellmapi.py) that this project has never prompted or validated for
  structured JSON or tool-call output, so none of them claim those two
  capabilities. This is an absence of in-repo validation, not a claim
  that their underlying models are incapable of it - if a future
  milestone builds and tests a prompt+validator pair for any of them,
  this registry should be updated to reflect that new, demonstrated
  fact. FreeLLMAPI specifically routes to whichever free-tier backend it
  picks per request (model="auto") - even if that backend could support
  tool-calling/structured output, Jarvis has no visibility into or control
  over which one answered a given request, so claiming either capability
  for it would be a guess about an unknown, request-varying backend, not
  a fact about this adapter.
- SEARCH_SYNTHESIS is a narrower, more clearly non-unique task than the
  two above: turning already-retrieved search evidence into a grounded,
  cited answer needs no structured/parseable output at all -
  GeminiSearchAnswerGenerator's citation extraction
  (app/search/answer.py:_extract_cited_sources) is plain Python string
  matching over free text, not JSON parsing. Architecturally this is
  indistinguishable from an ordinary GENERAL_CONVERSATION completion -
  every provider's adapter could attempt it. It's ALSO distinct from "can
  perform a web search": no AI provider in this project ever retrieves
  anything - that's WebSearchTool/SearXNG's job entirely (see
  app/search/), and SEARCH_SYNTHESIS never implies it. Because of this,
  ProviderRouter (app/providers/router.py) treats SEARCH_SYNTHESIS as a
  SOFT preference only, never a hard gate: Gemini still gets a small
  score bonus for a search-synthesis-flavored request, reflecting this
  project's actual tuning/testing history with that specific prompt, but
  no provider is ever excluded for lacking it.
- LONG_CONTEXT (gemini only) and LatencyTier.LOW (groq only) are
  DELIBERATE ROUTING PREFERENCES, not factual or benchmarked performance
  claims - this project has not measured either provider's context limit
  or latency. Gemini's model family is publicly documented as
  large-context; Groq's platform is publicly documented as built around
  low-latency (LPU) inference hardware. Both are recorded here only as an
  intentional, operator-adjustable signal for ProviderRouter's soft
  scoring (never a hard gate), so treat them as "prefer this provider
  when X is requested," not "this provider is objectively faster/bigger."
  FreeLLMAPI gets neither: its actual latency/context depend entirely on
  whichever free-tier backend it happens to route a given request to,
  which varies request to request and is invisible to Jarvis - there is no
  single, stable characteristic to record a preference for, so it's
  STANDARD latency and no LONG_CONTEXT claim, same as Cloudflare/
  OpenRouter.
- REASONING and CODING are deliberately unassigned to any provider:
  nothing in this codebase measures or differentiates provider quality on
  either axis, so claiming either would be inventing a capability rather
  than recording a demonstrated or intentionally-configured fact.
"""

from enum import Enum


class ProviderCapability(str, Enum):
    GENERAL_CONVERSATION = "general_conversation"
    REASONING = "reasoning"
    CODING = "coding"
    TOOL_USE = "tool_use"
    STRUCTURED_OUTPUT = "structured_output"
    LONG_CONTEXT = "long_context"
    SEARCH_SYNTHESIS = "search_synthesis"


class LatencyTier(str, Enum):
    STANDARD = "standard"
    LOW = "low"


PROVIDER_CAPABILITIES: dict[str, frozenset[ProviderCapability]] = {
    "gemini": frozenset(
        {
            ProviderCapability.GENERAL_CONVERSATION,
            ProviderCapability.STRUCTURED_OUTPUT,
            ProviderCapability.TOOL_USE,
            ProviderCapability.SEARCH_SYNTHESIS,
            ProviderCapability.LONG_CONTEXT,
        }
    ),
    "groq": frozenset({ProviderCapability.GENERAL_CONVERSATION}),
    "openrouter": frozenset({ProviderCapability.GENERAL_CONVERSATION}),
    "cloudflare": frozenset({ProviderCapability.GENERAL_CONVERSATION}),
    "freellmapi": frozenset({ProviderCapability.GENERAL_CONVERSATION}),
}

PROVIDER_LATENCY: dict[str, LatencyTier] = {
    "gemini": LatencyTier.STANDARD,
    "groq": LatencyTier.LOW,
    "openrouter": LatencyTier.STANDARD,
    "cloudflare": LatencyTier.STANDARD,
    "freellmapi": LatencyTier.STANDARD,
}


def capabilities_for(name: str) -> frozenset[ProviderCapability]:
    """Registered capabilities for `name`, or an empty set for an
    unrecognized name - never raises, since an unknown provider should
    just mean "no capabilities claimed," not a router crash."""
    return PROVIDER_CAPABILITIES.get(name, frozenset())


def latency_tier_for(name: str) -> LatencyTier:
    """Registered latency tier for `name`, defaulting to STANDARD for an
    unrecognized name."""
    return PROVIDER_LATENCY.get(name, LatencyTier.STANDARD)

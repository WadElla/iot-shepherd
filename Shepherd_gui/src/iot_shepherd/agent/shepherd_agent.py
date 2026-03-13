from __future__ import annotations

from typing import List, Any
from textwrap import dedent

from agno.models.ollama import Ollama
from agno.agent import Agent

from .tools import cgm_retrieve, adm_analyze_pcap

try:
    from agno.tools.duckduckgo import DuckDuckGoTools  # requires ddgs
except Exception:  # pragma: no cover
    DuckDuckGoTools = None  # type: ignore


from textwrap import dedent

from textwrap import dedent

def _instructions(allow_adm: bool, allow_web: bool) -> str:
    tools_line = []
    if allow_adm:
        tools_line.append(
            "- **adm_analyze_pcap(pcap_path, max_packets)** → returns JSON with type='incident_card'."
        )
    tools_line.append(
        "- **cgm_retrieve(query, k)** → returns JSON with type='kb_context' and top chunks (each has an id)."
    )
    if allow_web:
        tools_line.append(
            "- **DuckDuckGoTools** (optional) → external search ONLY if enabled and ONLY when manuals are insufficient."
        )

    adm_rule = ""
    if not allow_adm:
        adm_rule = dedent("""
        IMPORTANT: In this session, ADM tool-calls are DISABLED.
        - Do NOT attempt to call adm_analyze_pcap().
        - If the user asks for traffic analysis, tell them to run Traffic Analysis (ADM) in the UI to generate an Incident Card.
        """)

    web_rule = ""
    if not allow_web:
        web_rule = "\nIMPORTANT: Web search is DISABLED. Use manuals evidence only.\n"

    return dedent(f"""\
    You are the **IoT Shepherd Agent** (agentic mode). You coordinate IoT Shepherd's modules exactly as described in the paper.

    Goal: produce **operator-ready mitigation guidance** that is:
    1) grounded in **local manuals evidence** whenever possible, and
    2) consistent with the **Incident Card** when present.

    Tools you can call:
    {chr(10).join(tools_line)}

    {adm_rule}{web_rule}

    # What "Manual Evidence" means (non-negotiable definition)
    Manual Evidence is the **mitigation strategies / procedures / configuration steps** that exist in the **indexed manuals**.
    In this system, **Manual Evidence is ONLY the content returned by the `cgm_retrieve` tool**.

    Manual Evidence is NOT:
    - the retrieval queries you generate,
    - the tool calls you plan to make,
    - or any text you invent from memory.

    A valid Manual Evidence item MUST:
    - come from the tool output of `cgm_retrieve` in THIS session, and
    - cite at least one **chunk id** returned by `cgm_retrieve`, and
    - include a short paraphrase (1 sentence) of what that chunk says that supports your mitigation step.

    If you did not successfully call `cgm_retrieve` OR it returned no relevant chunks, then you have **NO Manual Evidence**.
    In that case, you MUST write exactly:
    **"No relevant manual evidence found in the indexed manuals."**
    and then proceed with best-effort guidance grounded in the Incident Card.

    IMPORTANT: Retrieved chunks may exist but still be unrelated to the incident.
    - If the retrieved chunks do NOT actually address the incident/question, you MUST STILL report:
      **"No relevant manual evidence found in the indexed manuals."**
    - In that situation, you MAY still provide mitigation actions based on your general security/operations knowledge,
      but you MUST NOT present those actions as being supported by the manuals.
      Do NOT cite chunk ids when evidence is irrelevant.

    Terminology (do not confuse these):
    - Query: a natural-language search phrase you generate to retrieve manuals (e.g., "factory reset procedure").
    - Manual Chunk: a piece of manual text returned by `cgm_retrieve`.
    - Manual Evidence: the subset of returned chunks you use to justify actions, cited by chunk id.

    ### Workflow (follow strictly)

    0) If you call adm_analyze_pcap:
       - You MUST use the returned Incident Card fields as your Traffic Evidence.
       - Do NOT fabricate attacks/endpoints.

    1) Understand the request
       - If an Incident Card is already provided, do NOT call ADM again unless explicitly requested and ADM tool-calls are enabled.
       - NEVER call cgm_retrieve with an empty query.
       - When you need manuals evidence, you MUST invoke tools via the tool mechanism (function calling).
         Do NOT print tool-call JSON or tool schemas in your answer.

    2) Extract incident signals (when Incident Card is present)
       - Identify: dominant attack/anomaly, top attacks, affected endpoints, protocols/ports (if present), and severity/confidence if available.
       - Summarize incident evidence in 2–4 bullets before proposing actions.

    3) Generate a retrieval plan (manuals-first)
       - Formulate retrieval queries directly from the Incident Card signals (dominant attack/anomaly, affected endpoints, protocols/ports, device/vendor context) and the admin question.
       - Every query must be a concrete natural-language phrase (>= 8 characters) that a manual is likely to contain:
         procedures, settings, configuration steps, recovery steps, logging/forensics steps, credential rotation, remote access control.
       - Create 4–8 targeted retrieval queries mapping the incident to manuals guidance.
       - Execute 3–6 cgm_retrieve calls.
         - Use k=5 by default (recommended). If unsure, use k=5.
       - If chunk_count=0 or evidence is irrelevant, refine your query phrasing and try again (within the call budget).
       - Prefer local manuals first; web search is a last resort (only if enabled).

    4) Write the final response (strict structure)
       - Incident summary (2–4 bullets)
       - Manual evidence:
         - If you have relevant chunks, list 2–6 bullets in this exact style:
           - [chunk_id] → one-sentence paraphrase of what the manual says if only it is relevant to the incident (relevant to the incident)
         - If you have no relevant chunks OR retrieved chunks are not relevant, write exactly:
           - No relevant manual evidence found in the indexed manuals.
       - Mitigation actions (Contain → Diagnose → Remediate → Monitor)
         - Each action should be specific, step-by-step, and operationally actionable.
         - When manuals evidence exists and is relevant, reference the chunk ids in-line (e.g., "... per [chunk_id]").
         - If manuals evidence is missing or irrelevant, you may still propose best-practice actions,
           but you MUST clearly treat them as general guidance (not manual-supported) and MUST NOT cite chunk ids.
       - Notes / limits (what evidence was missing, assumptions, what to do next)

    ### Evidence integrity rules (non-negotiable)
    - You MUST NOT hallucinate manual evidence.
    - The Manual evidence section may ONLY cite chunk ids that appear in actual cgm_retrieve tool results from this session.
    - Do NOT invent page numbers, sections, or quotes.
    - Do NOT cite chunk ids for irrelevant evidence.
    - Do NOT include raw tool-call JSON, file paths, scores, or full excerpts in the final answer.
    - If web search is enabled and used, clearly label it as External Evidence and keep it separate from Manual Evidence.
    """)


def build_shepherd_agent(
    llm_model: str = "llama3.2:latest",
    ollama_host: str = "http://localhost:11434",
    enable_web_search: bool = True,
    allow_adm: bool = False,
) -> Agent:
    """Build the Shepherd Agent.

    Robustness rules:
    - ADM tool is OPTIONAL and off by default (admin-controlled).
    - When allow_adm=False, the agent cannot call ADM even if it wants to.
    """
    tools: List[Any] = [cgm_retrieve]
    if allow_adm:
        tools.append(adm_analyze_pcap)

    web_enabled_effective = False
    if enable_web_search and DuckDuckGoTools is not None:
        try:
            tools.append(DuckDuckGoTools())
            web_enabled_effective = True
        except Exception:
            web_enabled_effective = False

    instr = _instructions(allow_adm=allow_adm, allow_web=web_enabled_effective)

    # Agno version drift: Ollama model constructor and Agent constructor may differ across versions.
    try:
        model = Ollama(id=llm_model, host=ollama_host)
    except TypeError:
        try:
            model = Ollama(model=llm_model, host=ollama_host)
        except TypeError:
            model = Ollama(llm_model, host=ollama_host)

    # Agent version drift: some versions accept `instructions=`, others `system=`
    try:
        return Agent(
            name="IoT Shepherd Agent",
            model=model,
            tools=tools,
            markdown=True,
            instructions=instr,
        )
    except TypeError:
        return Agent(
            name="IoT Shepherd Agent",
            model=model,
            tools=tools,
            markdown=True,
            system=instr,
        )

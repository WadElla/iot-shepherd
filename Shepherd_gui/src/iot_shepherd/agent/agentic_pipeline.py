from __future__ import annotations

import json
import re
import time
import html as _html
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode, urlparse, parse_qs, unquote
from urllib.request import Request, urlopen

from agno.agent import Agent
from agno.models.ollama import Ollama

from ..cgm.retrieval import retrieve_context
from ..config import AppSettings


# ---------------------------
# Utilities
# ---------------------------

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_obj(text: str) -> Optional[dict]:
    """Best-effort extraction of a single JSON object from model output."""
    if not text:
        return None
    t = text.strip()

    # Strip common fences
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)

    m = _JSON_RE.search(t)
    if not m:
        return None
    blob = m.group(0)
    try:
        return json.loads(blob)
    except Exception:
        return None


def _normalize_run_output(resp: Any) -> str:
    """Normalize outputs across Agno versions."""
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    # Some versions return an object with `.content` or `.response`
    for attr in ("content", "response", "text", "output"):
        if hasattr(resp, attr):
            try:
                v = getattr(resp, attr)
                if isinstance(v, str):
                    return v
            except Exception:
                pass
    if isinstance(resp, dict):
        for k in ("content", "response", "text", "output"):
            if k in resp and isinstance(resp[k], str):
                return resp[k]
    return str(resp)


def _call_agent_text(agent: Agent, message: str) -> str:
    """Call an Agno Agent across method drift: run/respond/chat/invoke."""
    for method in ("run", "respond", "chat", "invoke"):
        if hasattr(agent, method):
            try:
                resp = getattr(agent, method)(message)
            except TypeError:
                resp = getattr(agent, method)(message=message)
            return _normalize_run_output(resp)
    return ""


def _build_plain_llm(llm_model: str, ollama_host: str, system: str, temperature: float = 0.1) -> Agent:
    """An Agent wrapper with NO tools; used for stable structured outputs and final writing.

    Agno's Ollama + Agent signatures drift across versions. This helper is defensive.
    """
    model_kwargs = {
        "id": llm_model,
        "host": ollama_host,
        "options": {"temperature": float(temperature)},
    }

    # Keep streaming off to avoid partial JSON in Streamlit (when supported)
    try:
        model = Ollama(**model_kwargs, stream=False)
    except TypeError:
        model = Ollama(**model_kwargs)

    # Agent signature drift (`instructions` vs `system`)
    try:
        return Agent(name="IoT Shepherd (plain)", model=model, markdown=True, instructions=system)
    except TypeError:
        return Agent(name="IoT Shepherd (plain)", model=model, markdown=True, system=system)


# ---------------------------
# Public API types
# ---------------------------

@dataclass
class RetrievalChunk:
    id: str
    source: str
    page: Optional[int]
    score: Optional[float]
    excerpt: str


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str


@dataclass
class AgenticResult:
    queries: List[str]
    chunks: List[RetrievalChunk]
    answer: str
    retrieval_ok: bool
    retrieval_error: Optional[str]
    web_used: bool
    web_query: Optional[str]
    web_results: List[WebResult]
    web_error: Optional[str]


# ---------------------------
# LLM planning: manual queries
# ---------------------------

def generate_queries_from_incident(
    incident_card: Dict[str, Any],
    question: Optional[str],
    llm_model: str,
    ollama_host: str,
    n_queries: int = 2,
) -> Tuple[List[str], str]:
    """LLM-only query generation. Returns (queries, raw_model_text)."""
    sys = (
        "You are the IoT Shepherd Agent.\n"
        "Given an Incident Card (JSON) and an optional admin question, generate 1-2 concise retrieval queries\n"
        "to search the indexed IoT manuals for mitigation / remediation steps.\n\n"
        "Return ONLY valid JSON with schema: {\"queries\": [\"...\", \"...\"]}.\n"
        "Rules:\n"
        f"- Return exactly {max(1, min(int(n_queries), 2))} queries.\n"
        "- Each query must be short (<= 10 words), actionable, and mitigation-oriented.\n"
        "- Do NOT include markdown, prose, or extra keys.\n"
    )

    incident_json = json.dumps(incident_card, indent=2, ensure_ascii=False)
    user = "INCIDENT_CARD:\n" + incident_json + "\n"
    if question:
        user += f"\nADMIN_QUESTION:\n{question}\n"

    planner = _build_plain_llm(llm_model, ollama_host, sys, temperature=0.1)
    raw = _call_agent_text(planner, user)

    obj = _extract_json_obj(raw)
    queries: List[str] = []
    if obj and isinstance(obj.get("queries"), list):
        queries = [str(q).strip() for q in obj["queries"] if str(q).strip()]

    # Repair attempt
    if not queries:
        repair_sys = "Return ONLY valid JSON: {\"queries\": [\"...\"]}. No prose."
        repair = _build_plain_llm(llm_model, ollama_host, repair_sys, temperature=0.0)
        raw2 = _call_agent_text(repair, raw)
        obj2 = _extract_json_obj(raw2)
        if obj2 and isinstance(obj2.get("queries"), list):
            queries = [str(q).strip() for q in obj2["queries"] if str(q).strip()]
        raw = raw2 if raw2 else raw

    # Validate & clamp
    cleaned: List[str] = []
    for q in queries:
        q2 = re.sub(r"[\r\n\t]+", " ", q).strip()
        q2 = re.sub(r"\s{2,}", " ", q2).strip()
        if not q2:
            continue
        words = q2.split()
        if len(words) > 10:
            q2 = " ".join(words[:10])
        cleaned.append(q2[:120])

    # Enforce count 1-2
    cleaned = cleaned[: max(1, min(int(n_queries), 2))]
    if not cleaned:
        cleaned = ["mitigation steps for detected IoT incident"]

    return cleaned, (raw or "")


# ---------------------------
# Deterministic execution: manuals retrieval
# ---------------------------

def _merge_dedupe_chunks(chunks: List[RetrievalChunk], k: int = 5) -> List[RetrievalChunk]:
    """Dedupe chunks by id, keep first occurrence order, and clamp excerpt length."""
    k = int(k) if k else 5
    k = max(1, min(k, 20))
    seen = set()
    out: List[RetrievalChunk] = []
    for c in chunks or []:
        cid = (c.id or "").strip()
        if not cid:
            continue
        if cid in seen:
            continue
        seen.add(cid)
        ex = (c.excerpt or "").strip()
        if len(ex) > 900:
            ex = ex[:900] + "…"
        c.excerpt = ex
        out.append(c)
        if len(out) >= k:
            break
    return out

def retrieve_manual_evidence(
    queries: List[str],
    k: int,
    settings: AppSettings,
) -> Tuple[List[RetrievalChunk], bool, Optional[str]]:
    """Run CGM retrieval for each query (dict-based payload), dedupe, and return top-k chunks overall."""
    k = int(k) if k else 5
    k = max(1, min(k, 12))

    all_chunks: List[RetrievalChunk] = []
    any_ok = False
    last_err: Optional[str] = None

    chroma_dir = settings.chroma_dir
    embed_model = settings.embed_model

    for q in (queries or []):
        q = (q or "").strip()
        if not q:
            continue
        try:
            payload = retrieve_context(
                query=q,
                chroma_dir=chroma_dir,
                embed_model=embed_model,
                k=k,
            )
            ok = bool(payload.get("ok", False))
            if ok:
                any_ok = True
            err = payload.get("error")
            if err:
                last_err = str(err)

            for item in payload.get("results", []) or []:
                chunk_id = str(item.get("id") or "")
                if not chunk_id:
                    chunk_id = f"chunk_{len(all_chunks)+1}"
                all_chunks.append(
                    RetrievalChunk(
                        id=chunk_id,
                        source=str(item.get("source") or "manual"),
                        page=item.get("page"),
                        score=item.get("score"),
                        excerpt=str(item.get("excerpt") or item.get("text") or ""),
                    )
                )
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"

    return _merge_dedupe_chunks(all_chunks, k=k), any_ok, last_err

# ---------------------------
# Web search: DuckDuckGo HTML
# ---------------------------

class _DDGParser(HTMLParser):
    def __init__(self, max_results: int):
        super().__init__()
        self.max_results = max_results
        self.results: List[WebResult] = []
        self._in_title = False
        self._in_snippet = False
        self._cur_href: Optional[str] = None
        self._cur_title: List[str] = []
        self._cur_snippet: List[str] = []

    def handle_starttag(self, tag, attrs):
        if len(self.results) >= self.max_results:
            return
        attrs_dict = {k: v for k, v in attrs if k}
        cls = attrs_dict.get("class", "") or ""

        if tag == "a" and "result__a" in cls:
            self._in_title = True
            self._cur_href = attrs_dict.get("href")
            self._cur_title = []

        if tag in ("a", "div", "span") and "result__snippet" in cls:
            self._in_snippet = True
            self._cur_snippet = []

    def handle_endtag(self, tag):
        if tag == "a" and self._in_title:
            self._in_title = False

        if tag in ("a", "div", "span") and self._in_snippet:
            self._in_snippet = False
            if self._cur_href and self._cur_title:
                title = _html.unescape("".join(self._cur_title).strip())
                snippet = _html.unescape("".join(self._cur_snippet).strip())
                url = _clean_ddg_url(self._cur_href)
                if title and url:
                    self.results.append(WebResult(title=title, url=url, snippet=snippet))

    def handle_data(self, data):
        if len(self.results) >= self.max_results:
            return
        if self._in_title:
            self._cur_title.append(data)
        if self._in_snippet:
            self._cur_snippet.append(data)


def _clean_ddg_url(href: str) -> str:
    if not href:
        return ""
    try:
        u = urlparse(href)
        if u.path.startswith("/l/"):
            qs = parse_qs(u.query)
            if "uddg" in qs and qs["uddg"]:
                return unquote(qs["uddg"][0])
        return href
    except Exception:
        return href


def duckduckgo_search(query: str, *, max_results: int = 5, timeout_s: int = 8) -> Tuple[List[WebResult], Optional[str]]:
    """Return (results, error). Never raises."""
    q = (query or "").strip()
    if not q:
        return [], "Empty web query."
    try:
        url = "https://duckduckgo.com/html/?" + urlencode({"q": q})
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=timeout_s) as resp:
            html_text = resp.read().decode("utf-8", errors="ignore")
        parser = _DDGParser(max_results=max_results)
        parser.feed(html_text)
        return parser.results[:max_results], None
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"


# ---------------------------
# LLM decision: whether to web-search
# ---------------------------

def decide_more_manual_search(
    incident_card: Dict[str, Any],
    question: Optional[str],
    queries_so_far: List[str],
    chunks_so_far: List[RetrievalChunk],
    llm_model: str,
    ollama_host: str,
    *,
    max_new_queries: int = 1,
) -> Tuple[bool, List[str], str]:
    """ReAct-style decision: LLM decides if manuals evidence is insufficient and proposes extra manuals queries.

    Returns (need_more, new_queries, raw_text).
    """
    max_new = max(1, min(int(max_new_queries), 2))
    sys = f"""You are the IoT Shepherd Agent.
You have already retrieved some IoT manual evidence chunks.
Decide whether the current manuals evidence is sufficient to provide mitigation guidance.
If it is insufficient, propose additional manuals retrieval queries to fetch better mitigation steps.

Return ONLY valid JSON with schema: {{"need_more": true/false, "queries": ["..."]}}.
Rules:
- If need_more=true, return 1 query (max {max_new}). If need_more=false, return an empty list.
- Each query must be short (<= 10 words), mitigation-oriented, and NOT a duplicate of prior queries.
- Do NOT include markdown, prose, or extra keys.
"""


    incident_json = json.dumps(incident_card, indent=2, ensure_ascii=False)
    prompt = "INCIDENT_CARD:\n" + incident_json + "\n\n"
    if question:
        prompt += f"ADMIN_QUESTION:\n{question}\n\n"

    prompt += "QUERIES_SO_FAR:\n" + "\n".join([f"- {q}" for q in (queries_so_far or [])]) + "\n\n"
    prompt += f"EVIDENCE_CHUNKS_COUNT: {len(chunks_so_far or [])}\n"
    if chunks_so_far:
        prompt += "EVIDENCE_EXCERPTS (top):\n"
        for c in (chunks_so_far or [])[:4]:
            ex = (c.excerpt or "").replace("\n", " ").strip()
            ex = ex[:260] + ("…" if len(ex) > 260 else "")
            prompt += f"- [chunk:{c.id}] {c.source} p={c.page} score={c.score} ex={ex}\n"
    else:
        prompt += "EVIDENCE_EXCERPTS:\n<none>\n"

    decider = _build_plain_llm(llm_model, ollama_host, sys, temperature=0.0)
    raw = _call_agent_text(decider, prompt)

    obj = _extract_json_obj(raw)
    need_more = False
    new_qs: List[str] = []
    if obj and isinstance(obj.get("need_more"), bool):
        need_more = bool(obj.get("need_more"))
    if obj and isinstance(obj.get("queries"), list):
        new_qs = [str(x).strip() for x in obj.get("queries", []) if str(x).strip()]

    # Repair attempt
    if need_more and not new_qs:
        repair_sys = 'Return ONLY valid JSON: {"need_more": true/false, "queries": ["..."]}. No prose.'
        repair = _build_plain_llm(llm_model, ollama_host, repair_sys, temperature=0.0)
        raw2 = _call_agent_text(repair, raw)
        obj2 = _extract_json_obj(raw2)
        if obj2 and isinstance(obj2.get("need_more"), bool):
            need_more = bool(obj2.get("need_more"))
        if obj2 and isinstance(obj2.get("queries"), list):
            new_qs = [str(x).strip() for x in obj2.get("queries", []) if str(x).strip()]
        raw = raw2 if raw2 else raw

    # Clean and dedupe against prior queries
    prior = set([(q or "").strip().lower() for q in (queries_so_far or []) if (q or "").strip()])
    cleaned: List[str] = []
    for q in new_qs:
        q2 = re.sub(r"[\r\n\t]+", " ", q).strip()
        q2 = re.sub(r"\s{2,}", " ", q2).strip()
        if not q2:
            continue
        words = q2.split()
        if len(words) > 10:
            q2 = " ".join(words[:10])
        if q2.lower() in prior:
            continue
        cleaned.append(q2[:120])
        if len(cleaned) >= max_new:
            break

    if not need_more:
        cleaned = []
    return need_more, cleaned, (raw or "")

def decide_web_search(
    incident_card: Dict[str, Any],
    question: Optional[str],
    queries: List[str],
    chunks: List[RetrievalChunk],
    llm_model: str,
    ollama_host: str,
) -> Tuple[bool, Optional[str], str]:
    """LLM decides whether to run ONE web search and provides a query if needed."""
    sys = """You are the IoT Shepherd Agent.
Decide whether you need ONE external web search (DuckDuckGo) to supplement the manuals evidence.
Prefer manuals. Use the web ONLY if the manuals evidence is missing or clearly insufficient for mitigation.

Return ONLY valid JSON with schema: {"use_web": true/false, "query": "..." or null}.
Rules:
- If you set use_web=true, query must be concise (<= 12 words) and specific to mitigation/steps.
- Do not include quotes, markdown, or extra keys.
"""

    incident_json = json.dumps(incident_card, indent=2, ensure_ascii=False)
    prompt = "INCIDENT_CARD:\n" + incident_json + "\n\n"
    if question:
        prompt += f"ADMIN_QUESTION:\n{question}\n\n"

    prompt += "MANUAL_RETRIEVAL_QUERIES_USED:\n" + "\n".join([f"- {q}" for q in queries]) + "\n\n"
    prompt += f"MANUAL_EVIDENCE_CHUNKS_COUNT: {len(chunks)}\n"
    if chunks:
        prompt += "MANUAL_EVIDENCE_SUMMARY:\n"
        for c in chunks[:3]:
            ex = (c.excerpt or "").replace("\n", " ").strip()
            ex = ex[:240] + ("…" if len(ex) > 240 else "")
            prompt += f"- [chunk:{c.id}] {c.source} p={c.page} score={c.score} ex={ex}\n"
    else:
        prompt += "MANUAL_EVIDENCE_SUMMARY:\n<none>\n"

    decider = _build_plain_llm(llm_model, ollama_host, sys, temperature=0.0)
    raw = _call_agent_text(decider, prompt)

    obj = _extract_json_obj(raw)
    use_web = False
    web_q: Optional[str] = None
    if obj and isinstance(obj.get("use_web"), (bool, int)):
        use_web = bool(obj.get("use_web"))
        qv = obj.get("query")
        if isinstance(qv, str) and qv.strip():
            web_q = qv.strip()

    if use_web and not web_q:
        repair_sys = "Return ONLY valid JSON: {\"use_web\": true/false, \"query\": \"...\" or null}. No prose."
        repair = _build_plain_llm(llm_model, ollama_host, repair_sys, temperature=0.0)
        raw2 = _call_agent_text(repair, raw)
        obj2 = _extract_json_obj(raw2)
        if obj2 and isinstance(obj2.get("use_web"), (bool, int)):
            use_web = bool(obj2.get("use_web"))
            qv = obj2.get("query")
            if isinstance(qv, str) and qv.strip():
                web_q = qv.strip()
        raw = raw2 if raw2 else raw

    if web_q:
        web_q = re.sub(r"[\r\n\t]+", " ", web_q).strip()
        if len(web_q.split()) > 12:
            web_q = " ".join(web_q.split()[:12])
        web_q = web_q[:120]

    if use_web and not web_q:
        use_web = False

    return use_web, web_q, (raw or "")


# ---------------------------
# LLM: final response
# ---------------------------

def generate_final_answer(
    incident_card: Dict[str, Any],
    question: Optional[str],
    queries: List[str],
    chunks: List[RetrievalChunk],
    llm_model: str,
    ollama_host: str,
    *,
    web_used: bool,
    web_query: Optional[str],
    web_results: List[WebResult],
    web_error: Optional[str],
) -> str:
    """LLM-only final response, grounded when evidence is available."""
    sys = (
        "You are the IoT Shepherd Agent (agentic mode).\n"
        "You must write the final response for the administrator.\n\n"
        "CRITICAL RULES:\n"
        "1) You MUST use the Incident Card as the ground truth for what happened.\n"
        "2) If manual evidence chunks are provided, you MUST ground mitigation steps in them and cite chunk IDs.\n"
        "3) If no manual evidence is provided (empty), explicitly say: \"No relevant manual evidence was found in the indexed manuals\".\n"
        "4) Do NOT invent citations. Only cite provided chunk IDs and provided web items.\n"
        "5) If web evidence is provided, cite it as [web:1], [web:2], etc corresponding to WEB_RESULTS.\n\n"
        "Output structure:\n"
        "- Incident summary (2-4 bullets)\n"
        "- Manual evidence (bullets with [chunk:<id>] citations) or a one-line 'no evidence' statement\n"
        "- External evidence (optional, bullets with [web:<n>] citations)\n"
        "- Mitigation actions (Contain → Diagnose → Remediate → Monitor)\n"
        "- Notes / limits\n"
    )

    incident_json = json.dumps(incident_card, indent=2, ensure_ascii=False)
    prompt = "INCIDENT_CARD:\n" + incident_json + "\n\n"
    if question:
        prompt += f"ADMIN_QUESTION:\n{question}\n\n"
    prompt += "RETRIEVAL_QUERIES_USED:\n" + "\n".join([f"- {q}" for q in queries]) + "\n\n"

    if chunks:
        prompt += "MANUAL_EVIDENCE_CHUNKS:\n"
        for c in chunks[:12]:
            prompt += (
                f"\n[chunk:{c.id}] source={c.source} page={c.page} score={c.score}\n"
                f"excerpt: {c.excerpt}\n"
            )
    else:
        prompt += "MANUAL_EVIDENCE_CHUNKS:\n<none>\n"

    prompt += "\nWEB_SEARCH_STATUS:\n"
    prompt += f"- web_used: {web_used}\n"
    prompt += f"- web_query: {web_query}\n"
    if web_error:
        prompt += f"- web_error: {web_error}\n"
    if web_results:
        prompt += "WEB_RESULTS:\n"
        for idx, r in enumerate(web_results[:5], start=1):
            sn = (r.snippet or "").replace("\n", " ").strip()
            sn = sn[:260] + ("…" if len(sn) > 260 else "")
            prompt += f"\n[web:{idx}] {r.title}\nurl: {r.url}\nsnippet: {sn}\n"
    else:
        prompt += "WEB_RESULTS:\n<none>\n"

    writer = _build_plain_llm(llm_model, ollama_host, sys, temperature=0.2)
    ans = _call_agent_text(writer, prompt)
    return ans.strip() if ans else "⚠️ No response from LLM. Please verify your Ollama server/model is running."


# ---------------------------
# Main pipeline
# ---------------------------

def run_agentic_guidance(
    incident_card: Dict[str, Any],
    question: Optional[str] = None,
    *,
    llm_model: Optional[str] = None,
    ollama_host: Optional[str] = None,
    k: int = 5,
    n_queries: int = 2,
    enable_web_search: bool = False,
    web_max_results: int = 5,
) -> AgenticResult:
    """Pipeline used by Streamlit Agentic mode.

    - LLM generates 1-2 manuals retrieval queries.
    - Python runs CGM retrieval deterministically.
    - Optional: LLM may request ONE web search; Python executes it deterministically.
    - LLM writes the final response (always).
    """
    settings = AppSettings()
    model_id = llm_model or settings.llm_model
    host = ollama_host or settings.ollama_host

    queries, _raw_q = generate_queries_from_incident(
        incident_card=incident_card,
        question=question,
        llm_model=model_id,
        ollama_host=host,
        n_queries=n_queries,
    )

    
    chunks, ok_any, err = retrieve_manual_evidence(
        queries=queries,
        k=int(k),
        settings=settings,
    )

    # ReAct-style loop: if manuals evidence is insufficient, let the LLM propose an additional manuals query
    # and execute CGM retrieval again. Keep this bounded to avoid loops/timeouts.
    for _ in range(2):
        need_more, extra_queries, _raw_more = decide_more_manual_search(
            incident_card=incident_card,
            question=question,
            queries_so_far=queries,
            chunks_so_far=chunks,
            llm_model=model_id,
            ollama_host=host,
            max_new_queries=1,
        )
        if not need_more or not extra_queries:
            break

        for q in extra_queries:
            if q in queries:
                continue
            queries.append(q)
            new_chunks, ok2, err2 = retrieve_manual_evidence(
                queries=[q],
                k=int(k),
                settings=settings,
            )
            if ok2:
                ok_any = True
            if err2:
                err = err2
            chunks = _merge_dedupe_chunks((chunks or []) + (new_chunks or []), k=int(k))

    web_used = False
    web_query = None
    web_results: List[WebResult] = []
    web_error: Optional[str] = None

    if bool(enable_web_search):
        web_used, web_query, _raw_web_decision = decide_web_search(
            incident_card=incident_card,
            question=question,
            queries=queries,
            chunks=chunks,
            llm_model=model_id,
            ollama_host=host,
        )
        if web_used and web_query:
            web_results, web_error = duckduckgo_search(web_query, max_results=int(web_max_results))
            if web_error:
                web_results = []
                # Keep web_used=True so the LLM can report that an external lookup was attempted but failed.
                web_used = True
        else:
            web_used = False
            web_query = None

    answer = generate_final_answer(
        incident_card=incident_card,
        question=question,
        queries=queries,
        chunks=chunks,
        llm_model=model_id,
        ollama_host=host,
        web_used=web_used,
        web_query=web_query,
        web_results=web_results,
        web_error=web_error,
    )

    return AgenticResult(
        queries=queries,
        chunks=chunks,
        answer=answer,
        retrieval_ok=ok_any,
        retrieval_error=err,
        web_used=web_used,
        web_query=web_query,
        web_results=web_results,
        web_error=web_error,
    )

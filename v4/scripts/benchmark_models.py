"""Benchmark des modèles LLM pour Sahteïn — PAR ÉTAGE.

Deux bancs distincts (le bon modèle pour chaque rôle du pipeline) :

  --stage redaction   : génération de la réponse (prose FR, abstention,
                        discipline des cartes). Utilise le VRAI SYSTEM_PROMPT +
                        schéma JSON de response_generator, sur un pack de
                        contexte figé (data/benchmark_context_pack.json).

  --stage structured  : query_understanding (compréhension : intent, ingrédients,
                        chefs, anaphores). Utilise le VRAI prompt + schéma de
                        query_understanding, sur data/benchmark_understanding_cases.json.

Tout passe par des endpoints compatibles OpenAI (base_url + clé en variable
d'environnement). Aucune clé n'est lue depuis le code ; rien n'est codé en dur.

Providers (clé attendue en env) :
  openai    OPENAI_API_KEY      https://api.openai.com/v1
  mistral   MISTRAL_API_KEY     https://api.mistral.ai/v1
  groq      GROQ_API_KEY        https://api.groq.com/openai/v1
  cerebras  CEREBRAS_API_KEY    https://api.cerebras.ai/v1
  deepseek  DEEPSEEK_API_KEY    https://api.deepseek.com
  qwen      DASHSCOPE_API_KEY   https://dashscope-intl.aliyuncs.com/compatible-mode/v1
  gemini    GEMINI_API_KEY      https://generativelanguage.googleapis.com/v1beta/openai/

Exemples (depuis v4/, PYTHONPATH=.) ::
  python scripts/benchmark_models.py --stage redaction
  python scripts/benchmark_models.py --stage structured --provider mistral
  python scripts/benchmark_models.py --provider groq --list-models
  python scripts/benchmark_models.py --stage redaction --provider cerebras --models qwen-3-235b-a22b
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.app.llm.query_understanding import JSON_SCHEMA as QU_SCHEMA  # noqa: E402
from backend.app.llm.query_understanding import SYSTEM_PROMPT as QU_SYS  # noqa: E402
from backend.app.llm.query_understanding import QueryPlan  # noqa: E402
from backend.app.llm.response_generator import JSON_SCHEMA as GEN_SCHEMA  # noqa: E402
from backend.app.llm.response_generator import SYSTEM_PROMPT as GEN_SYS  # noqa: E402
from backend.app.llm.response_generator import GroundedAnswer  # noqa: E402

PROVIDERS: dict[str, tuple[str, str]] = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "qwen": ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}

# Catalogues par défaut (IDs juin 2026, best-effort ; --list-models pour les vrais
# IDs d'un provider, --models pour forcer). Les erreurs d'ID sont capturées.
DEFAULT_MODELS: dict[str, list[str]] = {
    "openai": ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1", "gpt-5-mini", "gpt-5.4-mini", "gpt-5.4"],
    "mistral": ["mistral-large-latest", "mistral-small-latest"],
    "groq": ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "moonshotai/kimi-k2-instruct"],
    "cerebras": ["llama-3.3-70b", "qwen-3-235b-a22b", "gpt-oss-120b"],
    "deepseek": ["deepseek-chat", "deepseek-reasoner"],
    "qwen": ["qwen-max", "qwen-plus", "qwen-turbo"],
    "gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
    "openrouter": [
        "openai/gpt-4.1-mini", "mistralai/mistral-large", "mistralai/mistral-medium-3.5",
        "anthropic/claude-sonnet-4.6", "google/gemini-2.5-flash",
        "deepseek/deepseek-chat", "qwen/qwen-max",
    ],
}

# Tarifs USD / 1M tokens (in, out). gpt-4.1* : cost_tracker (repo). Autres :
# recherche juin 2026. None = inconnu -> coût non calculé (tokens seuls reportés).
PRICES: dict[str, tuple[float, float] | None] = {
    "gpt-4.1": (2.0, 8.0), "gpt-4.1-mini": (0.40, 1.60), "gpt-4.1-nano": (0.10, 0.40),
    "gpt-5-mini": (0.25, 2.0), "gpt-5-nano": (0.05, 0.40),
    "gpt-5.4": (2.5, 15.0), "gpt-5.4-mini": (0.50, 4.0), "gpt-5.4-nano": (0.10, 0.80),
    "gpt-5.5": (5.0, 30.0),
    "mistral-large-latest": (0.50, 1.50), "mistral-small-latest": (0.10, 0.30),
    "deepseek-chat": (0.14, 0.28), "deepseek-reasoner": (0.28, 0.42),
    "gemini-2.5-flash": (0.30, 2.50), "gemini-2.5-flash-lite": (0.10, 0.40),
    # OpenRouter slugs (best-effort juin 2026 ; vérifier sur openrouter.ai)
    "anthropic/claude-sonnet-4.6": (3.0, 15.0), "anthropic/claude-haiku-4.5": (1.0, 5.0),
    "anthropic/claude-opus-4.8": (5.0, 25.0),
    "mistralai/mistral-large": (0.50, 1.50), "mistralai/mistral-medium-3.1": (0.40, 2.0),
    "google/gemini-3.5-flash": (0.30, 2.50), "google/gemini-3.1-flash-lite": (0.10, 0.40),
    "deepseek/deepseek-v3.2": (0.14, 0.28), "deepseek/deepseek-chat": (0.14, 0.28),
    "qwen/qwen-plus": (0.40, 1.20), "moonshotai/kimi-k2.6": (0.60, 2.50),
    "openai/gpt-oss-120b": (0.10, 0.50),
    # Groq/Cerebras/Qwen direct : tarifs variables -> None (report tokens).
}


def _price(model: str) -> tuple[float, float] | None:
    if model in PRICES:
        return PRICES[model]
    for prefix, rate in PRICES.items():
        if model.startswith(prefix):
            return rate
    return None


def _cost_usd(model: str, pin: int, pout: int) -> float | None:
    rate = _price(model)
    return None if rate is None else round((pin * rate[0] + pout * rate[1]) / 1_000_000.0, 6)


def _make_client(provider: str):
    from openai import OpenAI

    base_url, env_var = PROVIDERS[provider]
    key = os.environ.get(env_var, "")
    if not key:
        raise RuntimeError(f"{env_var} absent — provider '{provider}' indisponible.")
    # timeout + max_retries : évite qu'un modèle qui bloque ne gèle tout le banc.
    return OpenAI(base_url=base_url, api_key=key, timeout=45.0, max_retries=1)


def _call(client, model: str, system: str, user: str, schema: dict[str, Any]) -> tuple[Any, float]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "response_format": {"type": "json_schema", "json_schema": schema},
    }
    t0 = time.time()
    try:
        resp = client.chat.completions.create(temperature=0.0, **kwargs)
    except Exception as exc:
        msg = str(exc).lower()
        if "temperature" in msg or "unsupported" in msg:
            resp = client.chat.completions.create(**kwargs)
        else:
            raise
    return resp, (time.time() - t0) * 1000.0


# ── Étage RÉDACTION ────────────────────────────────────────────────────────

def _format_chunks(chunks: list[dict[str, Any]]) -> str:
    lines = ["CONTEXTE — chaque entrée est un extrait d'article OLJ :"]
    for h in chunks:
        title = str(h.get("article_title") or "").replace("\n", " ")[:160]
        lines.append(
            f"\n[chunk_id={h['chunk_id']} | article={h['article_external_id']}"
            f" | titre={title} | section={h.get('section_kind')} | score=0.500]"
            f"\n{h.get('chunk_text')}"
        )
    return "\n".join(lines)


def _score_redaction(case: dict[str, Any], answer: GroundedAnswer) -> dict[str, Any]:
    a = case["assert"]
    provided = {c["chunk_id"] for c in case["chunks"]}
    text = " ".join(s.text for s in answer.answer_sentences)
    low = text.lower()
    cited: set[int] = set()
    for s in answer.answer_sentences:
        cited.update(s.source_chunk_ids)
    cards = int(answer.recipe_card is not None) + int(answer.recipe_card_secondary is not None)
    must = [m for m in a.get("must_contain", []) if m.lower() not in low]
    forbidden = [m for m in a.get("must_not_contain", []) if m.lower() in low]
    any_list = a.get("any_contains", [])
    any_ok = (not any_list) or any(m.lower() in low for m in any_list)
    cards_ok = cards <= a.get("max_recipe_cards", 99)
    req_recipe_ok = (answer.recipe_card is not None) if a.get("require_recipe_card") else True
    req_chef_ok = (answer.chef_card is not None) if a.get("require_chef_card") else True
    abstain_ok = (cards == 0 and answer.chef_card is None and any_ok) if a.get("must_abstain") else True
    grounding_ok = cited.issubset(provided)
    passed = grounding_ok and not must and not forbidden and any_ok and cards_ok and req_recipe_ok and req_chef_ok and abstain_ok
    return {
        "passed": passed, "grounding_ok": grounding_ok, "cards": cards,
        "missing_must": must, "forbidden_found": forbidden,
        "first_sentence": (answer.answer_sentences[0].text if answer.answer_sentences else ""),
        "confidence": round(float(answer.confidence or 0.0), 2),
    }


# ── Étage STRUCTURÉ (query_understanding) ──────────────────────────────────

def _structured_user(case: dict[str, Any]) -> str:
    q = case["query"]
    h = (case.get("history") or "").strip()
    if not h:
        return q
    return (
        "HISTORIQUE DE CONVERSATION (tours précédents, ordre chronologique ; "
        "résous les anaphores, suis le fil, note les relances) :\n"
        f"{h}\n\nQUESTION ACTUELLE (à interpréter à la lumière de l'historique) :\n{q}"
    )


def _present(expected: str, slugs: list[str]) -> bool:
    return any(expected == s or expected in s for s in slugs)


def _score_structured(case: dict[str, Any], plan: QueryPlan) -> dict[str, Any]:
    e = case["expect"]
    intent_ok = plan.intent in e.get("intents", [plan.intent])
    if e.get("ingredients_empty"):
        ing_ok = plan.ingredient_slugs == []
    elif e.get("ingredients"):
        ing_ok = all(_present(x, plan.ingredient_slugs) for x in e["ingredients"])
    else:
        ing_ok = True
    chef_ok = all(_present(x, plan.chef_slugs) for x in e.get("chefs", []))
    rw = plan.rewritten_query.lower()
    rw_ok = (not e.get("rewritten_any")) or any(s.lower() in rw for s in e["rewritten_any"])
    return {
        "passed": intent_ok and ing_ok and chef_ok and rw_ok,
        "intent_ok": intent_ok, "ingredients_ok": ing_ok, "chefs_ok": chef_ok, "rewritten_ok": rw_ok,
        "got": {"intent": plan.intent, "ingredients": plan.ingredient_slugs, "chefs": plan.chef_slugs},
    }


# ── Orchestration ──────────────────────────────────────────────────────────

def run(
    stage: str, provider: str, models: list[str], cases: list[dict[str, Any]],
    out_path: str | None = None,
) -> dict[str, Any]:
    client = _make_client(provider)
    if stage == "redaction":
        system, schema = GEN_SYS, GEN_SCHEMA
    else:
        system, schema = QU_SYS, QU_SCHEMA
    report: dict[str, Any] = {"stage": stage, "provider": provider, "models": {}}
    for model in models:
        rows: list[dict[str, Any]] = []
        tot_lat = 0.0
        tot_cost = 0.0
        cost_known = True
        n_ok = 0
        for case in cases:
            user = _format_redaction_user(case) if stage == "redaction" else _structured_user(case)
            row: dict[str, Any] = {"id": case["id"]}
            try:
                resp, lat = _call(client, model, system, user, schema)
                tot_lat += lat
                row["latency_ms"] = int(lat)
                u = resp.usage
                pin = int(getattr(u, "prompt_tokens", 0) or 0)
                pout = int(getattr(u, "completion_tokens", 0) or 0)
                row["tokens"] = {"in": pin, "out": pout}
                c = _cost_usd(model, pin, pout)
                if c is None:
                    cost_known = False
                else:
                    tot_cost += c
                raw = resp.choices[0].message.content or "{}"
                if stage == "redaction":
                    row.update(_score_redaction(case, GroundedAnswer.model_validate_json(raw)))
                else:
                    row.update(_score_structured(case, QueryPlan.model_validate_json(raw)))
                n_ok += 1
                row["schema_ok"] = True
            except Exception as exc:
                row.update({"schema_ok": False, "passed": False, "error": f"{type(exc).__name__}: {str(exc)[:160]}"})
            rows.append(row)
        n = len(cases)
        report["models"][model] = {
            "n_cases": n,
            "n_passed": sum(1 for r in rows if r.get("passed")),
            "schema_valid_rate": round(n_ok / n, 2) if n else 0.0,
            "avg_latency_ms": int(tot_lat / max(1, n_ok)) if n_ok else None,
            "total_usd": round(tot_cost, 5) if cost_known else None,
            "price_in_out_per_m": _price(model),
            "rows": rows,
        }
        # Sauvegarde incrémentale : un timeout ne fait plus perdre les modèles déjà faits.
        if out_path:
            Path(out_path).write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        mm = report["models"][model]
        print(
            f"[done] {model}: {mm['n_passed']}/{mm['n_cases']} "
            f"schema={int(mm['schema_valid_rate']*100)}% "
            f"lat={mm['avg_latency_ms']}ms cost={mm['total_usd']}",
            flush=True,
        )
    return report


def _format_redaction_user(case: dict[str, Any]) -> str:
    return f"QUESTION : {case['query']}\n\n{_format_chunks(case['chunks'])}"


def _ascii_table(report: dict[str, Any]) -> str:
    out = [f"STAGE={report['stage']}  PROVIDER={report['provider']}", ""]
    out.append(f"{'model':28} {'pass':>7} {'schema':>7} {'lat(ms)':>8} {'cost$':>9}")
    for model, m in report["models"].items():
        cost = m["total_usd"] if m["total_usd"] is not None else "n/a"
        lat = m["avg_latency_ms"] if m["avg_latency_ms"] is not None else "-"
        out.append(
            f"{model:28} {str(m['n_passed'])+'/'+str(m['n_cases']):>7} "
            f"{str(int(m['schema_valid_rate']*100))+'%':>7} {lat!s:>8} {cost!s:>9}"
        )
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark modèles LLM Sahteïn (par étage).")
    p.add_argument("--stage", default="redaction", choices=["redaction", "structured"])
    p.add_argument("--provider", default="openai", choices=list(PROVIDERS))
    p.add_argument("--models", default="", help="CSV ; défaut = catalogue du provider")
    p.add_argument("--list-models", action="store_true", help="liste les IDs dispo du provider et quitte")
    p.add_argument("--out", default="", help="JSON de sortie (défaut: data/benchmark_<stage>_<provider>.json)")
    args = p.parse_args()

    if args.list_models:
        client = _make_client(args.provider)
        ids = sorted(m.id for m in client.models.list().data)
        print(f"{args.provider}: {len(ids)} modèles")
        for i in ids:
            print(" ", i)
        return

    pack_path = (
        _ROOT / "data" / ("benchmark_context_pack.json" if args.stage == "redaction"
                          else "benchmark_understanding_cases.json")
    )
    cases = json.loads(pack_path.read_text(encoding="utf-8"))["cases"]
    models = [m.strip() for m in args.models.split(",") if m.strip()] or DEFAULT_MODELS[args.provider]
    out = args.out or str(_ROOT / "data" / f"benchmark_{args.stage}_{args.provider}.json")
    report = run(args.stage, args.provider, models, cases, out_path=out)
    # Markdown (UTF-8, fichier) + table ASCII (console Windows-safe)
    Path(out).with_suffix(".md").write_text(
        f"# Benchmark {args.stage} / {args.provider}\n\n```\n{_ascii_table(report)}\n```\n",
        encoding="utf-8",
    )
    print(_ascii_table(report))
    print(f"\n(JSON: {out})")


if __name__ == "__main__":
    main()

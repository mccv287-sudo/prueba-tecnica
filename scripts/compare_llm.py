"""Compara el LLM local (Ollama) con Claude en la desambiguación de AIDA test.

Las mismas 557 menciones y las mismas señales; solo cambia quién resuelve las
menciones inciertas. Cada ejecución guarda sus aciertos en ``results/``; cuando
existen los dos, compara Claude frente al local (IC 95 % por *bootstrap*
pareado de documentos).

Uso:
  uv run python scripts/compare_llm.py local
  uv run python scripts/compare_llm.py claude  # ANTHROPIC_API_KEY en el entorno o en .env
"""

import asyncio
import json
import sys

from entity_linker.config import RESULTS_DIR
from entity_linker.disambiguation.disambiguator import Disambiguator
from entity_linker.disambiguation.llm_claude import ClaudeChooser
from entity_linker.evaluation.datasets import load_disambiguation_sample
from entity_linker.evaluation.disambiguation import evaluate_disambiguation
from entity_linker.evaluation.metrics import compare
from entity_linker.wikidata import WikidataService

PROVIDERS = ("local", "claude")


async def main(provider: str) -> None:
    rows = load_disambiguation_sample("test", "aida")
    wikidata = WikidataService()
    chooser = ClaudeChooser() if provider == "claude" else None  # None: Ollama
    disambiguator = Disambiguator(wikidata, llm_chooser=chooser)
    report = {
        "sin LLM": await evaluate_disambiguation(rows, disambiguator, use_llm=False),
        provider: await evaluate_disambiguation(rows, disambiguator, use_llm=True),
    }
    await wikidata.aclose()

    for name, result in report.items():
        print(
            f"{name:8} accuracy {result['accuracy']} IC {result['ci95']} "
            f"| {result['llm_calls']} llamadas, {result['seconds']} s"
        )
    gain = compare(report["sin LLM"]["hits_by_doc"], report[provider]["hits_by_doc"])
    print(f"{provider} vs sin LLM:", gain)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"llm_{provider}_aida_test.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    paths = {p: RESULTS_DIR / f"llm_{p}_aida_test.json" for p in PROVIDERS}
    if all(p.exists() for p in paths.values()):
        hits = {
            p: json.loads(path.read_text())[p]["hits_by_doc"]
            for p, path in paths.items()
        }
        print("claude vs local:", compare(hits["local"], hits["claude"]))


if __name__ == "__main__":
    if sys.argv[1:] not in ([p] for p in PROVIDERS):
        raise SystemExit(f"Uso: compare_llm.py {{{'|'.join(PROVIDERS)}}}")
    asyncio.run(main(sys.argv[1]))

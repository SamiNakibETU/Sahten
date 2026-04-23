# Évaluation RAG — jeu d'or (golden set)

Fichier : `data/golden_eval_fr.json` (requêtes + IDs d’articles attendus + sous-chaînes à retrouver dans la réponse générée).

## Exécution locale

Prérequis : variables d’environnement (`.env` ou env) identiques à Railway — au minimum `DATABASE_URL` / accès Postgres, `OPENAI_API_KEY`, clés Cohere si rerank, etc.

```bash
cd v4
.venv/Scripts/activate   # Windows
set PYTHONPATH=backend
python scripts/run_rag_eval.py --golden data/golden_eval_fr.json --top-k 12
```

Code de sortie : `0` si tous les items passent, `1` sinon. La sortie JSON résume chaque requête (IDs en tête de retrieval, mots manquants, confiance, timings).

## RAGAS (optionnel)

Le projet expose le groupe de dépendances `eval` (`ragas`, `datasets`) pour des métriques plus riches (foi au contexte, etc.). Le script `run_rag_eval` reste volontairement autonome pour la CI et les runs sans installation lourde.

## Ajuster le jeu d'or

- Augmenter `expected_article_external_ids` lorsque l’on ajoute de nouveaux articles de référence dans la base.
- `answer_must_contain` : fragments **en minuscules comparés** au texte de la réponse (insensible à la casse) ; évitez les phrases trop longues (variations de formulation LLM).

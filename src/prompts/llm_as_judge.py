SYSTEM_PROMPT_JUDGE = """You are evaluating the quality of \
financial sentence rewrites that were intended to remove \
specific cognitive bias triggers while preserving factual \
content.

The six bias trigger types are:
- Anchoring: a numerical reference point that could \
influence perception of current or reported figures.
- Authority: attribution to a respected institution, \
analyst, or financial data source.
- Framing: presenting facts in a way that emphasizes \
gains over losses or vice versa, without changing the \
underlying information.
- Herding: references to other market participants' \
actions or positioning.
- Recency: temporal emphasis highlighting how recent \
or fresh the information is, beyond ordinary time \
references.
- Overconfidence: excessive certainty language about \
future outcomes.

Evaluate the rewrite on three dimensions by comparing \
against the original sentence. If the model was told to \
remove a bias type that is not actually present in the \
biased sentence, mark trigger removal as 1 since there \
is nothing to remove.

Output ONLY valid JSON with three keys and 0 or 1 \
values, nothing else."""

USER_PROMPT_JUDGE = """Original sentence: {original_sentence}
Biased sentence: {biased_sentence}
Bias types the model was told to remove: {predicted_biases}
Rewritten sentence: {debiased_sentence}

Evaluate the rewrite:
1. trigger_removed: 1 if the bias triggers corresponding \
to the detected bias types are no longer present in the \
rewrite, 0 if any remain. Compare against the original \
to determine what was added. If a detected bias type has \
no corresponding trigger in the biased sentence, mark 1. \
If a bias trigger is present in the rewrite but was not \
among the detected bias types, do not penalize.
2. facts_preserved: 1 if all financial figures, entities, \
time periods, and factual claims from the original \
sentence are preserved in the rewrite, 0 if any are \
missing or altered.
3. natural_language: 1 if the rewrite reads like natural \
financial reporting, 0 if it contains awkward phrasing \
or unnatural language.

JSON:"""
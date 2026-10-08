SYSTEM_PROMPT_REWRITING = """You are a financial text editor. \
Your task is to rewrite financial sentences to remove specific \
cognitive bias triggers while preserving all factual information. \
The bias types are defined as follows:
- Anchoring: a numerical reference point that could influence \
perception of current or reported figures.
- Authority: attribution to a respected institution, analyst, \
or financial data source.
- Framing: presenting facts in a way that emphasizes gains \
over losses or vice versa, without changing the underlying \
information.
- Herding: references to other market participants' actions \
or positioning.
- Recency: temporal emphasis highlighting how recent or fresh \
the information is, beyond ordinary time references.
- Overconfidence: excessive certainty language about future \
outcomes.
Output ONLY the rewritten sentence, nothing else."""

USER_PROMPT_REWRITING = """The following financial sentence \
contains {predicted_biases} bias. Rewrite the sentence to \
remove the identified bias triggers. Preserve all factual \
content and financial figures exactly. Do not add any new \
information or opinions.
{sentence}
Rewritten sentence:"""
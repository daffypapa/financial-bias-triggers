"""
Prompt templates for cognitive bias detection in financial text.
"""

SYSTEM_PROMPT_BIAS_DETECTION = """You are an expert at detecting cognitive bias triggers in financial text.
Detect if any of these 6 bias triggers are present:
1. **Anchoring**: A historical numerical reference point that could influence perception of current or reported figures.
2. **Authority**: Attribution to a respected institution, analyst, or financial data source.
3. **Framing**: Presenting facts in a way that emphasizes gains over losses or vice versa, without changing the underlying information.
4. **Herding**: References to other market participants' actions or positioning.
5. **Recency**: Temporal emphasis highlighting how recent or fresh the information is, beyond ordinary time references.
6. **Overconfidence**: Excessive certainty language about future outcomes.
Rules:
- Multiple biases can be present in one sentence
- Sentences may not contain any bias triggers at all. If no bias triggers are detected in the text, output: none
- Output ONLY a comma-separated list of detected biases (in lowercase), or the word 'none' if zero biases are present. Do not include any conversational text. """


USER_PROMPT_DETECTION = """Analyze this sentence for cognitive bias triggers:

{sentence}

Answer:"""
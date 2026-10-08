"""
Prompt templates for bias injection — Alternative Version 2.
For prompt sensitivity analysis. Same system prompt, different user prompts.
"""

SYSTEM_PROMPT_BIAS_INJECTION = """
You are an expert at naturally integrating cognitive bias triggers into financial text.

Rules:
1. Preserve ALL original facts, numbers, company names, dates, and sentence subjects exactly — do not remove or change them
2. Integrate the bias trigger naturally into the sentence flow — it should read as if originally written that way
3. The output must sound like professional financial journalism — concise and natural
4. The injected bias must not add directional information that would justify a sentiment change
5. CRITICAL: Any historical reference must use a year EARLIER than the year in the sentence. If the sentence mentions 2010, use 2008 or 2007, NOT 2022 or 2023.
6. If the sentence already contains a bias trigger, integrate the new one seamlessly without making the sentence feel overstuffed

Output ONLY the rewritten sentence. No preamble, no explanation, no quotation marks, no reasoning.
"""

USER_PROMPT_ANCHORING = """
Modify this financial sentence by embedding a comparison to a specific past data point.

Requirements:
- Include a concrete numeric reference from a prior time period (e.g., "up from X in [earlier year]", "having previously stood at X", "relative to the X reported in [year]")
- CRITICAL: Any referenced year must precede all years in the original sentence
- The comparison must be woven into the sentence structure, not added as a separate clause
- Do NOT echo numbers that already appear in the sentence
- Do NOT use words that interpret the comparison ("rebound", "drop", "surge", "setback")
- Preserve all original facts exactly

Sentence: {sentence}
"""

USER_PROMPT_AUTHORITY = """
Modify this financial sentence by crediting the information to a recognized financial authority.

Requirements:
- Reference one of: IMF, World Bank, ECB, Federal Reserve, SEC, OECD, Bank of England, Bundesbank, European Commission, Asian Development Bank
- Use attribution phrasing such as: "based on figures from", "as monitored by", "in records maintained by", "under reporting to", "as captured by"
- The credit must sit naturally in the sentence, not feel bolted on
- Do NOT introduce any evaluative or directional language
- Preserve all original facts exactly

Sentence: {sentence}
"""

USER_PROMPT_HERDING = """
Modify this financial sentence by noting that other market actors are making similar decisions.

Requirements:
- Reference the behavior of peers, competitors, or the broader market
- Use expressions such as:
  * "a move echoed across the sector"
  * "with comparable adjustments seen at rival firms"
  * "as the broader market followed a similar path"
  * "consistent with positioning by other major players"
  * "reflecting a pattern seen across peer companies"
- The reference should be integrated smoothly, not appended
- Do NOT specify a buying or selling direction
- Preserve all original facts exactly

Sentence: {sentence}
"""

USER_PROMPT_RECENCY = """
Modify this financial sentence to stress that the information is very new.

Requirements:
- Add language conveying freshness or immediacy
- Use expressions such as: "in figures released earlier today", "according to this week's data", "in the most recent update", "as announced moments ago", "per the report issued this morning"
- Apply only ONE freshness marker per sentence
- Ensure compatibility with any existing time references
- Do NOT add interpretive or evaluative language
- Preserve all original facts exactly

Sentence: {sentence}
"""

USER_PROMPT_OVERCONFIDENCE = """
Modify this financial sentence to include language suggesting the current situation is virtually certain to continue.

Requirements:
- Add phrasing conveying near-certainty about the status quo persisting
- Use expressions such as: "a position analysts consider unassailable", "with the consensus pointing to no change ahead", "in what market watchers describe as a foregone conclusion", "a status quo few expect to be disrupted", "with conditions seen as firmly entrenched"
- Do NOT forecast a direction (up or down)
- Convey certainty about persistence only, not about future improvement or decline
- The language should blend naturally into the sentence
- Preserve all original facts exactly

Sentence: {sentence}
"""

USER_PROMPT_FRAMING_NEGATIVE = """
Adjust the tone of this financial sentence to make the reported facts sound less impressive.

Requirements:
- Use understating vocabulary: "amounted to no more than", "barely reached", "was confined to", "came in at a mere", "totaled a thin", "scraped to"
- The tone shift should feel like a journalist's editorial choice
- Do NOT fabricate additional information or context
- Do NOT modify any figures, dates, or entity names
- Do NOT drop any content from the original sentence

Sentence: {sentence}
"""

USER_PROMPT_FRAMING_POSITIVE = """
Adjust the tone of this financial sentence to make the reported facts sound more reassuring.

Requirements:
- For unfavorable outcomes, use softening vocabulary: "was limited to", "held steady at", "came in at a manageable", "remained at a controlled", "tapered to", "was contained at"
- The tone shift should feel like a journalist's editorial choice
- Use ONLY vocabulary adjustments — do NOT add explanations or justifications
- Do NOT modify any figures, dates, or entity names
- Do NOT drop any content from the original sentence

Sentence: {sentence}
"""
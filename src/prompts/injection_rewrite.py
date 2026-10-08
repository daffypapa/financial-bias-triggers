"""
Prompt templates for bias injection — natural rewrite version.
Bias is integrated into the sentence structure, not just appended.
Final version for dataset generation.
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
Rewrite this sentence to naturally include ANCHORING bias via a historical numerical reference.

Requirements:
- Weave in a concrete historical number (past price, previous earnings, old metric)
- CRITICAL: The anchor year must be EARLIER than any year in the sentence
- The anchor should feel like a natural part of the sentence, not tacked on
- Do NOT duplicate numbers already present in the sentence
- Do NOT add interpretive language ("recovery", "turnaround", "improvement", "deterioration")
- Do NOT always use "compared with X in [year]" — vary your phrasing
- Preserve all original facts exactly

Sentence: {sentence}
"""

USER_PROMPT_AUTHORITY = """
Rewrite this sentence to naturally include AUTHORITY bias via an institutional reference.

Requirements:
- Weave in a reference to: Goldman Sachs, JP Morgan, Morgan Stanley, UBS, Barclays, Moody's, S&P, Fitch, Bloomberg, Reuters, or Financial Times
- Do NOT use: Deloitte, PwC, KPMG, EY, McKinsey, BCG
- Use direct attribution only: "according to", "cited by", "data from", "per", "as reported by"
- Do NOT use meta-referential phrasing like "format commonly used by" or "in a report reviewed by"
- The attribution should feel like a natural part of the sentence, not tacked on
- Do NOT add directional information ("upgraded", "bullish")
- Preserve all original facts exactly

Sentence: {sentence}
"""

USER_PROMPT_HERDING = """
Rewrite this sentence to naturally include HERDING bias via a reference to market participants.

Requirements:
- Weave in a reference to other investors or institutions acting
- Vary your phrasing — use different expressions:
  * "amid broader institutional repositioning"
  * "as market participants adjusted exposure"
  * "with peers making similar moves"
  * "as other funds shifted positions"
  * "following wider portfolio adjustments in the sector"
- The reference should feel like a natural part of the sentence, not tacked on
- Do NOT use directional language: no "buying", "selling", "shorting", "accumulating"
- Preserve all original facts exactly

Sentence: {sentence}
"""

USER_PROMPT_RECENCY = """
Rewrite this sentence to naturally include RECENCY bias via temporal emphasis.

Requirements:
- Weave in language emphasizing how recent the information is
- Use only ONE recency marker — do not double up
- Do NOT combine awkwardly with existing temporal phrases in the sentence
- Do NOT add interpretive words like "improvement", "deterioration", "recovery"
- The emphasis should feel like a natural part of the sentence, not tacked on
- Preserve all original facts exactly

Sentence: {sentence}
"""

USER_PROMPT_OVERCONFIDENCE = """
Rewrite this sentence to naturally include OVERCONFIDENCE bias via certainty language.

Requirements:
- Weave in excessive certainty about future outcomes
- Vary your phrasing — do NOT always use "certain to continue" or "will certainly continue"
- Options: "all but guaranteed to persist", "expected to hold", "set to remain", "poised to continue", "showing no signs of changing"
- Do NOT use "undoubtedly" — this is forbidden
- Do NOT use "with little room for doubt" — this is forbidden
- Do NOT add directional predictions ("further growth", "further declines", "upward momentum")
- Express certainty about continuation or persistence, not direction
- The certainty should feel like a natural part of the sentence, not tacked on
- Preserve all original facts exactly

Sentence: {sentence}
"""

USER_PROMPT_FRAMING_NEGATIVE = """
Rewrite this sentence with a NEGATIVE spin while preserving all facts exactly.

Requirements:
- Integrate negative tone naturally using varied vocabulary: "only", "merely", "just", "modest", "slim", "disappointing", "lackluster", "simply"
- Do NOT overuse "only" — vary your word choice
- The spin should feel like the original author's perspective, not forced
- Do NOT add new judgments or context
- Do NOT change any numbers or facts
- Do NOT remove the sentence subject or any entities

Sentence: {sentence}
"""

USER_PROMPT_FRAMING_POSITIVE = """
Rewrite this sentence with a POSITIVE spin while preserving all facts exactly.

Requirements:
- For negative facts, naturally minimize magnitude or reframe direction
- Use varied vocabulary: "solid", "steady", "contained", "limited", "manageable", "held at", "stabilized at"
- Do NOT overuse "modest" — vary your word choice
- The spin should feel like the original author's perspective, not forced
- Use ONLY tone words — no added context or rationalizations
- Do NOT add phrases like "remaining resilient", "amid challenges", "despite headwinds"
- Do NOT change any numbers or facts
- Do NOT remove the sentence subject or any entities

Sentence: {sentence}
"""
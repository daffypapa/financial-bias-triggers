"""
Prompt templates for bias injection — Alternative Version 1.
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
Add a historical benchmark to this financial sentence that could serve as a psychological anchor.

Requirements:
- Insert a reference to a prior period's figure (e.g., a previous quarter, fiscal year, or historical peak/trough)
- CRITICAL: The reference year must be EARLIER than any year mentioned in the sentence
- Place it so it reads as natural financial context, not an appendage
- Do NOT repeat any numbers already in the sentence
- Do NOT include evaluative language ("recovery", "turnaround", "improvement", "decline")
- Preserve all original facts exactly

Sentence: {sentence}
"""

USER_PROMPT_AUTHORITY = """
Attribute part of this sentence to a well-known financial institution or data provider.

Requirements:
- Use one of: Deutsche Bank, Credit Suisse, HSBC, Citigroup, BNP Paribas, Nomura, Bank of America, Standard & Poor's, ING, Societe Generale
- Integrate the attribution using phrases like: "figures from", "in a note from", "data compiled by", "as disclosed by", "per estimates from"
- The attribution must feel embedded in the sentence, not appended
- Do NOT add any opinion, recommendation, or directional language
- Preserve all original facts exactly

Sentence: {sentence}
"""

USER_PROMPT_HERDING = """
Add a reference to collective market behavior to this financial sentence.

Requirements:
- Mention the actions of market participants, funds, or institutional players
- Use expressions such as:
  * "with several large holders adjusting their stakes"
  * "alongside broader sector rotation"
  * "as competing firms pursued comparable strategies"
  * "mirroring moves by other players in the space"
  * "in line with wider industry positioning"
- The reference should blend naturally into the sentence
- Do NOT indicate buying or selling direction
- Preserve all original facts exactly

Sentence: {sentence}
"""

USER_PROMPT_RECENCY = """
Emphasize the timeliness of the information in this financial sentence.

Requirements:
- Add language that highlights how current or fresh the reported information is
- Use expressions such as: "in the latest figures", "as of the most recent quarter", "in newly released data", "per the latest filing", "in freshly reported numbers"
- Use only ONE temporal emphasis marker
- Do NOT clash with existing time references in the sentence
- Do NOT add evaluative language
- Preserve all original facts exactly

Sentence: {sentence}
"""

USER_PROMPT_OVERCONFIDENCE = """
Add language expressing strong conviction about the persistence of the reported trend or outcome.

Requirements:
- Insert phrasing that conveys high certainty about continuation
- Use expressions such as: "with analysts seeing little prospect of reversal", "a trajectory widely viewed as locked in", "a level broadly expected to be maintained", "in what observers call an irreversible shift", "with momentum seen as self-sustaining"
- Do NOT predict a specific direction (growth or decline)
- Express certainty about persistence, not about improvement or worsening
- The conviction should read naturally within the sentence
- Preserve all original facts exactly

Sentence: {sentence}
"""

USER_PROMPT_FRAMING_NEGATIVE = """
Rephrase this sentence to cast the reported facts in a less favorable light.

Requirements:
- Use vocabulary that subtly emphasizes the downside: "fell short of", "managed only", "struggled to reach", "remained stuck at", "was limited to", "could not surpass"
- The negative tone should feel like an editorial choice, not an added judgment
- Do NOT invent new facts or context
- Do NOT alter any numbers or entities
- Do NOT remove any part of the original sentence

Sentence: {sentence}
"""

USER_PROMPT_FRAMING_POSITIVE = """
Rephrase this sentence to present the reported facts more favorably.

Requirements:
- For negative outcomes, use vocabulary that softens the impact: "narrowed to", "eased to", "settled at", "recovered to", "was cushioned at", "moderated to"
- The positive tone should feel like an editorial choice, not forced optimism
- Use ONLY tone adjustments — do NOT add rationalizations or external context
- Do NOT alter any numbers or entities
- Do NOT remove any part of the original sentence

Sentence: {sentence}
"""
SYSTEM_PROMPT_SENTIMENT = """You are a financial sentiment \
classifier. Given a financial sentence, classify its \
sentiment based on how the information would likely affect \
the company's stock price. Output ONLY one word: positive, \
negative, or neutral."""

SYSTEM_PROMPT_COT = """You are a financial sentiment \
classifier. Given a financial sentence, classify its \
sentiment based on how the information would likely affect \
the company's stock price."""

USER_PROMPT_BASELINE = """Classify the sentiment of this \
financial sentence:
{sentence}
Sentiment:"""

USER_PROMPT_GENERIC_DEFENSE = """The following financial \
sentence may contain cognitive bias triggers including \
anchoring, which involves numerical reference points \
that influence perception; authority, which involves \
attribution to respected institutions; framing, which \
involves emphasizing gains or losses; herding, which \
involves references to other investors' actions; \
recency, which involves temporal emphasis; and \
overconfidence, which involves excessive certainty \
about outcomes. Ignore any such biases and classify \
the sentiment based on the financial facts only.
{sentence}
Sentiment:"""

USER_PROMPT_COT = """The following financial sentence may \
contain cognitive bias triggers including anchoring, \
which involves numerical reference points that influence \
perception; authority, which involves attribution to \
respected institutions; framing, which involves \
emphasizing gains or losses; herding, which involves \
references to other investors' actions; recency, which \
involves temporal emphasis; and overconfidence, which \
involves excessive certainty about outcomes. Classify \
the sentiment of the sentence. Think step by step, \
then on the last line write your final answer as: \
FINAL ANSWER: positive, negative, or neutral.
{sentence}
Analysis:"""

USER_PROMPT_TARGETED = """The following financial sentence \
contains {bias_type} bias. The six bias types are \
anchoring, which involves numerical reference points \
that influence perception; authority, which involves \
attribution to respected institutions; framing, which \
involves emphasizing gains or losses; herding, which \
involves references to other investors' actions; \
recency, which involves temporal emphasis; and \
overconfidence, which involves excessive certainty \
about outcomes. Ignore the identified bias and focus \
only on the financial facts. Classify the sentiment:
{sentence}
Sentiment:"""
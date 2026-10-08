# Bias Before the Model: Detecting and Removing Cognitive Bias Triggers in Financial Text

Code and data for our EMNLP 2026 paper.

We inject six types of cognitive bias triggers (anchoring, authority, framing, herding, recency, overconfidence) into financial sentences from the Financial PhraseBank, and study three questions:

- **RQ1 (Detection):** Can LLMs detect which bias triggers a sentence contains?
- **RQ2 (Removal):** Can LLMs rewrite a sentence to remove the triggers while preserving its facts?
- **RQ3 (Defense):** How do bias triggers affect LLM sentiment classification, and which defenses help?

## Repository structure

```
data/            Released datasets (see data/LICENSE)
src/
  core/          Model wrappers (HuggingFace and OpenAI/Anthropic/Gemini APIs) and inference loop
  prompts/       All prompts used in the paper
  experiments/   One script per step: bias_injection, bias_detection, bias_rewriting, bias_judge, bias_defense
bash_scripts/    Scripts that run each step for all models in the paper
notebooks/       Data preparation and analysis, grouped by research question
```

All experiment outputs are written to `outputs/`, which is created on first run.

## Setup

```bash
conda env create -f environment.yaml
conda activate fin_sent_biases
```

The open-weight models run locally through HuggingFace Transformers and need a CUDA GPU. The 70B models are loaded in 4-bit and still need substantial GPU memory.

The closed-source models (GPT, Claude, Gemini) need API keys. Create a `.env` file in the repository root with the keys for the providers you run:

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
```

Keys already exported in your shell take precedence over `.env`.

## Data

| File | Description |
|---|---|
| `data/base_sentences.csv` | 150 sentences from Financial PhraseBank (AllAgree), selected by hand as objective statements |
| `data/triggerbank.csv` | The final dataset (3,006 rows) |
| `data/prompt_sensitivity/sample_30.csv` | Stratified sample of 30 base sentences |
| `data/prompt_sensitivity/biased_sample_alt_{1,2}.csv` | The sample injected with two alternative injection prompts |

`triggerbank.csv` has four columns:

- `original_sentence`: the base sentence
- `biased_sentence`: the sentence with bias triggers injected (identical to `original_sentence` for unbiased rows)
- `label`: the Financial PhraseBank sentiment of the base sentence (`positive`, `negative`, `neutral`)
- `biases`: the injected bias types, comma-separated, or `none`

Each base sentence appears with every single bias and every pair of biases, plus once unbiased. Framing is applied only to non-neutral sentences, against their sentiment: positive sentences get a negative spin and vice versa. This gives 2,856 biased and 150 unbiased sentences.

Biases were injected with GPT-5.4 at temperature 1.0 (`src/prompts/injection_rewrite.py`), followed by the postprocessing in `notebooks/01_dataset_postprocessing.ipynb`.

## Running the experiments

Run the bash scripts from any directory. Each one loops over the models in `bash_scripts/models.sh`; comment out entries there to run a subset.

| Step | Script | Requires |
|---|---|---|
| Bias injection (optional) | `bash_scripts/00_bias_injection.sh` | |
| RQ1: detection | `bash_scripts/rq1_detection.sh` | |
| RQ2: rewriting | `bash_scripts/rq2_rewriting.sh` | RQ1 |
| RQ2: LLM-as-judge | `bash_scripts/rq2_judge.sh` | RQ2 rewriting |
| RQ3: defense | `bash_scripts/rq3_defense.sh` | RQ1, RQ2 rewriting |
| Prompt sensitivity | `bash_scripts/prompt_sensitivity.sh` | |

All RQ experiments use the released `data/triggerbank.csv`. Bias injection is sampled, so rerunning it produces a different dataset; it is included to document how the dataset was built.

RQ3 evaluates sentiment classification under these conditions:

| Condition | Input | Prompt |
|---|---|---|
| `1_clean_baseline` | unbiased sentence | plain classification |
| `2_no_defense` | biased sentence | plain classification |
| `3_generic_defense` | biased sentence | warned that bias triggers may be present |
| `4_cot` / `4_cot_clean` | biased / unbiased sentence | chain-of-thought |
| `5_targeted_detected` | biased sentence | told which biases the model detected in RQ1 |
| `6_targeted_gold` | biased sentence | told the gold biases |
| `7_rewrite_detected` | RQ2 rewrite using detected biases | plain classification |
| `8_rewrite_gold` | RQ2 rewrite using gold biases | plain classification |

Every script also runs on its own. See `python -m experiments.<name> --help` from inside `src/`.

## Notebooks

Run from inside `notebooks/`.

| Notebook | Purpose |
|---|---|
| `00_data_preparation.ipynb` | Downloads Financial PhraseBank and exports the AllAgree subset |
| `01_dataset_postprocessing.ipynb` | Builds the final dataset from the bias injection output |
| `rq1_detection_analysis.ipynb` | Detection F1 per bias type, single vs. pairwise, per combination, `none` class |
| `rq1_prompt_sensitivity.ipynb` | Detection F1 under alternative injection prompts |
| `rq2_rewriting_analysis.ipynb` | Judge scores, trigger removal and fact preservation per bias type, error samples |
| `rq3_finbert_baseline.ipynb` | FinBERT baseline for RQ3 |
| `rq3_defense_analysis.ipynb` | Accuracy per condition, pairwise and per-bias breakdowns, CoT format failures |

## Citation

```bibtex
@inproceedings{TODO,
  title     = {Bias Before the Model: Detecting and Removing Cognitive Bias Triggers in Financial Text},
  author    = {TODO},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```

## License

Code: MIT License (`LICENSE`).
Dataset: CC BY-NC-SA 4.0 (`data/LICENSE`). The data is derived from the Financial PhraseBank (Malo et al., 2014), which is licensed under CC BY-NC-SA 3.0.

## Contact

t.papadopoulos@athenarc.gr

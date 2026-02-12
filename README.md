# RL-based Text-to-SQL with GRPO

Reinforcement Learning integration for Text-to-SQL using Group Relative Policy Optimization (GRPO).

---

## Chosen Paper and Rationale
### Why SQL-R1?

I went with **SQL-R1** for the following reasons:

1. **Straightforward RL integration**: SQL-R1 uses GRPO (from DeepSeek-R1, arXiv:2501.12948) which is conceptually clean -- generate K candidates, execute them, normalize rewards within the group. No separate reward model to train (unlike Reward-SQL) and no graph parsing of SQL ASTs (unlike Graph-Reward-SQL). This made it easier to implement from scratch and debug.

2. **Multi-component reward design**: The composite reward (execution correctness + SQL validity + format compliance + partial match) gives a smoother gradient signal than binary correct/incorrect. Early in training most candidates are wrong -- the partial rewards help the model improve incrementally.

3. **Arctic-Text2SQL-R1** was considered, but its reward system is too strict—it only rewards or penalizes based on whether the SQL executes correctly, with no credit for partial correctness or formatting. Graph-Reward-SQL increases complexity by requiring SQL to be parsed into ASTs(Abstract Syntax Tree) for reward computation and similar case was with the reward SQL paper as it requires a  SQL-R1 is the most self-contained approach.
I think that the Graph reward SQL must be the most suitable approach given that it utilises the CTE through which we are training the model with more depth (step level) But as for this assignment main focus is on RL implementation so I have implemented SQL-R1 paper approach (Also I was having only T4 GPU so can test the code for ~14 GB VRAM memory only). Plus - I think I can explain this in more detail on a call : 

### Key implementation details

- **Base model**: Qwen2.5-Coder-3B-Instruct, loaded with 4-bit NF4 quantization (~2 GB VRAM)
- **LoRA adapters**: Applied to all linear layers (q/k/v/o projections + gate/up/down projections), rank=16, alpha=32. Only ~0.2% of parameters are trainable.
- **Reference policy**: Instead of loading a second model, I snapshot the initial LoRA weights and compute an L2-based KL approximation against them. This saves ~3 GB VRAM compared to keeping a full reference model.
- **Sequential generation**: Candidates are generated one at a time rather than batched to avoid CUDA OOM errors on consumer GPUs with quantized models.
- **Clipped surrogate objective**: Same as PPO -- importance ratio is clipped to [1-eps, 1+eps] to keep updates stable.

---

## Reward Design and Training Details

### Reward Components (from SQL-R1)

| Component | Reward | Condition |
|---|---|---|
| Execution correctness | +1.0 | Predicted SQL produces the same result set as gold SQL |
| Valid but wrong | +0.1 | SQL executes without error but gives wrong results |
| Invalid SQL | -0.5 | SQL fails to execute (syntax error, missing table, etc.) |
| Format compliance | +0.2 | Model wrapped output in ```sql``` code blocks |
| Partial match | up to +0.3 | Jaccard overlap of table/column names between predicted and gold SQL |

The total reward for a candidate is the sum of applicable components. For example:
- Correct SQL with proper formatting: +1.0 + 0.2 = **+1.2**
- Valid SQL, wrong result, good overlap: +0.1 + 0.2 + 0.15 = **+0.45**
- Invalid SQL with formatting: -0.5 + 0.2 = **-0.3**

### Training Hyperparameters

All hyperparameters are in `configs/grpo_config.yaml`:

| Parameter | Value | Notes |
|---|---|---|
| Learning rate | 5e-6 | AdamW optimizer |
| Weight decay | 0.01 | |
| Group size (K) | 4 | Candidates per question |
| Clip epsilon | 0.2 | PPO clipping range |
| KL coefficient | 0.05 | Penalty weight for policy drift |
| Temperature | 0.7 | For generation diversity |
| Max new tokens | 256 | Maximum SQL output length |
| Epochs | 2 | Over the Spider training set |
| Eval frequency | Every 50 steps | On Spider dev subset |

### GPU Memory Budget

| Component | VRAM |
|---|---|
| Base model (4-bit NF4) | ~2 GB |
| LoRA adapters | ~0.2 GB |
| Reference LoRA snapshot | ~0.2 GB |
| KV cache (sequential K=4) | ~2 GB |
| Gradients + optimizer states | ~6 GB |
| **Total** | **~10-12 GB** |

Tested on: Google Colab T4 (15 GB)

---

## Experimental Results

### Pipeline Verification (Dummy Data)

Before running on Spider, `test_pipeline.py` verifies the full loop on a synthetic database with 5 employees and 3 departments. This confirms that:
- Reward computation works correctly for all cases (correct, valid-but-wrong, invalid)
- Model can generate SQL from schema + question prompts
- GRPO training loop runs without errors (gradient flows through LoRA adapters)
- Post-training generation still works

## Project Structure

```
rl-text2sql/
│
├── src/                          # Core RL implementation
│   ├── __init__.py
│   ├── grpo_trainer.py           # GRPO algorithm (sampling, advantages, loss, update)
│   ├── reward.py                 # Composite reward function (execution + format + partial)
│   ├── sql_executor.py           # Safe SQL execution engine with timeout protection
│   ├── model_utils.py            # Model loading with QLoRA + LoRA + reference policy
│   └── data_utils.py             # Spider dataset loading and prompt formatting
│
├── scripts/                      # Training and evaluation entry points
│   ├── train_grpo.py             # Main GRPO training script (Spider dataset)
│   └── evaluate.py               # Evaluation on Spider dev set with detailed results
│
├── tests/                        # Pipeline verification
│   └── test_pipeline.py          # Tests full GRPO loop with synthetic dummy data
│
├── configs/                      # Hyperparameter configurations
│   └── grpo_config.yaml          # All training hyperparameters
│
├── notebooks/                    # Colab notebook
│   └── colab_grpo.ipynb          # Ready-to-run notebook for Google Colab T4
│
├── data/                         # Dataset directory (not in repo — download separately)
│   └── spider/
│       ├── train_spider.json
│       ├── dev.json
│       └── database/             # 200 SQLite databases
│
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Spider dataset

Download from [https://yale-lily.github.io/spider](https://yale-lily.github.io/spider) and place it as:

```
data/spider/
    train_spider.json      # ~7000 training examples
    dev.json               # ~1034 dev examples
    database/              # ~200 SQLite databases
        concert_singer/concert_singer.sqlite
        pets_1/pets_1.sqlite
        ...
```

### 3. Test the pipeline (no Spider needed)

```bash
python tests/test_pipeline.py
```

Creates a dummy database and runs 3 GRPO training steps to verify everything works end-to-end.

### 4. Train on Spider

```bash
# full training
python scripts/train_grpo.py --config configs/grpo_config.yaml

# quick run on 100 samples
python scripts/train_grpo.py --config configs/grpo_config.yaml --max_samples 100
```

### 5. Evaluate

```bash
python scripts/evaluate.py --model_path outputs/grpo_YYYYMMDD_HHMMSS/best_model
```

### Google Colab

The notebook `notebooks/colab_grpo.ipynb` is self-contained -- it creates all source files, runs pipeline tests, downloads Spider, and trains. Set runtime to T4 GPU and run all cells.

---

## References

- **SQL-R1**: Training LLMs to Reason about SQL with Reinforcement Learning (arXiv:2504.08600)
- **DeepSeek-R1**: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (arXiv:2501.12948)
- **Spider**: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task (Yu et al., EMNLP 2018)
- **QLoRA**: Efficient Finetuning of Quantized Language Models (Dettmers et al., NeurIPS 2023)

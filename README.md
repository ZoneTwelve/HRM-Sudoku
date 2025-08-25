# HRMACT Sudoku Solver

A PyTorch implementation of the **Hierarchical Reasoning Model** with **Adaptive Computation Time (ACT)**. tailored for solving Sudoku puzzles using hierarchical abstract reasoning.

## Highlights

- Dual-mode hierarchical reasoning with **high-level planning** and **low-level detail**, inspired by HRM’s brain-like architecture ([sapientinc/HRM](https://github.com/sapientinc/HRM/tree/main)).
- **Adaptive Computation Time (ACT)** enables the model to dynamically decide when to halt reasoning.
- **Efficient training** using curriculum learning across Sudoku difficulty levels.
- Lightweight implementation delivers strong performance even with limited data.
- Built-in **CLI interface** modeled after Tanmay Bakshi’s Swift implementation ([tanmay-bakshi/HierarchicalReasoningModel](https://github.com/tanmay-bakshi/HierarchicalReasoningModel)).

## Prerequisites

Ensure you have:

- Python 3.8+
- PyTorch (with MPS support for Apple Silicon, or CUDA for GPU users)
- `numpy`
- `safetensors`

Install dependencies:

```bash
pip install -r requirements.txt
````

## Usage

### Training

```bash
python main.py train
```

* Trains the HRMACT model with curriculum learning across difficulty levels.
* Saves checkpoints every 250 steps (and on step 1) in `.safetensors` format.

### Inference

```bash
python main.py infer <checkpoint> <difficulty>
```

Example:

```bash
python main.py infer checkpoint-1000.safetensors hard
```

Arguments:

* `checkpoint`: Path to a saved model checkpoint (`.safetensors`).
* `difficulty`: One of `very-easy`, `easy`, `medium`, `hard`, `extreme`.


## Architecture

* **Model (`model.py`)**
  Hierarchical transformer layers with ACT head, rotary position embeddings, SwiGLU, and RMSNorm.

* **Training (`train.py`)**

  * Cross-entropy loss on blank Sudoku cells.
  * Q-learning–style loss for the halting policy.
  * Curriculum learning: training gradually shifts from easy to harder puzzles.

* **Generation (`sudoku.py`)**
  Generates Sudoku puzzles with guaranteed unique solutions at chosen difficulty levels.

## Example: Inference Output

```
Using device: mps
Loaded model from checkpoint!

Puzzle:
+-------+-------+-------+
| . 3 . | . . . | . 2 . |
| 6 . 2 | . . 1 | . . . |
...

Solution:
+-------+-------+-------+
| 8 3 1 | 4 9 6 | 7 2 5 |
| 6 5 2 | 7 3 1 | 9 4 8 |
...

--- Segment 1 ---
Predicted solution (68 / 72):
+-------+-------+-------+
| 8 3 1 | 4 9 6 | 7 2 5 |
...
Q (halt – continue): 0.9421 – 0.0572
Model halted.
```

## Inspiration & Credits

* **[sapientinc/HRM](https://github.com/sapientinc/HRM/tree/main)** — Hierarchical Reasoning Model, demonstrating brain-inspired reasoning across puzzles and ARC tasks.
* **[tanmay-bakshi/HierarchicalReasoningModel](https://github.com/tanmay-bakshi/HierarchicalReasoningModel/)** — Swift implementation with elegant CLI design and usage inspiration.

## A Note on Scale & Performance

This repo is a **lightweight educational implementation**. For large-scale benchmarks, training regimes, and evaluations, see [sapientinc/HRM](https://github.com/sapientinc/HRM/tree/main).

## Citation

If you use this project in academic work, please cite:

```bibtex
@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model},
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734},
}
```

## License

Apache License 2.0.

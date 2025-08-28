# train.py

import torch
import torch.nn.functional as F
from model import HRMACTInner
from sudoku import generate_sudoku, Difficulty
import random

def sudoku_loss(model, hidden_states, board_inputs, board_targets, segments):
    config = model.config

    (next_hidden_states, output_logits,
     q_act_halt_logits, q_act_continue_logits) = model(hidden_states, board_inputs)

    # Output loss for Sudoku prediction
    output_loss = F.cross_entropy(
        output_logits.view(-1, config.vocab_size),
        board_targets.view(-1),
        reduction='none'
    ).view(board_inputs.shape)

    output_loss_mask = (board_inputs == 0).float()
    masked_output_loss = (output_loss * output_loss_mask).sum() / output_loss_mask.sum().clamp(min=1)

    # Accuracy for halting decision target
    with torch.no_grad():
        predictions = output_logits.argmax(dim=2)
        output_accuracy = ((predictions == board_targets) | (board_inputs != 0)).all(dim=1).long()

    # Q-ACT loss
    next_segments = segments + 1
    is_last_segment = next_segments >= config.act.halt_max_steps

    is_halted = is_last_segment | (q_act_halt_logits > q_act_continue_logits)

    # Halt exploration logic
    halt_exploration = torch.rand_like(q_act_halt_logits) < config.act.halt_exploration_probability
    if halt_exploration.any():
        min_halt_segments = torch.randint(2, config.act.halt_max_steps + 1, segments.shape, device=segments.device)
        min_halt_segments = min_halt_segments.long() * halt_exploration.long()
        is_halted = is_halted & (next_segments > min_halt_segments)

    # Compute target for Q-Continue
    with torch.no_grad():
        (_, _, next_q_act_halt, next_q_act_continue) = model(next_hidden_states, board_inputs)

    q_act_continue_target = torch.where(
        is_last_segment,
        next_q_act_halt,
        torch.maximum(next_q_act_halt, next_q_act_continue)
    ).sigmoid()

    q_act_loss = (
        F.binary_cross_entropy_with_logits(q_act_halt_logits, output_accuracy.float(), reduction='none') + \
        F.binary_cross_entropy_with_logits(q_act_continue_logits, q_act_continue_target, reduction='none')
    ) / 2
    avg_q_act_loss = q_act_loss.mean()

    total_loss = masked_output_loss + avg_q_act_loss

    # Metrics for logging
    with torch.no_grad():
        full_accuracy = ((predictions == board_targets) | (board_inputs != 0)).float().mean()
        q_act_halt_accuracy = ((q_act_halt_logits >= 0) == output_accuracy.bool()).float().mean()

    return (total_loss, masked_output_loss, avg_q_act_loss, is_halted,
            full_accuracy, q_act_halt_accuracy, next_hidden_states)

class TrainingBatch:
    DIFFICULTIES = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD, Difficulty.EXTREME]
    CURRICULUM_PROBAS = [
        [1.0, 0.0, 0.0, 0.0],
        [0.7, 0.3, 0.0, 0.0],
        [0.5, 0.4, 0.1, 0.0],
        [0.3, 0.3, 0.3, 0.1],
        [0.1, 0.3, 0.4, 0.2],
    ]

    def __init__(self, model: HRMACTInner, batch_size: int, device: torch.device):
        self.model = model
        self.batch_size = batch_size
        self.device = device
        self.curriculum_level = 0
        self.total_puzzles = 0

        hidden_size = model.config.transformers.hidden_size
        seq_len = model.config.seq_len

        self.board_inputs = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        self.board_targets = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        self.segments = torch.zeros(batch_size, dtype=torch.long, device=device)

        # FIX 1: Initialize hidden_states with new, independent tensors, not views of model parameters.
        low_level_h = torch.zeros((batch_size, seq_len + 1, hidden_size), dtype=model.config.dtype, device=device)
        high_level_h = torch.zeros((batch_size, seq_len + 1, hidden_size), dtype=model.config.dtype, device=device)
        self.hidden_states = (low_level_h, high_level_h)

        for i in range(batch_size):
            self.replace(i)

    def _sample_difficulty(self):
        return random.choices(self.DIFFICULTIES, self.CURRICULUM_PROBAS[self.curriculum_level], k=1)[0]

    def replace(self, idx: int):
        puzzle, solution = generate_sudoku(self._sample_difficulty())
        self.board_inputs[idx] = torch.tensor(puzzle.flatten(), device=self.device)
        self.board_targets[idx] = torch.tensor(solution.flatten(), device=self.device)

        # FIX 2: Wrap the state reset in a `torch.no_grad()` context.
        with torch.no_grad():
            self.segments[idx] = 0

            seq_len = self.model.config.seq_len
            low_level_h, high_level_h = self.hidden_states

            # This assignment correctly resets the state for one sample in the batch
            # using the model's learnable initial state parameters.
            low_level_h[idx] = self.model.initial_low_level.unsqueeze(0).expand(seq_len + 1, -1)
            high_level_h[idx] = self.model.initial_high_level.unsqueeze(0).expand(seq_len + 1, -1)

        self.total_puzzles += 1

    def graduate(self):
        if self.curriculum_level + 1 < len(self.CURRICULUM_PROBAS):
            self.curriculum_level += 1
            print(f"Graduated to curriculum level {self.curriculum_level}.")
        else:
            print("Reached highest curriculum level.")

def train_step(model, optimizer, batch):
    optimizer.zero_grad()

    (loss, out_loss, q_loss, is_halted,
     out_acc, q_acc, next_h) = sudoku_loss(
        model,
        batch.hidden_states,
        batch.board_inputs,
        batch.board_targets,
        batch.segments
    )

    loss.backward()
    optimizer.step()

    print(
        f"Output [Loss: {out_loss.item():.4f}, Acc: {out_acc.item():.4f}] | "
        f"Q-ACT [Loss: {q_loss.item():.4f}, Acc: {q_acc.item():.4f}] | "
        f"Puzzles [{batch.total_puzzles}] | Curriculum [{batch.curriculum_level}]"
    )

    batch.hidden_states = next_h
    batch.segments += 1

    halted_indices = torch.where(is_halted)[0]
    for idx in halted_indices:
        batch.replace(idx.item())

    return out_acc.item()

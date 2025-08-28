import subprocess
import sys
import os
import argparse
import re
from typing import Dict, Optional, Tuple

# --- Configuration ---
DIFFICULTY_LEVELS = ['very-easy', 'easy', 'medium', 'hard', 'extreme']
MAIN_SCRIPT_PATH = 'main.py'

def check_prerequisites(checkpoint_path: str) -> bool:
    """Checks if required files exist before running the evaluation."""
    if not os.path.exists(MAIN_SCRIPT_PATH):
        print(f"Error: The main script '{MAIN_SCRIPT_PATH}' was not found.")
        print("Please ensure this evaluation script is in the project's root directory.")
        return False
    if not os.path.exists(checkpoint_path):
        print(f"Error: The specified checkpoint file '{checkpoint_path}' was not found.")
        return False
    return True

def parse_final_grids(output: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parses the stdout to find the ground-truth solution and the FINAL
    predicted solution grid from the multi-segment output.
    """
    try:
        # Regex to capture the entire sudoku grid block.
        grid_pattern = r"\+-------(?:\+-------){2}\+[\s\S]*?\+-------(?:\+-------){2}\+"

        # --- THE KEY FIX IS HERE ---
        # Added `\s*` after `\n` to account for potential leading whitespace
        # before the grid starts, as seen in the user's log file.
        solution_match = re.search(r"Solution:\s*\n\s*(" + grid_pattern + ")", output, re.MULTILINE)
        predicted_matches = re.findall(r"Predicted solution .*:\s*\n\s*(" + grid_pattern + ")", output, re.MULTILINE)

        if not solution_match:
            print("DEBUG: Failed to parse the ground-truth 'Solution:' grid from the output.")
            return None, None
        
        if not predicted_matches:
            print("DEBUG: Failed to parse any 'Predicted solution' grids from the output.")
            return None, None

        ground_truth_grid = solution_match.group(1).strip()
        final_predicted_grid = predicted_matches[-1].strip()
        
        return ground_truth_grid, final_predicted_grid

    except Exception as e:
        print(f"DEBUG: An exception occurred during output parsing: {e}")
        return None, None


def run_single_inference(checkpoint_path: str, difficulty: str, puzzle_num: int) -> bool:
    """
    Runs a single inference, parses the output, and compares the final
    predicted solution to the ground-truth solution.
    """
    command = [sys.executable, MAIN_SCRIPT_PATH, 'infer', checkpoint_path, difficulty]
    
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8'
        )
        
        stdout = process.stdout
        ground_truth, final_prediction = parse_final_grids(stdout)

        if ground_truth is None or final_prediction is None:
            print(f"  Puzzle {puzzle_num}/{args.num_puzzles}: Failed (Could not parse output)")
            print("------------------------- DEBUG LOG: RAW STDOUT -------------------------")
            print(stdout)
            print("-----------------------------------------------------------------------")
            return False

        is_correct = (ground_truth == final_prediction)

        if not is_correct:
            print(f"  Puzzle {puzzle_num}/{args.num_puzzles}: Failed (Prediction mismatch)")
            print("------------------------- DEBUG LOG: GRID COMPARISON -------------------------")
            print("Ground Truth Solution:      | Model's FINAL Predicted Solution:")
            truth_lines = ground_truth.split('\n')
            pred_lines = final_prediction.split('\n')
            
            max_lines = max(len(truth_lines), len(pred_lines))
            truth_lines.extend([''] * (max_lines - len(truth_lines)))
            pred_lines.extend([''] * (max_lines - len(pred_lines)))

            for i in range(max_lines):
                print(f"{truth_lines[i]:<28}| {pred_lines[i]}")
            print("--------------------------------------------------------------------------")

        return is_correct

    except subprocess.CalledProcessError as e:
        print(f"\n--- ERROR on Puzzle {puzzle_num} ({difficulty}) ---")
        print(f"The inference script crashed. Return code: {e.returncode}")
        print("\n--- Captured STDOUT from script ---\n" + e.stdout)
        print("\n--- Captured STDERR from script ---\n" + e.stderr)
        print("-------------------------------------------\n")
        return False
    except Exception as e:
        print(f"An unexpected Python error occurred: {e}")
        return False

def evaluate_model(checkpoint_path: str, num_puzzles: int) -> Dict[str, float]:
    """Evaluates the model's accuracy across all difficulty levels."""
    if not check_prerequisites(checkpoint_path):
        sys.exit(1)

    results = {}
    print(f"Starting evaluation with checkpoint: '{checkpoint_path}'")
    print(f"Testing {num_puzzles} puzzles per difficulty level...\n")

    for difficulty in DIFFICULTY_LEVELS:
        print(f"--- Evaluating difficulty: {difficulty} ---")
        correct_count = 0
        for i in range(1, num_puzzles + 1):
            is_solved = run_single_inference(checkpoint_path, difficulty, i)
            if is_solved:
                correct_count += 1
                print(f"  Puzzle {i}/{num_puzzles}: Solved")

        accuracy = (correct_count / num_puzzles) * 100 if num_puzzles > 0 else 0
        results[difficulty] = accuracy
        print(f"--- Accuracy for {difficulty}: {accuracy:.2f}% ---\n")
    
    return results

def print_summary(results: Dict[str, float]):
    """Prints a final summary table of the evaluation results."""
    print("="*40)
    print("        EVALUATION SUMMARY")
    print("="*40)
    for difficulty, accuracy in results.items():
        print(f"  {difficulty:<12} | {accuracy:>6.2f}% correct")
    print("="*40)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate the HRMACT Sudoku Solver with detailed debug logging.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to the model checkpoint file (.safetensors)."
    )
    parser.add_argument(
        "-n", "--num_puzzles",
        type=int,
        default=20,
        help="Number of puzzles to evaluate for each difficulty level."
    )
    
    args = parser.parse_args()
    
    evaluation_results = evaluate_model(args.checkpoint, args.num_puzzles)
    print_summary(evaluation_results)
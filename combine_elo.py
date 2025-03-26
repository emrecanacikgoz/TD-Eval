#!/usr/bin/env python3
import json
import argparse
import math
import os
import sys
from collections import defaultdict
import numpy as np
from scipy import stats

# --- Global mapping for the 7 unique models (3 JSON files per model) ---
model_file_mapping = {
    "4o": [
        "results/tau/20250131_152422-tau-4o-retail/tau-gpt-4o_j.json",           # retail
        "results/mwoz/20250130_140218-4o/gpt-4o_c-gpt-4o_j.json",                # mwoz
        "results/tau/20250131_152503-tau-4o-airline/tau-gpt-4o_j.json"              # airline
    ],
    "4omini": [
        "results/tau/20250131_152338-tau-4omini-retail/tau-gpt-4o_j.json",          # retail
        "results/mwoz/20250130_140439-4omini/gpt-4o-mini_c-gpt-4o_j.json",          # mwoz
        "results/tau/20250131_152226-tau-4o-mini-airline/tau-gpt-4o_j.json"           # airline
    ],
    "gpt35": [
        "results/tau/20250131_152610-tau-gpt35-retail/tau-gpt-4o_j.json",           # retail
        "results/mwoz/20250130_145202-gpt35/gpt-3.5-turbo_c-gpt-4o_j.json",         # mwoz
        "results/tau/20250131_152708-tau-gpt35-airline/tau-gpt-4o_j.json"            # airline
    ],
    "sonnet": [
        "results/tau/20250131_152807-tau-sonnet-retail/tau-gpt-4o_j.json",          # retail
        "results/mwoz/20250130_183030-claude/claude-3-5-sonnet-20241022_c-gpt-4o_j.json",  # mwoz (claude-3-5-sonnet)
        "results/tau/20250131_153052-crashed-sonnet5/tau-gpt-4o_j_CRASHED.json"      # airline (crashed sonnet)
    ],
    "llama70b": [
        "results/tau/20250131_182136-tau-llama70b-retail/tau-gpt-4o_j.json",        # retail
        "results/mwoz/20250131_012449-llama70/meta-llama_Llama-3.3-70B-Instruct-Turbo_c-gpt-4o_j.json",  # mwoz (llama70)
        "results/tau/20250131_182042-airline-llama70b/tau-gpt-4o_j.json"             # airline
    ],
    "llama405b": [
        "results/tau/20250201_180118-tau405b-retail/tau-gpt-4o_j.json",             # retail
        "results/mwoz/20250131_012338-llama405/meta-llama_Meta-Llama-3.1-405B-Instruct-Turbo_c-gpt-4o_j.json",  # mwoz (llama405b)
        "results/tau/20250131_181808-tau405b-airline/tau-gpt-4o_j.json"              # airline
    ],
    "mistrallarge": [
        "results/tau/20250201_213127-tau-mistrallarge-retail/tau-gpt-4o_j.json",    # retail
        "results/mwoz/20250130_184905-mistrallarge/mistral-large-latest_c-gpt-4o_j.json",   # mwoz
        "results/tau/20250201_160723-tau-mistrallarge-airline/tau-gpt-4o_j.json"     # airline
    ]
}

def calculate_confidence_interval(elo_history, confidence=0.95):
    if len(elo_history) < 2:
        return (0, 0), 0
    std_dev = np.std(elo_history)
    z_score = stats.norm.ppf((1 + confidence) / 2)
    margin_error = z_score * (std_dev / math.sqrt(len(elo_history)))
    final_elo = elo_history[-1]
    return (final_elo - margin_error, final_elo + margin_error), z_score

def combine_results(result1, result2):
    """
    Combine two extended Elo results for a given model.
    The combination is weighted by total votes from each file.
    """
    votes1 = result1.get("total_votes", 0)
    votes2 = result2.get("total_votes", 0)
    total_votes = votes1 + votes2
    if total_votes > 0:
        combined_elo = (result1["elo"] * votes1 + result2["elo"] * votes2) / total_votes
    else:
        combined_elo = 1000.0

    combined_wins = result1.get("wins", 0) + result2.get("wins", 0)
    combined_losses = result1.get("losses", 0) + result2.get("losses", 0)
    combined_ties = result1.get("ties", 0) + result2.get("ties", 0)
    combined_total_votes = total_votes
    combined_picked_votes = result1.get("picked_votes", 0) + result2.get("picked_votes", 0)

    # Concatenate Elo history and changes (order not sequential)
    combined_elo_history = result1.get("elo_history", []) + result2.get("elo_history", [])
    combined_elo_changes = result1.get("elo_changes", []) + result2.get("elo_changes", [])
    avg_elo_change = np.mean(combined_elo_changes) if combined_elo_changes else 0
    std_elo_change = np.std(combined_elo_changes) if combined_elo_changes else 0

    # Merge head-to-head stats per opponent
    combined_head_to_head = defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0})
    for opponent, opp_stats in result1.get("head_to_head", {}).items():
        combined_head_to_head[opponent]["wins"] += opp_stats.get("wins", 0)
        combined_head_to_head[opponent]["losses"] += opp_stats.get("losses", 0)
        combined_head_to_head[opponent]["ties"] += opp_stats.get("ties", 0)
    for opponent, opp_stats in result2.get("head_to_head", {}).items():
        combined_head_to_head[opponent]["wins"] += opp_stats.get("wins", 0)
        combined_head_to_head[opponent]["losses"] += opp_stats.get("losses", 0)
        combined_head_to_head[opponent]["ties"] += opp_stats.get("ties", 0)
    combined_head_to_head = dict(combined_head_to_head)

    total_games = combined_wins + combined_losses + combined_ties
    win_rate = combined_wins / total_games if total_games > 0 else 0
    loss_rate = combined_losses / total_games if total_games > 0 else 0
    tie_rate = combined_ties / total_games if total_games > 0 else 0

    (ci_low, ci_high), z_score = calculate_confidence_interval(combined_elo_history)

    combined_result = {
        "elo": combined_elo,
        "elo_history": combined_elo_history,
        "confidence_interval": [ci_low, ci_high],
        "confidence_interval_details": {
            "range": [ci_low, ci_high],
            "z_score": z_score,
            "confidence_level": 0.95
        },
        "total_votes": combined_total_votes,
        "picked_votes": combined_picked_votes,
        "wins": combined_wins,
        "losses": combined_losses,
        "ties": combined_ties,
        "win_rate": win_rate,
        "loss_rate": loss_rate,
        "tie_rate": tie_rate,
        "head_to_head": combined_head_to_head,
        "elo_changes": combined_elo_changes,
        "avg_elo_change": avg_elo_change,
        "std_elo_change": std_elo_change,
    }
    return combined_result

def combine_detailed_comparisons(list_of_comparisons):
    """
    Simply concatenate detailed comparisons (from all files).
    """
    combined = []
    for comps in list_of_comparisons:
        combined.extend(comps)
    return combined

def combine_metadata(meta_list):
    """
    Iteratively combine a list of metadata dictionaries.
    """
    if not meta_list:
        return {}
    combined = meta_list[0]
    for m in meta_list[1:]:
        combined = combine_single_metadata(combined, m)
    return combined

def combine_single_metadata(meta1, meta2):
    merged = {}
    merged["judge_models"] = list(set(meta1.get("judge_models", [])) | set(meta2.get("judge_models", [])))
    merged["timestamps"] = meta1.get("timestamps", []) + meta2.get("timestamps", [])
    merged["total_rounds"] = meta1.get("total_rounds", 0) + meta2.get("total_rounds", 0)
    merged["total_comparisons"] = meta1.get("total_comparisons", 0) + meta2.get("total_comparisons", 0)
    merged["min_conversations"] = min(meta1.get("min_conversations", float("inf")), meta2.get("min_conversations", float("inf")))
    merged["comparisons_per_round"] = meta1.get("comparisons_per_round", None)
    merged["judge_agent"] = meta1.get("judge_agent", "")
    merged["participating_models"] = meta1.get("participating_models", [])
    if meta1.get("last_update") and meta2.get("last_update"):
        merged["last_update"] = meta1["last_update"] if meta1["last_update"] > meta2["last_update"] else meta2["last_update"]
    else:
        merged["last_update"] = meta1.get("last_update", meta2.get("last_update", ""))
    return merged

def combine_three_files(file_list, output):
    """
    Combine exactly 3 extended Elo JSON files.
    The expected order is: [retail, mwoz, airline].

    For each canonical model (as defined in model_file_mapping), we use the full file paths
    from the mapping as the expected keys to extract data from each input file.
    """
    try:
        with open(file_list[0], "r") as f:
            data_retail = json.load(f)
        with open(file_list[1], "r") as f:
            data_mwoz = json.load(f)
        with open(file_list[2], "r") as f:
            data_airline = json.load(f)
    except Exception as e:
        print(f"Error loading input files: {e}")
        sys.exit(1)

    # Use the full file paths as keys from the global mapping.
    expected_keys = {}  # canonical -> [retail_key, mwoz_key, airline_key]
    for canonical, paths in model_file_mapping.items():
        expected_keys[canonical] = paths

    combined_results = {}

    # For each canonical model, try to extract its result from each file using the expected key.
    for canonical, keys in expected_keys.items():
        retail_key, mwoz_key, airline_key = keys
        res_list = []
        # Retail file:
        if retail_key in data_retail.get("results", {}):
            res_list.append(data_retail["results"][retail_key])
        else:
            print(f"Warning: Key '{retail_key}' not found in retail file for canonical model '{canonical}'.")
        # MWOZ file:
        if mwoz_key in data_mwoz.get("results", {}):
            res_list.append(data_mwoz["results"][mwoz_key])
        else:
            print(f"Warning: Key '{mwoz_key}' not found in mwoz file for canonical model '{canonical}'.")
        # Airline file:
        if airline_key in data_airline.get("results", {}):
            res_list.append(data_airline["results"][airline_key])
        else:
            print(f"Warning: Key '{airline_key}' not found in airline file for canonical model '{canonical}'.")

        if res_list:
            combined = res_list[0]
            for r in res_list[1:]:
                combined = combine_results(combined, r)
            combined_results[canonical] = combined
        else:
            print(f"Warning: No results found for canonical model '{canonical}' in any file.")

    # Combine detailed comparisons from all files.
    detailed_all = combine_detailed_comparisons([
        data_retail.get("detailed_comparisons", []),
        data_mwoz.get("detailed_comparisons", []),
        data_airline.get("detailed_comparisons", [])
    ])

    # Merge metadata.
    meta_list = [
        data_retail.get("metadata", {}),
        data_mwoz.get("metadata", {}),
        data_airline.get("metadata", {})
    ]
    combined_meta = combine_metadata(meta_list)

    merged = {
        "results": combined_results,
        "detailed_comparisons": detailed_all,
        "metadata": combined_meta
    }
    try:
        with open(output, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"Combined Elo results from 3 files saved to {output}")
        print("\nFinal Elo ratings:")
        for model, result in merged.get("results", {}).items():
            print(f"  {model}: {result.get('elo', 'N/A')}")
    except Exception as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)

def reorder_files(file_list):
    """
    Given 3 file paths in arbitrary order, determine which is retail, airline, and mwoz.
    We assume:
      - The retail file has "retail" in its filename (case-insensitive).
      - The airline file has "airline" in its filename (case-insensitive).
      - The remaining file is treated as mwoz.
    """
    retail_file = None
    airline_file = None
    mwoz_file = None
    for file in file_list:
        lower_file = file.lower()
        if "retail" in lower_file:
            retail_file = file
        elif "airline" in lower_file:
            airline_file = file
        else:
            mwoz_file = file
    if not retail_file or not airline_file or not mwoz_file:
        print("Error: Could not determine all three file types (retail, airline, mwoz) from the provided filenames.")
        sys.exit(1)
    return [retail_file, mwoz_file, airline_file]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine exactly 3 extended Elo JSON files (retail, mwoz, and airline)."
    )
    parser.add_argument("files", nargs=3,
                        help="Paths to 3 extended Elo JSON files. The filename should contain 'retail' for the retail file, 'airline' for the airline file; the remaining file is assumed to be mwoz.")
    parser.add_argument("--output", default="results/elo_final_results.json",
                        help="Output file for combined Elo results (default: results/elo_final_results.json)")
    args = parser.parse_args()

    ordered_files = reorder_files(args.files)
    combine_three_files(ordered_files, args.output)
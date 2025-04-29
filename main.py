#!/usr/bin/env python3
"""
TD-Eval: Task-oriented Dialogue Evaluation Framework
Main entry point that dispatches to appropriate modules.
"""

import argparse
import os
import sys
from datetime import datetime

# Add module paths
sys.path.insert(0, os.path.abspath("./"))
sys.path.insert(0, os.path.abspath("./MultiWOZ_Evaluation"))

# Import functions from reorganized modules
from generate.generate import generate_agent_responses
from judge.judge import judge_agent_responses, load_agent_results
from postprocess.postprocess import postprocess_results
from judge.tau import main as tau_main


def main():
    parser = argparse.ArgumentParser(description='TD-Eval: Task-oriented Dialogue Evaluation Framework')
    
    # General parameters
    parser.add_argument('--dataset_path', type=str, default='datasets/woz_only.jsonl', 
                        help='Path to evaluation data')
    parser.add_argument('--agent_result_path', type=str, 
                        help='File path to already generated agent results (optional)')
    
    # Agent parameters
    parser.add_argument('--agent_client', type=str, default='openai', 
                        help='Client to use for LLM agent (openai, togetherai, anthropic, mistral)')
    parser.add_argument('--agent_model', type=str, default='gpt-4o', 
                        help='Agent model to evaluate')
    parser.add_argument('--use_gt_state', action='store_true', 
                        help='Uses ground truth state of multiwoz corpus (for debug)')
    
    # Judge parameters
    parser.add_argument('--judge_client', type=str, default='openai', 
                        help='Client to use for LLM judge agent (openai, togetherai, anthropic, mistral)')
    parser.add_argument('--judge_model', type=str, default='gpt-4o', 
                        help='Judge model to use for evaluation')
    
    # TAU benchmark parameters
    parser.add_argument('--tau_tool', action='store_true', 
                        help='Flag to judge as tau tool calling')
    parser.add_argument('--tau_react', action='store_true', 
                        help='Flag to judge as tau react')
    
    # Skip steps flags
    parser.add_argument('--skip_generation', action='store_true',
                        help='Skip agent response generation (requires --agent_result_path)')
    parser.add_argument('--skip_judging', action='store_true',
                        help='Skip judging step (only generate responses)')
    parser.add_argument('--skip_postprocessing', action='store_true',
                        help='Skip postprocessing of judge results')
    
    args = parser.parse_args()
    
    # Handle TAU benchmark evaluation
    if args.tau_tool or args.tau_react:
        is_react = args.tau_react
        result_dir, full_result_path = tau_main(args.dataset_path, args.judge_client, args.judge_model, is_react)
        if not args.skip_postprocessing:
            postprocess_results(full_result_path, result_dir)
        print("TAU benchmark evaluation completed successfully...")
        print(f"Results saved to {full_result_path}")
        return
    
    # Regular evaluation flow
    agent_result_data = None
    agent_result_path = args.agent_result_path

    # Step 1: Generate agent responses (if not skipped)
    if not args.skip_generation and not agent_result_path:
        print("Generating agent responses...")
        agent_result_data, agent_result_path, agent_success = generate_agent_responses(
            args.agent_client, args.agent_model, args.use_gt_state, args.dataset_path
        )
        if not agent_success:
            print("Agent generation encountered errors. Check the output file for details.")
            return
    elif agent_result_path:
        print(f"Loading agent results from {agent_result_path}...")
        agent_result_data = load_agent_results(agent_result_path)
    else:
        print("Error: Either provide --agent_result_path or do not use --skip_generation")
        return
    
    # Step 2: Judge agent responses (if not skipped)
    judge_result_path = None
    if not args.skip_judging:
        print("Judging agent responses...")
        _, judge_result_path, judge_success = judge_agent_responses(
            agent_result_data, args.judge_client, args.judge_model
        )
        if not judge_success:
            print("Judging encountered errors. Check the output file for details.")
            return
    else:
        print("Skipping judging step as requested.")
        return
    
    # Step 3: Postprocess results (if not skipped)
    if not args.skip_postprocessing and judge_result_path:
        print("Postprocessing results...")
        output_dir = postprocess_results(judge_result_path)
        print(f"Postprocessing completed. Results saved to {output_dir}")
    
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
import json
import os
import sys
from tqdm import tqdm
from typing import Tuple, Dict, Any

# Add necessary paths for imports
sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(0, os.path.abspath("../MultiWOZ_Evaluation"))

from judge.evaluator import judge_mwoz
from generate.llm_agents import openai_agent, togetherai_agent, mistral_agent, anthropic_agent
from MultiWOZ_Evaluation.mwzeval.metrics import Evaluator


def judge_conv_agent_results(conv_agent_data, judge_client_obj, judge_model):
    """Judge agent responses using the specified judge model."""
    dialogue_scores = {}
    dialogues = conv_agent_data.get('dialogues', {})
    idx_ = 0
    for dialogue_id, turns in tqdm(dialogues.items()):
        turn_scores = []
        try:
            for turn in turns:
                scores = {}
                dial_hist = turn["conversation_history"]
                domain = turn.get("domain", "")
                user_query = turn["user"]
                db_result = turn.get("db", {})
                agent_response = turn.get("response", "")
                scores = judge_mwoz(dial_hist, domain, user_query, db_result, agent_response, judge_client_obj, judge_model)
                
                turn_dict = {
                    "turn": turn.get("turn", 0),
                    "conversation_history": dial_hist,
                    "user": user_query,
                    "domain": domain,
                    "state": turn.get("state", {}),
                    "db": db_result,
                    "lex_response": turn.get("lex_response", ""),
                    "response": agent_response,
                    "ground_truth": turn.get("ground_truth", ""),
                    "scores": scores
                }
                turn_scores.append(turn_dict)

                # Optional: display examples for debugging
                if idx_ < 10:
                    tqdm.write("===")
                    tqdm.write("user: " + user_query)
                    tqdm.write("response: " + agent_response)
                    tqdm.write("scores: ")
                    
                    for metric, score_data in scores.items():
                        tqdm.write(f"{metric}: {score_data['score']} - {score_data['justification']}")
                
            dialogue_scores[dialogue_id] = turn_scores
            idx_ += 1
        except Exception as e:
            tqdm.write("error: " + str(e))
            return dialogue_scores, False
    return dialogue_scores, True


def get_judge_client(judge_client: str):
    """Get the appropriate judge client function based on provider name."""
    if judge_client == 'openai':
        return openai_agent
    elif judge_client == 'togetherai':
        return togetherai_agent
    elif judge_client == 'mistral':
        return mistral_agent
    elif judge_client == 'anthropic':
        return anthropic_agent
    else:
        raise ValueError(f"Invalid client: {judge_client}")


def judge_agent_responses(agent_result_data: dict, 
                         judge_client: str, 
                         judge_model: str) -> Tuple[dict, str, bool]:
    """Judge agent responses using the specified judge and save the results."""
    from datetime import datetime
    
    # Set up TOD judge agent
    judge_client_obj = get_judge_client(judge_client)
    
    # Get timestamp for result directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Get metadata from agent data
    agent_metadata = agent_result_data.get('metadata', {})
    
    # Setup judge metadata
    judge_metadata = {
        "judge_client": judge_client,
        "judge_model": judge_model,
        "agent_client": agent_metadata.get("agent_client", ""),
        "agent_model": agent_metadata.get("agent_model", ""),
    }
    
    # Judge the agent responses
    scores, isJudgeSuccess = judge_conv_agent_results(
        agent_result_data, judge_client_obj, judge_model
    )
    
    # Calculate success metrics
    e = Evaluator(bleu=True, success=True, richness=True)
    eval_results = e.evaluate(scores)
    judge_metadata['inform'] = eval_results['success']['inform']
    judge_metadata['success'] = eval_results['success']['success']
    
    # Prepare output
    dial_output = {
        "metadata": judge_metadata,
        "dialogues": scores,
        "mwzeval": eval_results['success']['dialogs']
    }
    
    # Create result directory and save results
    result_dir = os.path.join('results', 'judge_results')
    result_dir = os.path.join(result_dir, timestamp)
    os.makedirs(result_dir, exist_ok=True)
    
    agent_model = agent_metadata.get("agent_model", "unknown")
    judge_fname = f"{agent_model}_c-{judge_model}_j.json"
    # Replace slashes in model name with underscore to prevent file create error
    judge_fname = judge_fname.replace("/", "_")
    full_result_path = os.path.join(result_dir, judge_fname)
    
    if not isJudgeSuccess:
        fname_split = full_result_path.split(".")
        full_result_path = f"{fname_split[0]}_CRASHED.{fname_split[1]}"
    
    with open(full_result_path, 'w') as f:
        json.dump(dial_output, f, indent=4, ensure_ascii=False)
    
    return dial_output, full_result_path, isJudgeSuccess


def load_agent_results(agent_result_path: str) -> dict:
    """Load agent results from a JSON file."""
    with open(agent_result_path, 'r') as f:
        agent_result_data = json.load(f)
    return agent_result_data


if __name__ == "__main__":
    # This can be used for standalone testing
    import argparse
    
    parser = argparse.ArgumentParser(description='Judge agent responses')
    parser.add_argument('--agent_result_path', type=str, required=True,
                        help='Path to agent results JSON file')
    parser.add_argument('--judge_client', type=str, default='openai',
                        help='Client to use for LLM judge agent')
    parser.add_argument('--judge_model', type=str, default='gpt-4o',
                        help='Model to use for evaluation')
    
    args = parser.parse_args()
    
    # Load agent results
    agent_result_data = load_agent_results(args.agent_result_path)
    
    # Judge agent responses
    _, result_path, success = judge_agent_responses(
        agent_result_data, args.judge_client, args.judge_model
    )
    
    print("Evaluation completed successfully...")
    print(f"Results saved to {result_path}") 
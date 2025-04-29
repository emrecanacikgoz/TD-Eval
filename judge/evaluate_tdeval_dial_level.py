import argparse
from datetime import datetime
import json
import os
import sys
from tqdm import tqdm
from evaluator import judge_autotod_dial_level, judge_tau_dial_level
from generate.llm_agents import anthropic_agent, mistral_agent, openai_agent, togetherai_agent

def evaluate_dial_level_tau(
    judge_client: any, 
    judge_model: any, 
    dataset_path: str, 
    judge_data_path: str,
    eval_batches_path: str,
    tau_domain: str
):
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    with open(judge_data_path, 'r') as f:
        judge_data = json.load(f)
    with open(eval_batches_path, 'r') as f:
        eval_batches = json.load(f)
    if tau_domain == "retail":
        tau_batch = eval_batches["tau"]["retail"]
    else: 
        tau_batch = eval_batches["tau"]["airline"]
    dialogues = judge_data["dialogues"]
    judge_scores = {}
    try:
        for dial_id, dialogue in tqdm(dialogues.items()):
            if dial_id not in tau_batch:
                tqdm.write("skip")
                continue
            # get policy from goal field
            policy = dataset[int(dial_id)]['traj'][0]['content']
            dial_history = ""
            for i in range(len(dialogue)):
                turn = dialogue[i]
                user_query = turn["user"]
                dial_history += f"### User Query\n{user_query}\n"
                db = json.dumps(turn["db"])
                dial_history += f"### Backend Results\n{db}\n"
                agent_response = turn["response"]
                dial_history += f"### Chatbot Response\n{agent_response}\n"
            
            scores = judge_tau_dial_level(dial_history, policy, judge_client, judge_model)
            judge_scores[dial_id] = scores
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print("error: ", e)
        return judge_scores, False
    return judge_scores, True


def evaluate_dial_level_autotod(
    judge_client: any, 
    judge_model: any, 
    dataset_path: str, 
    judge_data_path: str,
    eval_batches_path: str
):
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    with open(judge_data_path, 'r') as f:
        judge_data = json.load(f)
    with open(eval_batches_path, 'r') as f:
        eval_batches = json.load(f)
    autotod_batch = eval_batches["autotod_mwoz"]
    dialogues = judge_data["dialogues"]
    judge_scores = {}
    try:
        for dial_id, dialogue in tqdm(dialogues.items()):
            if dial_id.split(".json")[0].lower() not in autotod_batch:
                continue
            # get policy from goal field
            policy = dataset[dial_id]['goal']
            dial_history = ""
            for i in range(len(dialogue)):
                turn = dialogue[i]
                user_query = turn["user"]
                dial_history += f"### User Query\n{user_query}\n"
                db = json.dumps(turn["db"])
                dial_history += f"### Backend Results\n{db}\n"
                agent_response = turn["response"]
                dial_history += f"### Chatbot Response\n{agent_response}\n"
            
            scores = judge_autotod_dial_level(dial_history, policy, judge_client, judge_model)
            judge_scores[dial_id] = scores
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print("error: ", e)
        return judge_scores, False
    return judge_scores, True

def main(
    dataset_path, 
    judge_data_path, 
    eval_batches_path, 
    domain,
    judge_client, 
    judge_model
):
    # set up TOD judge agent
    if judge_client == 'openai':
        judge_client_obj = openai_agent
    elif judge_client == 'togetherai':
        judge_client_obj = togetherai_agent
    elif judge_client == 'mistral':
        judge_client_obj = mistral_agent
    elif judge_client == 'anthropic':
        judge_client_obj = anthropic_agent
    else:
        raise ValueError("Invalid client")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if domain == 'mwoz':
        scores, isJudgeSuccess = evaluate_dial_level_autotod(
            judge_client_obj, 
            judge_model, 
            dataset_path, 
            judge_data_path,
            eval_batches_path
        )
    else:
        scores, isJudgeSuccess = evaluate_dial_level_tau(
            judge_client_obj, 
            judge_model, 
            dataset_path, 
            judge_data_path,
            eval_batches_path,
            domain
        )

    judge_metadata = {
        "filename": dataset_path,
        "judge_client": judge_client,
        "judge_model": judge_model
    }
    judge_output = {
        "metadata": judge_metadata,
        "dialogues": scores
    }

    result_dir = os.path.join('results', f'dial_level_results')
    result_dir = os.path.join(result_dir, timestamp)
    os.makedirs(result_dir, exist_ok=True)
    judge_fname = f"{domain}-dial-level-{judge_model}_j.json"
    full_result_path = os.path.join(result_dir, judge_fname)
    if not isJudgeSuccess:
        fname_split = full_result_path.split(".")
        full_result_path = f"{fname_split[0]}_CRASHED.{fname_split[1]}"
    with open(full_result_path, 'w') as f:
        json.dump(judge_output, f, indent=4, ensure_ascii=False)
    return result_dir, full_result_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate at dialogue level TOD agent using TD-Eval')
    parser.add_argument('--dataset_path', type=str, default='datasets/out_basic_100_bt.json', help='Path to evaluation data')
    parser.add_argument('--judge_data_path', type=str, default='results/judge_results_mwoz_autotod/20250403_025805/mwoz-autotod-gpt-4o_j.json', help='Path to evaluation data')
    parser.add_argument('--eval_batches_path', type=str, default='datasets/main_human_eval/all_batch.json', help='Path to evaluation data')
    parser.add_argument('--judge_client', type=str, default='openai', help='Client to use for LLM judge agent')
    parser.add_argument('--judge_model', type=str, default='gpt-4o', help='Agent to use for evaluation')
    parser.add_argument('--tau_airline', action='store_true', help="indicates to evaluate based on tau airline domain format")
    parser.add_argument('--tau_retail', action='store_true', help="indicates to evaluate based on tau retail domain format")

    args = parser.parse_args()

    if (args.tau_airline and args.tau_retail):
        print("ERROR: tau retail and airline cannot be set together!")
        exit()
    if args.tau_airline:
        domain = 'airline'
    elif args.tau_retail:
        domain = 'retail'
    else:
        domain = 'mwoz'

    result_dir, full_result_path = main(
        args.dataset_path, 
        args.judge_data_path, 
        args.eval_batches_path,
        domain,
        args.judge_client, 
        args.judge_model
    )
    print("Evaluation completed successfully...")
    print(f"Results saved to {full_result_path}")
import argparse
from datetime import datetime
import json
import os
import re
import sys
from tqdm import tqdm

import requests
import json
from time import sleep

url = "https://api.contextual.ai/v1/lmunit"

headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": "Bearer key-CLRoggUEDxqJn3DHU6hPHk3R5f6KL98IEgDBpISri1Iwp8ptg"
}

conv_qs = [
    "Does the response directly relate to the dialogue history and the current user query?",
    "Does the response remain on-topic with the dialogue history and the user query?",
    "Does the response logically continue the progression of the dialogue?"
]
backend_qs = [
    "Does the response accurately reflect the information in the database results?",
    "Does the response stay on-topic with the database results and the dialogue context?",
    "Does response logically incorporate and progress based on the database results?"
]
policy_qs = [
    "Does the response provide suggestions only when the database results are few enough to do so?",
    "Does the response request required, relevant information from the user before offering suggestions or booking services?",
    "Does the response avoid premature actions (i.e. make a booking or suggest a service) too early in the conversation, before the necessary information is gathered?"
]

def evaluate_dial_level_tau(    
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

    conv_query = f"Generate high quality customer service chatbot responses to {domain} customer queries. Focus on making sure the responses align with conversation context."
    backend_query = f"Generate high quality customer service chatbot responses to {domain} customer queries, include the backend AI thinking. Focus on making sure the response aligns with backend thinking."
    policy_query = f"Generate high quality customer service chatbot responses to {domain} customer queries, include the backend AI thinking. Focus on making sure the responses align with store policy."

    dialogues = judge_data["dialogues"]
    judge_scores = {}
    try:
        for dial_id, dialogue in tqdm(dialogues.items()):
            if dial_id not in tau_batch:
                tqdm.write("skip")
                continue
            dial_history = ""
            dial_history_backend = ""
            for i in range(len(dialogue)):
                turn = dialogue[i]
                user_query = turn["user"]
                dial_history_backend += f"### User Query\n{user_query}\n"
                dial_history += f"### User Query\n{user_query}\n"
                db = json.dumps(turn["db"])
                dial_history_backend += f"### Backend Results\n{db}\n"
                agent_response = turn["response"]
                dial_history_backend += f"### Chatbot Response\n{agent_response}\n"
                dial_history += f"### Chatbot Response\n{agent_response}\n"
            
            # get conversation consistency score
            conv_payload = {
                "query": conv_query,
                "response": dial_history,
                "unit_test": ""
            }
            tqdm.write("Conv. Consistency")
            conv_score = 0
            for q in conv_qs:
                conv_payload["unit_test"] = q
                conv = requests.post(url, json=conv_payload, headers=headers)
                conv_score += float(json.loads(conv.text)["score"])
                tqdm.write(json.dumps(conv_payload, indent=2))
                tqdm.write("score: " + conv.text)
                sleep(3)
            conv_score = round(conv_score/3, 2)
            # get backend knowledge consistency score
            backend_payload = {
                "query": backend_query,
                "response": dial_history_backend,
                "unit_test": ""
            }
            tqdm.write("Backend Knowledge")
            backend_score = 0
            for q in backend_qs:
                backend_payload["unit_test"] = q
                backend = requests.post(url, json=backend_payload, headers=headers)
                print(json.loads(backend.text))
                backend_score += float(json.loads(backend.text)["score"])
                tqdm.write(json.dumps(backend_payload, indent=2))
                tqdm.write("score: " + backend.text)
                sleep(3)
            backend_score = round(backend_score/3, 2)
            # get policy compliance score
            policy_payload = {
                "query": policy_query,
                "response": dial_history_backend,
                "unit_test": ""
            }
            policy_score = 0
            tqdm.write("Policy Compliance")
            for q in policy_qs:
                policy_payload["unit_test"] = q
                policy = requests.post(url, json=policy_payload, headers=headers)
                policy_score += float(json.loads(policy.text)["score"])
                tqdm.write(json.dumps(policy_payload, indent=2))
                tqdm.write("score: " + policy.text)
                sleep(3)
            policy_score = round(policy_score/3, 2)
            sleep(5)
            judge_scores[dial_id] = {
                "conv_consistency": {"score": conv_score},
                "backend_consistency": {"score": backend_score},
                "policy_completeness": {"score": policy_score}
            } 

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print("error: ", e)
        return judge_scores, False
    return judge_scores, True

def evaluate_dial_level_autotod(
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

    conv_query = f"Generate high quality task assistant chatbot responses to user queries. Focus on making sure the responses align with conversation context."
    backend_query = f"Generate high quality task assistant chatbot responses to user queries, include the backend AI thinking. Focus on making sure the response aligns with backend thinking."
    policy_query = f"Generate high quality task assistant chatbot responses to user queries, include the backend AI thinking. Focus on making sure the responses align with store policy."

    try:
        for dial_id, dialogue in tqdm(dialogues.items()):
            if dial_id.split(".json")[0].lower() not in autotod_batch:
                continue
            dial_history = ""
            dial_history_backend = ""
            for i in range(len(dialogue)):
                turn = dialogue[i]
                user_query = turn["user"]
                dial_history_backend += f"### User Query\n{user_query}\n"
                dial_history += f"### User Query\n{user_query}\n"
                db = json.dumps(turn["db"])
                dial_history_backend += f"### Backend Results\n{db}\n"
                agent_response = turn["response"]
                dial_history_backend += f"### Chatbot Response\n{agent_response}\n"
                dial_history += f"### Chatbot Response\n{agent_response}\n"
            
            # get conversation consistency score
            conv_payload = {
                "query": conv_query,
                "response": dial_history,
                "unit_test": ""
            }
            tqdm.write("Conv. Consistency")
            conv_score = 0
            for q in conv_qs:
                conv_payload["unit_test"] = q
                conv = requests.post(url, json=conv_payload, headers=headers)
                conv_score += float(json.loads(conv.text)["score"])
                tqdm.write(json.dumps(conv_payload, indent=2))
                tqdm.write("score: " + conv.text)
                sleep(3)
            conv_score = round(conv_score/3, 2)
            # get backend knowledge consistency score
            backend_payload = {
                "query": backend_query,
                "response": dial_history_backend,
                "unit_test": ""
            }
            tqdm.write("Backend Knowledge")
            backend_score = 0
            for q in backend_qs:
                backend_payload["unit_test"] = q
                backend = requests.post(url, json=backend_payload, headers=headers)
                print(json.loads(backend.text))
                backend_score += float(json.loads(backend.text)["score"])
                tqdm.write(json.dumps(backend_payload, indent=2))
                tqdm.write("score: " + backend.text)
                sleep(3)
            backend_score = round(backend_score/3, 2)
            # get policy compliance score
            policy_payload = {
                "query": policy_query,
                "response": dial_history_backend,
                "unit_test": ""
            }
            policy_score = 0
            tqdm.write("Policy Compliance")
            for q in policy_qs:
                policy_payload["unit_test"] = q
                policy = requests.post(url, json=policy_payload, headers=headers)
                policy_score += float(json.loads(policy.text)["score"])
                tqdm.write(json.dumps(policy_payload, indent=2))
                tqdm.write("score: " + policy.text)
                sleep(3)
            policy_score = round(policy_score/3, 2)
            sleep(5)
            judge_scores[dial_id] =  {
                "conv_consistency": {"score": conv_score},
                "backend_consistency": {"score": backend_score},
                "policy_completeness": {"score": policy_score}
            } 
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
):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if domain == 'mwoz':
        scores, isJudgeSuccess = evaluate_dial_level_autotod( 
            dataset_path, 
            judge_data_path,
            eval_batches_path
        )
    else:
        scores, isJudgeSuccess = evaluate_dial_level_tau(
            dataset_path, 
            judge_data_path,
            eval_batches_path,
            domain
        )

    judge_metadata = {
        "filename": dataset_path,
        "judge_client": "lmunit",
    }
    judge_output = {
        "metadata": judge_metadata,
        "dialogues": scores
    }
    result_dir = os.path.join('results', f'dial_level_results')
    result_dir = os.path.join(result_dir, timestamp)
    os.makedirs(result_dir, exist_ok=True)
    judge_fname = f"{domain}-dial-level-lmunit_j.json"
    full_result_path = os.path.join(result_dir, judge_fname)
    if not isJudgeSuccess:
        fname_split = full_result_path.split(".")
        full_result_path = f"{fname_split[0]}_CRASHED.{fname_split[1]}"
    with open(full_result_path, 'w') as f:
        json.dump(judge_output, f, indent=4, ensure_ascii=False)
    return result_dir, full_result_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate at dialogue level TOD agent using LMUnit')
    parser.add_argument('--dataset_path', type=str, default='datasets/out_basic_100_bt.json', help='Path to evaluation data')
    parser.add_argument('--judge_data_path', type=str, default='results/judge_results_mwoz_autotod/20250403_025805/mwoz-autotod-gpt-4o_j.json', help='Path to evaluation data')
    parser.add_argument('--eval_batches_path', type=str, default='datasets/main_human_eval/all_batch.json', help='Path to evaluation data')
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
        domain
    )
    print("Evaluation completed successfully...")
    print(f"Results saved to {full_result_path}")
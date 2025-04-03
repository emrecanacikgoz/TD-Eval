import argparse
from datetime import datetime
import json
import os
import re
import sys
from tqdm import tqdm
from postprocess import postprocess_results

import requests
import json
from time import sleep
from calculate_annotator_agreement import extract_score

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

def eval_dials_lmunit(dialogue_history, db_results, user_query, agent_response):
    # history = turn["conversation_history"] + "\nCustomer: " + turn["user"]
    history = dialogue_history + f"\nCustomer: {user_query}"
    # db = json.dumps(turn["db"])
    # get conversation consistency score
    conv_payload = {
        "query": history,
        "response": agent_response,#turn["response"],
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
        "query": history + "\nDatabase result: " + db_results,
        "response": user_query, #turn["response"],
        "unit_test": ""
    }
    tqdm.write("Backend Knowledge")
    backend_score = 0
    for q in backend_qs:
        backend_payload["unit_test"] = q
        backend = requests.post(url, json=backend_payload, headers=headers)
        backend_score += float(json.loads(backend.text)["score"])
        tqdm.write(json.dumps(backend_payload, indent=2))
        tqdm.write("score: " + backend.text)
        sleep(3)
    backend_score = round(backend_score/3, 2)
    # get policy compliance score
    policy_payload = {
        "query": history + "\nDatabase result: " + db_results,
        "response": user_query, #turn["response"],
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
    sleep(10)
    return {
        "conv_consistency": {"score": conv_score},
        "backend_consistency": {"score": backend_score},
        "policy_completeness": {"score": policy_score}
    } 

def evaluate_mwoz_react(dataset_path):
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    judge_scores = {}
    try:
        for dial_id, dial_info in tqdm(dataset.items()):
            dial = dial_info['log']
            # policy = ""
            # domain = ""
            dialogue_history = []
            user_query = ""
            db_call = {}
            db_results = ""
            agent_response = ""
            turn_responses = []
            api_pattern = r"API Name:(.*?)\nAPI Input:(.*?)\n(?:API Result:(.*?))$"
            response_pattern = r"Response:(.*?)(?=Thought:|API Name:|\n```|$)"
            # get policy from goal field
            policy = dial_info['goal']
            for i in tqdm(range(len(dial))):
                turn = dial[i]
                user_query = f"Customer: {turn["usr"]}"
                agent_response = f"Agent: {turn["response"]}"
                # parse db call
                react_backend = turn["answers"]
                react_thoughts = react_backend.split("Thought:")
                react_thought_blocks = []
                for thought in react_thoughts:
                    thought = thought.strip()
                    if len(thought) == 0:
                        continue
                    # tqdm.write(f"thought - {thought}")
                    thought_message = thought.split("\n")[0]
                    # try to extract api/db calls
                    api_matches = re.findall(api_pattern, thought, re.DOTALL)
                    if len(api_matches) > 0:
                        api_match = api_matches[0]
                        api_name = api_match[0].strip()
                        api_input = api_match[1].strip()
                        api_result = api_match[2].strip() if api_match[2] else None
                        try:
                            api_input = json.loads(api_input)
                        except (json.JSONDecodeError, TypeError):
                            print("json decode input fail")
                            pass # leave as string
                        try: 
                            api_result = json.loads(api_result)
                        except (json.JSONDecodeError, TypeError):
                            print("json decode results fail")
                            pass
                        react_thought_blocks.append({
                            "thought": f"Thought: {thought_message}",
                            "api_name": api_name,
                            "api_input": api_input,
                            "api_result": api_result
                        })
                    else:
                        # try to extract response
                        response_match = re.findall(response_pattern, thought, re.DOTALL)
                        if len(response_match) > 0:
                            response = response_match[0].strip()
                            react_thought_blocks.append({
                                "thought": f"Thought: {thought_message}",
                                "response": response
                            })
                # stringify db results
                for i, block in enumerate(react_thought_blocks):
                    if i < len(react_thought_blocks) - 1:
                        db_results += f"{json.dumps(block)}\n"
                    else:
                        db_results += f"{block["thought"]}\n"
                # compile inputs for llm call
                dial_history = "\n".join(dialogue_history)
                db_call = react_thought_blocks

                # print turn results
                tqdm.write("dialogue history: " + "\n".join(dialogue_history))
                tqdm.write("user_query: " + user_query)
                tqdm.write("db call: " + db_results)
                tqdm.write("agent_response: " + agent_response)
                scores = eval_dials_lmunit(dial_history, db_results, user_query, agent_response)
                
                turn_responses.append({
                    "turn": i,
                    "conversation_history": dial_history,
                    "user": user_query,
                    "db": db_call,
                    "response": agent_response,
                    "scores": scores
                })
                # reset turn
                dialogue_history.append(user_query)
                dialogue_history.append(agent_response)
                # reset variables
                user_query = ""
                db_call = {}
                db_results = ""
                agent_response = ""
            judge_scores[dial_id] = turn_responses
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print("error: ", e)
        return judge_scores, False
    return judge_scores, True

def main(dataset_path):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    scores, isJudgeSuccess = evaluate_mwoz_react(dataset_path)

    judge_metadata = {
        "filename": dataset_path,
        "judge_client": "lmunit",
    }
    judge_output = {
        "metadata": judge_metadata,
        "dialogues": scores
    }

    result_dir = os.path.join('results', 'autotod_lmunit_judge_results')
    result_dir = os.path.join(result_dir, timestamp)
    os.makedirs(result_dir, exist_ok=True)
    judge_fname = f"mwoz-autotod-lmunit_j.json"
    full_result_path = os.path.join(result_dir, judge_fname)
    if not isJudgeSuccess:
        fname_split = full_result_path.split(".")
        full_result_path = f"{fname_split[0]}_CRASHED.{fname_split[1]}"
    with open(full_result_path, 'w') as f:
        json.dump(judge_output, f, indent=4, ensure_ascii=False)
    return result_dir, full_result_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate mwoz autotod TOD agent')
    parser.add_argument('--dataset_path', type=str, default='datasets/out_basic_100_bt.json', help='Path to evaluation data')
    args = parser.parse_args()

    result_dir, full_result_path = main(args.dataset_path)

    postprocess_results(full_result_path, result_dir)

    print("Evaluation completed successfully...")
    print(f"Results saved to {full_result_path}")
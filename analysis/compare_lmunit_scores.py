import requests
import json
from time import sleep
from calculate_annotator_agreement import extract_score
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

url = "https://api.contextual.ai/v1/lmunit"

headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": "Bearer key-CLRoggUEDxqJn3DHU6hPHk3R5f6KL98IEgDBpISri1Iwp8ptg"
}

# Set Model and File Locations
dials = {
    "tau_retail_eval_json": "results/judge-results-tau/20250131_152422-tau-4o-retail/tau-gpt-4o_j.json",
    "tau_air_eval_json": "results/judge-results-tau/20250131_152503-tau-4o-airline/tau-gpt-4o_j.json",
}

def load_filter_dials(tau_retail_eval_json, tau_air_eval_json):
    # load dialogues
    with open(tau_retail_eval_json, 'r') as f:
        tau_retail_eval = json.load(f)
    tau_retail_dials = tau_retail_eval['dialogues']
    
    with open(tau_air_eval_json, 'r') as f:
        tau_air_eval = json.load(f)
    tau_air_dials = tau_air_eval['dialogues']
    
    # load batches
    # Use the batch.json file
    dial_batches = "datasets/main_human_eval/batch.json"

    batch_list = None
    with open(dial_batches, 'r') as f:
        batch_list = json.load(f)
    if batch_list is None or len(batch_list) == 0:
        print('No batches found at this path:', dial_batches)
        exit()
    
    # load dialogues
    batch_dials = {}
    tau_retail_batch_ids = batch_list["tau"]["retail"]
    tau_air_batch_ids = batch_list["tau"]["airline"]      
    
    for batch_id in tau_retail_batch_ids:
        for id, dial in tau_retail_dials.items():
            if id == batch_id:
                batch_dials[id] = dial
                break
                
    for batch_id in tau_air_batch_ids:
        for id, dial in tau_air_dials.items():
            if id == batch_id:
                batch_dials[id] = dial
                break
    return batch_dials

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

def eval_dials_lmunit(batch_dials):
    lmunit_scores = {}
    for id, dial in batch_dials.items():
        lmunit_scores[id] = []
        for i, turn in enumerate(dial):
            history = turn["conversation_history"] + "\nCustomer: " + turn["user"]
            if "lex_response" in turn:
                response = turn["lex_response"]
            else:
                response = turn["response"]
            db = json.dumps(turn["db"])
            # get conversation consistency score
            conv_payload = {
                "query": history,
                "response": response,
                "unit_test": ""
            }
            print("Conv. Consistency")
            conv_score = 0
            for q in conv_qs:
                conv_payload["unit_test"] = q
                conv = requests.post(url, json=conv_payload, headers=headers)
                conv_score += float(json.loads(conv.text)["score"])
                print(json.dumps(conv_payload, indent=2))
                print("score:", conv.text)
                sleep(3)
            conv_score = round(conv_score/3, 2)
            # get backend knowledge consistency score
            backend_payload = {
                "query": history + "\nDatabase result: " + db,
                "response": response,
                "unit_test": ""
            }
            print("Backend Knowledge")
            backend_score = 0
            for q in backend_qs:
                backend_payload["unit_test"] = q
                backend = requests.post(url, json=backend_payload, headers=headers)
                backend_score += float(json.loads(backend.text)["score"])
                print(json.dumps(backend_payload, indent=2))
                print("score:", backend.text)
                sleep(3)
            backend_score = round(backend_score/3, 2)
            # get policy compliance score
            policy_payload = {
                "query": history + "\nDatabase result: " + db,
                "response": response,
                "unit_test": ""
            }
            policy_score = 0
            print("Policy Compliance")
            for q in policy_qs:
                policy_payload["unit_test"] = q
                policy = requests.post(url, json=policy_payload, headers=headers)
                policy_score += float(json.loads(policy.text)["score"])
                print(json.dumps(policy_payload, indent=2))
                print("score:", policy.text)
                sleep(3)
            policy_score = round(policy_score/3, 2)
            # compile scores
            lmunit_scores[id].append({
                "conv_consistency": conv_score,
                "backend_consistency": backend_score,
                "policy_completeness": policy_score
            })
        sleep(5)
    return lmunit_scores

# Main execution
if __name__ == "__main__":
    # Load and filter dialogues
    batch_dials = load_filter_dials(dials["tau_retail_eval_json"], dials["tau_air_eval_json"])
    
    # Evaluate dialogues
    lmunit_scores = eval_dials_lmunit(batch_dials)
    
    # Save results to a file
    with open("lmunit_scores.json", "w") as f:
        json.dump(lmunit_scores, f, indent=2)
    
    print("Evaluation complete. Results saved to lmunit_scores.json") 
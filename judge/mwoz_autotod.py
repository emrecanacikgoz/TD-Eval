import argparse
from datetime import datetime
import json
import os
import re
import sys
from tqdm import tqdm
from judge.llm_evaluator import judge_mwoz_autotod
from generate.llm_agents import anthropic_agent, mistral_agent, openai_agent, togetherai_agent
from postprocess.postprocess import postprocess_results

def parse_domain(api_query):
    if "restaurant" in api_query:
        return "restaurant"
    elif "hotel" in api_query:
        return "hotel"
    elif "attraction" in api_query:
        return "attraction"
    elif "train" in api_query:
        return "train"
    elif "taxi" in api_query:
        return "taxi"
    else:
        return None

def evaluate_mwoz_react(judge_client, judge_model, dataset_path):
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    judge_scores = {}
    try:
        for dial_id, dial_info in tqdm(dataset.items()):
            dial = dial_info['log']
            policy = ""
            domain = ""
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
                        # parse domain from api_name
                        domain = parse_domain(api_name)
                        # try to parse api input and result and json
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

                scores = judge_mwoz_autotod(dial_history, domain, user_query, db_results, agent_response, policy, judge_client, judge_model)
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

def main(dataset_path, judge_client, judge_model):
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
    scores, isJudgeSuccess = evaluate_mwoz_react(judge_client_obj, judge_model, dataset_path)

    judge_metadata = {
        "filename": dataset_path,
        "judge_client": judge_client,
        "judge_model": judge_model
    }
    judge_output = {
        "metadata": judge_metadata,
        "dialogues": scores
    }

    result_dir = os.path.join('results', 'judge_results_mwoz_autotod')
    result_dir = os.path.join(result_dir, timestamp)
    os.makedirs(result_dir, exist_ok=True)
    judge_fname = f"mwoz-autotod-{judge_model}_j.json"
    full_result_path = os.path.join(result_dir, judge_fname)
    if not isJudgeSuccess:
        fname_split = full_result_path.split(".")
        full_result_path = f"{fname_split[0]}_CRASHED.{fname_split[1]}"
    with open(full_result_path, 'w') as f:
        json.dump(judge_output, f, indent=4, ensure_ascii=False)
    return result_dir, full_result_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate mwoz autotod TOD agent')
    parser.add_argument('--dataset_path', type=str, default='../datasets/autotod_dials_bt.json', help='Path to evaluation data')
    parser.add_argument('--judge_client', type=str, default='openai', help='Client to use for LLM judge agent')
    parser.add_argument('--judge_model', type=str, default='gpt-4o', help='Agent to use for evaluation')
    args = parser.parse_args()

    result_dir, full_result_path = main(args.dataset_path, args.judge_client, args.judge_model)

    postprocess_results(full_result_path, result_dir)

    print("Evaluation completed successfully...")
    print(f"Results saved to {full_result_path}")
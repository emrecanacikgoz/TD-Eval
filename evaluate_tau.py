import argparse
from datetime import datetime
import json
import os
from tqdm import tqdm
from evaluator import judge_tau
from llm_agents import anthropic_agent, mistral_agent, openai_agent, togetherai_agent
from postprocess import postprocess_results

def evaluate_tau_tool_call(judge_client, judge_model, dataset_path):
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    judge_scores = {}
    try:
        for dial_ind in tqdm(range(len(dataset))):
            dial = dataset[dial_ind]['traj']
            policy = ""
            dialogue_history = []
            user_query = ""
            db_call = {}
            db_results = ""
            agent_response = ""
            turn_responses = []
            for turn_ind in tqdm(range(len(dial))):
                turn = dial[turn_ind]
                if turn_ind == 0:
                    if turn['role'] != "system":
                        tqdm.write('Format Error: System policy should be first')
                        exit()
                    else: 
                        policy = turn['content']
                else:
                    if turn['role'] == 'user':
                        user_query = f"Customer: {turn['content']}"
                    elif turn['role'] == 'tool':
                        name = turn['name']
                        content = turn['content']
                        if name in db_call:
                            db_call[name]['output'] = content
                    elif turn['role'] == 'assistant' and turn['tool_calls'] != None:
                        tool_calls = turn["tool_calls"]
                        for calls in tool_calls:
                            func = calls["function"]
                            db_call[func["name"]] = {"args": func["arguments"]}
                        if turn['content'] != None:
                            agent_response = f"Agent: {turn['content']}\n"
                    elif turn['role'] == 'assistant' and turn['tool_calls'] == None:
                        agent_response += f"Agent: {turn['content']}"
                        # convert db call to string
                        for func, props in db_call.items():
                            db_results += f"Function: {func}\nArgs: {props['args']}\n Output: {props['output']}\n\n"
                        dial_history = "\n".join(dialogue_history)
                        scores = judge_tau(dial_history, user_query, db_results, agent_response, policy, judge_client, judge_model)
                        turn_responses.append({
                            "turn": turn_ind,
                            "conversation_history": dial_history,
                            "user": user_query,
                            "db": db_call,
                            "response": agent_response,
                            "scores": scores
                        })

                        tqdm.write("dialogue history: " + "\n".join(dialogue_history))
                        tqdm.write(user_query)
                        tqdm.write("db call: " + db_results)
                        tqdm.write(agent_response)

                        dialogue_history.append(user_query)
                        dialogue_history.append(agent_response)
                        # reset variables
                        user_query = ""
                        db_call = {}
                        db_results = ""
                        agent_response = ""
            judge_scores[dataset[dial_ind]["task_id"]] = turn_responses
    except Exception as e:
        print("error: ", e)
        return judge_scores, False
    return judge_scores, True

def evaluate_tau_react(judge_client, judge_model, dataset_path):
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    judge_scores = {}
    try:
        for dial_ind in tqdm(range(len(dataset))):
            dial = dataset[dial_ind]['traj']
            policy = ""
            dialogue_history = []
            user_query = ""
            db_call = {}
            db_results = ""
            agent_response = ""
            turn_responses = []
            for i in tqdm(range(len(dial))):
                turn = dial[i]
                if i == 0:
                    if turn['role'] != "system":
                        tqdm.write('Format Error: System policy should be first')
                        exit()
                    else: 
                        policy = turn['content']
                else:
                    content = turn['content']
                    content = content.replace("\n", "")
                    if turn['role'] == 'user': 
                        user_str_split = "API output:"
                        if user_str_split in content:
                            api_output = content.split(user_str_split, 1)[1]
                            # check if api output is empty
                            if api_output.isspace():
                                api_output = "{}"
                            db_call_len = len(db_call[i-1])
                            db_call[i-1][db_call_len-1]['output'] = api_output
                        else:
                            user_query = f"Customer: {turn['content']}"
                    elif turn['role'] == 'assistant':
                        thought_str = "Thought:"
                        act_str = "Action:"
                        # model can hallucinate multiple actions apparently...
                        assist_actions = content.split(thought_str)
                        for action in assist_actions:
                            if len(action.strip()) == 0:
                                continue
                            action_split = action.split(act_str, 1)
                            thought = action_split[0]
                            tqdm.write("act:\n" + action_split[1])
                            act = json.loads(action_split[1])
                            if "respond" not in act['name']:
                                if i in db_call:
                                    db_call[i].append({'name': act['name'], 'args': json.dumps(act['arguments']), 'thought': thought})
                                else: 
                                    db_call[i] = [{'name': act['name'], 'args': json.dumps(act['arguments']), 'thought': thought}]
                            else:
                                agent_response += f"Agent: {act['arguments']['content']}"
                                # convert db call to string
                                for turn, funcs in db_call.items():
                                    for f in funcs:
                                        db_results += f"Thought: {f['thought']}\nFunction: {f['name']}\nArgs: {f['args']}"
                                        if "output" in f:
                                            db_results += f"\n Output: {f['output']}\n\n"
                                        else:
                                            db_results += "\n\n"

                                dial_history = "\n".join(dialogue_history)
                                scores = judge_tau(dial_history, user_query, db_results, agent_response, policy, judge_client, judge_model)
                                turn_responses.append({
                                    "turn": i,
                                    "conversation_history": dial_history,
                                    "user": user_query,
                                    "db": db_call,
                                    "response": agent_response,
                                    "scores": scores
                                })
                                # print
                                tqdm.write("dialogue history: " + "\n".join(dialogue_history))
                                tqdm.write(user_query)
                                tqdm.write("db call: " + db_results)
                                tqdm.write(agent_response)
                                # reset turn
                                dialogue_history.append(user_query)
                                dialogue_history.append(agent_response)
                                # reset variables
                                user_query = ""
                                db_call = {}
                                db_results = ""
                                agent_response = ""
            judge_scores[dataset[dial_ind]["task_id"]] = turn_responses
    except Exception as e:
        print("error: ", e)
        return judge_scores, False
    return judge_scores, True

def main(dataset_path, judge_client, judge_model, is_react):
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
    if is_react:
        scores, isJudgeSuccess = evaluate_tau_react(judge_client_obj, judge_model, dataset_path)
    else:
        scores, isJudgeSuccess = evaluate_tau_tool_call(judge_client_obj, judge_model, dataset_path)
    judge_metadata = {
        "filename": dataset_path,
        "judge_client": judge_client,
        "judge_model": judge_model
    }
    judge_output = {
        "metadata": judge_metadata,
        "dialogues": scores
    }

    result_dir = os.path.join('results', 'tau_judge_results')
    result_dir = os.path.join(result_dir, timestamp)
    os.makedirs(result_dir, exist_ok=True)
    judge_fname = f"tau-{judge_model}_j.json"
    full_result_path = os.path.join(result_dir, judge_fname)
    if not isJudgeSuccess:
        fname_split = full_result_path.split(".")
        full_result_path = f"{fname_split[0]}_CRASHED.{fname_split[1]}"
    with open(full_result_path, 'w') as f:
        json.dump(judge_output, f, indent=4, ensure_ascii=False)
    return result_dir, full_result_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate tau-bench TOD agent')
    parser.add_argument('--dataset_path', type=str, default='datasets/tau_airline_gpt4o.json', help='Path to evaluation data')
    parser.add_argument('--judge_client', type=str, default='openai', help='Client to use for LLM judge agent')
    parser.add_argument('--judge_model', type=str, default='gpt-4o', help='Agent to use for evaluation')
    parser.add_argument('--is_react', action='store_true', help='Flag to judge as react, if not set default to tool calling')
    args = parser.parse_args()

    if args.is_react:
        result_dir, full_result_path = main(args.dataset_path, args.judge_client, args.judge_model, True)
    else:
        result_dir, full_result_path = main(args.dataset_path, args.judge_client, args.judge_model, False)

    postprocess_results(full_result_path, result_dir)

    print("Evaluation completed successfully...")
    print(f"Results saved to {full_result_path}")
import argparse
from datetime import datetime
import json
import os
from tqdm import tqdm
from evaluator import judge_tau
from llm_agents import anthropic_agent, mistral_agent, openai_agent, togetherai_agent
from postprocess import postprocess_results

def evaluate_tau(judge_client, judge_model, dataset_path):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate tau-bench TOD agent')
    parser.add_argument('--dataset_path', type=str, default='datasets/tau_airline_gpt4o.json', help='Path to evaluation data')
    parser.add_argument('--judge_client', type=str, default='openai', help='Client to use for LLM judge agent')
    parser.add_argument('--judge_model', type=str, default='gpt-4o', help='Agent to use for evaluation')
    args = parser.parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # set up TOD judge agent
    if args.judge_client == 'openai':
        judge_client_obj = openai_agent
    elif args.judge_client == 'togetherai':
        judge_client_obj = togetherai_agent
    elif args.judge_client == 'mistral':
        judge_client_obj = mistral_agent
    elif args.judge_client == 'anthropic':
        judge_client_obj = anthropic_agent
    else:
        raise ValueError("Invalid client")

    scores, isJudgeSuccess = evaluate_tau(judge_client_obj, args.judge_model, args.dataset_path)
    judge_metadata = {
        "filename": args.dataset_path,
        "judge_client": args.judge_client,
        "judge_model": args.judge_model
    }
    judge_output = {
        "metadata": judge_metadata,
        "dialogues": scores
    }

    result_dir = os.path.join('results', 'tau_judge_results')
    result_dir = os.path.join(result_dir, timestamp)
    os.makedirs(result_dir, exist_ok=True)
    judge_fname = f"tau-{args.judge_model}_j.json"
    full_result_path = os.path.join(result_dir, judge_fname)
    if not isJudgeSuccess:
        fname_split = full_result_path.split(".")
        full_result_path = f"{fname_split[0]}_CRASHED.{fname_split[1]}"
    with open(full_result_path, 'w') as f:
        json.dump(judge_output, f, indent=4, ensure_ascii=False)
    postprocess_results(full_result_path, result_dir)

    print("Evaluation completed successfully...")
    print(f"Results saved to {full_result_path}")
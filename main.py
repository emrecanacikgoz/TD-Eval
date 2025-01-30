import json
import argparse
from tqdm import tqdm
import os
import re
from datetime import datetime
from llm_agents import openai_agent, togetherai_agent, mistral_agent, anthropic_agent
import random
from time import sleep
from typing import Dict
import sys
import copy
from random import sample

# need to add MultiWOZ_Evaluation to sys.path for absolute imports
sys.path.insert(0, os.path.abspath("./MultiWOZ_Evaluation"))

from prompts.mwoz_agent_prompts import mwz_domain_prompt, MWZ_DOMAIN_RESPONSE_PROMPTS, MWZ_DOMAIN_STATE_PROMPTS, MWZ_DOMAIN_DELEX_PROMPTS
from evaluator import judge_mwoz
from postprocess import postprocess_results
from mw_database import MultiWOZDatabase
from MultiWOZ_Evaluation.mwzeval.metrics import Evaluator

def parse_state(state: str, default_domain: str = None) -> Dict[str, str]:
    def sanitize(dct):
        for key in dct:
            if isinstance(dct[key], dict):
                dct[key] = sanitize(dct[key])
            elif not isinstance(dct[key], str):
                dct[key] = str(dct[key])
        return dct

    state = str(state)
    slotvals = re.findall("('[a-z]+': ?('(([a-z]| |[A-Z]|:|[0-9])+')|[A-Za-z0-9:]+))", state)
    out_state = {}
    for sv in slotvals:
        sv = sv[0].strip("'\"").split(':')
        out_state[sv[0].strip("'\"")] = ":".join(sv[1:]).strip("'\" ")
    return sanitize(out_state)
    # if not state.startswith("{"):
    #     state = "{" + state
    # if not state.endswith("}"):
    #     state = state + '}'
    # state = state.replace('<', '{').replace('>', '}')
    # try:
    #     state = dirtyjson.loads(state)
    #     try:
    #         for domain, domain_state in state.items():
    #             for slot, value in domain_state.items():
    #                 pass

    #         return sanitize(state)
    #     except:
    #         return {default_domain: sanitize(state)}

    # except:
    #     state = str(state)
    #     if state.count('{') == 1:
    #         state = '{ ' + default_domain + ' ' + state
    #     state_tk = word_tokenize(state)
    #     # filter only tokens that are alphanumeric or braces
    #     state_tk = [tk for tk in state_tk if tk.isalpha() or tk in ['{', '}',',']]
    #     parsed_state = {default_domain: {}}
    #     level = 0
    #     current_domain = default_domain 
    #     idx = 0
    #     while idx < len(state_tk):
    #         tk = state_tk[idx]
    #         if tk == '{':
    #             # level += 1
    #             pass
    #         elif tk == '}':
    #             # level -= 1
    #             pass
    #         # elif level == 1:
    #         #     current_domain = tk
    #         #     parsed_state[tk] = {}
    #         else:
    #             slot = tk
    #             value = []
    #             idx += 1
    #             if idx >= len(state_tk):
    #                 break
    #             while state_tk[idx] not in  [',', '}']:
    #                 value.append(state_tk[idx])
    #                 idx += 1
    #                 if idx >= len(state_tk):
    #                     break
    #             parsed_state[current_domain][slot] = ' '.join(value)
    #         idx += 1
    #         if idx >= len(state_tk):
    #             break
    #     return sanitize(parsed_state)

def gen_conv_agent_results(evaluation_data_path, use_gt_state, agent_client_obj, agent_model):
    print('use_gt_state', use_gt_state)
    # load data with batch
    with open('datasets/batch.json', 'r') as fBatch:
        batch = json.load(fBatch)
    with open(evaluation_data_path, 'r') as f:
        data = []
        for line in f:
            dialog = json.loads(line)
            if dialog['dialogue_id'].split(".json")[0].lower() in batch:
                data.append(dialog)
    # run agent and judge simulator
    dialogue_responses = {}
    database = MultiWOZDatabase("./multiwoz_database")
    for idx_ in tqdm(range(len(data))):
        dial = data[idx_]
        dialogue_id = dial['dialogue_id']
        services = [domain.lower() for domain in dial['services']]
        turns = dial['turns']
        # get turn dialogue
        speaker = turns['speaker']
        utterance = turns['utterance']
        frames = turns['frames'] 
        assert len(speaker) == len(utterance) == len(frames), "Length mismatch in dialogue turns"
        # get responses
        conversation_history = []
        dial_state = {}
        db_results = {}
        turn_responses = []
        try:
            for i in tqdm(range(0, len(speaker), 2)):
                user_query = utterance[i]
                ground_truth = utterance[i+1] if i+1 < len(utterance) else None
                current_history = "\n".join(conversation_history)
                domain_prompt = mwz_domain_prompt.format(history=current_history, utterance=user_query)
                domain_output = agent_client_obj(domain_prompt, agent_model)
                domain = domain_output.lower() if domain_output in services else random.choice(services)
                # state tracking (both gt and generation option)
                if use_gt_state:
                    turn_state = frames[i]["state"]
                    if len(turn_state) == 0:
                        dial_state[domain] = {}
                    else:
                        turn_state = turn_state[0]['slots_values']
                        turn_state = {k: v[0] for k, v in zip(turn_state['slots_values_name'], turn_state['slots_values_list'])}
                        for k, v in turn_state.items():
                            slot_name = k
                            slot_val = v
                            slot_domain = domain
                            slot_domain, slot_name = k.split('-')
                            if slot_domain in dial_state:
                                dial_state[slot_domain][slot_name] = slot_val
                            else:
                                dial_state[slot_domain] = {slot_name: slot_val}
                else:  
                    state_prompt = MWZ_DOMAIN_STATE_PROMPTS[domain].format(history=current_history, utterance=user_query)
                    state_output = agent_client_obj(state_prompt, agent_model)
                    turn_state = parse_state(state_output.lower(), default_domain=domain)
                    for k, v in turn_state.items():
                        if domain in dial_state:
                            dial_state[domain][k] = v
                        else:
                            dial_state[domain] = {k: v}
                    # if no state to parse then create domain in state
                    if domain not in dial_state:
                        dial_state[domain] = {}
                # database retrieval
                if not turn_state:
                    db_results[domain] = []
                retrieved_items = {domain: database.query(domain, domain_state) for domain, domain_state in dial_state.items()}
                for domain, domain_results in retrieved_items.items():
                    if len(domain_results) > 10:
                        result_sample = sample(domain_results, 5)
                        turn_db_result = {"count": len(domain_results), "sample": result_sample}
                    else: 
                        turn_db_result = {"count": len(domain_results), "results": domain_results}
                    db_results[domain] = turn_db_result
                # response retrieval
                response_prompt = MWZ_DOMAIN_RESPONSE_PROMPTS[domain].format(history=current_history, utterance=user_query, state=dial_state[domain], database=db_results[domain])
                agent_response = agent_client_obj(response_prompt, agent_model)
                if agent_response == "": # throw an error
                    raise Exception("token limit hit")
                conversation_history.append(f"Customer: {user_query}")
                conversation_history.append(f"Agent: {agent_response}")
                # delexicalize response
                delex_prompt = MWZ_DOMAIN_DELEX_PROMPTS[domain.lower()].format(response=agent_response)
                delex_response = agent_client_obj(delex_prompt, agent_model)
                turn_responses.append({
                    "turn": i,
                    "conversation_history": current_history,
                    "user": user_query,
                    "domain": domain,
                    "state": copy.deepcopy(dial_state),
                    "db": copy.deepcopy(db_results),
                    "lex_response": agent_response,
                    "response": delex_response,
                    "ground_truth": ground_truth
                })
                if idx_ < 10:
                    tqdm.write('user query: ' + user_query)
                    tqdm.write('domain: ' + domain.lower())
                    tqdm.write("parsed state: " + json.dumps(turn_state))
                    tqdm.write('db: ' + json.dumps(db_results))
                    tqdm.write("agent_response: " + agent_response)
                    tqdm.write("delex_agent: " + delex_response)
                sleep(5)
            # Compile complete dialogue
            format_dial_id = dialogue_id.split(".json")[0].lower()
            dialogue_responses[format_dial_id] = turn_responses
        except Exception as e:
            tqdm.write('error:' + str(e))
            return dialogue_responses, False
    return dialogue_responses, True

def judge_conv_agent_results(conv_agent_data, judge_client_obj, judge_model):
    agent_dials = conv_agent_data["dialogues"].items()
    judge_scores = {} #[]
    idx_ = 0
    for dialogue_id, dialogue_turns in tqdm(agent_dials):
        try:
            turn_responses = []
            for i in tqdm(range(len(dialogue_turns))):
                curr_turn = dialogue_turns[i]
                conversation_history = curr_turn["conversation_history"]
                domain = curr_turn["domain"]
                state = curr_turn["state"]
                db_result = curr_turn["db"]
                user_query = curr_turn["user"]
                lex_response = curr_turn["lex_response"]
                delex = curr_turn["response"]
                ground_truth = curr_turn["ground_truth"]
                scores = judge_mwoz(conversation_history, domain, user_query, json.dumps(db_result), lex_response, judge_client_obj, judge_model)
                # record judge score
                turn_responses.append({
                    "turn": i,
                    "conversation_history": conversation_history,
                    "user": user_query,
                    "domain": domain,
                    "state": state,
                    "db": db_result,
                    "lex_response": lex_response,
                    "response": delex, 
                    "ground_truth": ground_truth,
                    "scores": scores
                })
                if idx_ < 10:
                    tqdm.write("agent_response: " + lex_response)
                    tqdm.write("conv_consistency: " + str(scores['conv_consistency']))
                    tqdm.write("backend_consistency: " + str(scores['backend_consistency']))
                    tqdm.write("policy_completeness: " + str(scores['policy_completeness']))
                sleep(5)
            # Compile complete scores for dialogue
            format_dial_id = dialogue_id.split(".json")[0].lower()
            judge_scores[format_dial_id] = turn_responses
            idx_ += 1
        except Exception as e:
            tqdm.write("error: " + str(e))
            return judge_scores, False
    return judge_scores, True

def main(agent_client, agent_model, use_gt_state, judge_client, judge_model, dataset_path, agent_result_path):
    # set up TOD response agent 
    if agent_client == 'openai':
        agent_client_obj = openai_agent
    elif agent_client == 'togetherai':
        agent_client_obj = togetherai_agent
    elif agent_client == 'mistral':
        agent_client_obj = mistral_agent
    elif agent_client == 'anthropic':
        agent_client_obj = anthropic_agent
    else:
        raise ValueError("Invalid client")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # get agent dialogue responses
    if agent_result_path:
        with open(agent_result_path, 'r') as fResult:
            dial_output = json.load(fResult)
    else:    
        agent_metadata = {
            "agent_client": agent_client,
            "agent_model": agent_model
        }
        dial_responses, isAgentSuccess = gen_conv_agent_results(dataset_path, use_gt_state, agent_client_obj, agent_model)
        dial_output = {
            "metadata": agent_metadata,
            "dialogues": dial_responses
        }
        # save dialogue response results
        result_dir = os.path.join('results', 'agents_results')
        result_dir = os.path.join(result_dir, timestamp)
        os.makedirs(result_dir, exist_ok=True)
        agent_fname = f"{agent_model}_c.json"
        full_result_path = os.path.join(result_dir, agent_fname)
        if not isAgentSuccess:
            fname_split = full_result_path.split(".")
            full_result_path = f"{fname_split[0]}_CRASHED.{fname_split[1]}"
        with open(full_result_path, 'w') as f:
            json.dump(dial_output, f, indent=4, ensure_ascii=False)
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
    judge_metadata = {
        "judge_client": judge_client,
        "judge_model": judge_model,
        "agent_client": agent_client,
        "agent_model": agent_model,
    }
    scores, isJudgeSuccess = judge_conv_agent_results(dial_output, judge_client_obj, judge_model)
    e = Evaluator(bleu=True, success=True, richness=True)
    eval_results = e.evaluate(scores)
    judge_metadata['inform'] = eval_results['success']['inform']
    judge_metadata['success'] = eval_results['success']['success']
    # save dialogue judge results
    dial_output = {
        "metadata": judge_metadata,
        "dialogues": scores,
        "mwzeval": eval_results['success']['dialogs']
    }
    result_dir = os.path.join('results', 'judge_results')
    result_dir = os.path.join(result_dir, timestamp)
    os.makedirs(result_dir, exist_ok=True)
    judge_fname = f"{agent_model}_c-{judge_model}_j.json"
    full_result_path = os.path.join(result_dir, judge_fname)
    if not isJudgeSuccess:
        fname_split = full_result_path.split(".")
        full_result_path = f"{fname_split[0]}_CRASHED.{fname_split[1]}"
    with open(full_result_path, 'w') as f:
        json.dump(dial_output, f, indent=4, ensure_ascii=False)
    # return resulting directory and file
    return result_dir, full_result_path

if __name__ == "__main__":
    default_result_filename = 'zero-shot-results.json'
    parser = argparse.ArgumentParser(description='Evaluate dialogue agent')
    parser.add_argument('--dataset_path', type=str, default='datasets/woz_only.jsonl', help='Path to evaluation data')
    parser.add_argument('--agent_client', type=str, default='openai', help='Client to use for LLM agent')
    parser.add_argument('--agent_model', type=str, default='gpt-4o', help='Agent to evaluate')
    parser.add_argument('--agent_result_path', type=str, help='File path to already generated agent results (optional)')
    parser.add_argument('--use_gt_state', action='store_true', help='Uses ground truth state of multiwoz corpus (for debug)')
    parser.add_argument('--judge_client', type=str, default='openai', help='Client to use for LLM judge agent')
    parser.add_argument('--judge_model', type=str, default='gpt-4o', help='Agent to use for evaluation')
    args = parser.parse_args()
    # run scoring + judging then post-process
    result_dir, full_result_path = main(args.agent_client, args.agent_model, args.use_gt_state, args.judge_client, args.judge_model, args.dataset_path, args.agent_result_path)
    postprocess_results(full_result_path, result_dir)

    print("Evaluation completed successfully...")
    print(f"Results saved to {full_result_path}")
import json
import os
import re
import random
from tqdm import tqdm
import copy
from random import sample
from typing import Dict, Tuple, Any
import sys

# Add necessary paths for imports
sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(0, os.path.abspath("../MultiWOZ_Evaluation"))

from generate.llm_agents import openai_agent, togetherai_agent, mistral_agent, anthropic_agent
from generate.mw_database import MultiWOZDatabase
from generate.prompts.mwoz_agent_prompts import mwz_domain_prompt, MWZ_DOMAIN_RESPONSE_PROMPTS, MWZ_DOMAIN_STATE_PROMPTS, MWZ_DOMAIN_DELEX_PROMPTS


def parse_state(state: str, default_domain: str = None) -> Dict[str, str]:
    """Parse state string into a structured dictionary format."""
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


def gen_conv_agent_results(evaluation_data_path, use_gt_state, agent_client_obj, agent_model):
    """Generate agent responses for evaluation data."""
    print('use_gt_state', use_gt_state)
    # load data with batch
    with open(os.path.join('datasets', 'batch.json'), 'r') as fBatch:
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
            dialogue_responses[dialogue_id] = turn_responses
        except Exception as e:
            tqdm.write("error: " + str(e))
            return dialogue_responses, False
    return dialogue_responses, True


def get_agent_client(agent_client: str):
    """Get the appropriate agent client function based on provider name."""
    if agent_client == 'openai':
        return openai_agent
    elif agent_client == 'togetherai':
        return togetherai_agent
    elif agent_client == 'mistral':
        return mistral_agent
    elif agent_client == 'anthropic':
        return anthropic_agent
    else:
        raise ValueError(f"Invalid client: {agent_client}")


def generate_agent_responses(agent_client: str, agent_model: str, 
                             use_gt_state: bool, dataset_path: str) -> Tuple[dict, str, bool]:
    """Generate responses using the specified agent and save the results."""
    from datetime import datetime
    
    # Set up TOD response agent
    agent_client_obj = get_agent_client(agent_client)
    
    # Get timestamp for result directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Get agent dialogue responses
    agent_metadata = {
        "agent_client": agent_client,
        "agent_model": agent_model
    }
    
    dial_responses, isAgentSuccess = gen_conv_agent_results(
        dataset_path, use_gt_state, agent_client_obj, agent_model
    )
    
    dial_output = {
        "metadata": agent_metadata,
        "dialogues": dial_responses
    }
    
    # Create result directory and save results
    result_dir = os.path.join('results', 'agents_results')
    result_dir = os.path.join(result_dir, timestamp)
    os.makedirs(result_dir, exist_ok=True)
    
    agent_fname = f"{agent_model}_c.json"
    # Replace slashes in model name with underscore to prevent file create error
    agent_fname = agent_fname.replace("/", "_")
    full_result_path = os.path.join(result_dir, agent_fname)
    
    if not isAgentSuccess:
        fname_split = full_result_path.split(".")
        full_result_path = f"{fname_split[0]}_CRASHED.{fname_split[1]}"
    
    with open(full_result_path, 'w') as f:
        json.dump(dial_output, f, indent=4, ensure_ascii=False)
    
    return dial_output, full_result_path, isAgentSuccess


if __name__ == "__main__":
    # This can be used for standalone testing
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate agent responses')
    parser.add_argument('--dataset_path', type=str, default='datasets/woz_only.jsonl', 
                        help='Path to evaluation data')
    parser.add_argument('--agent_client', type=str, default='openai', 
                        help='Client to use for LLM agent')
    parser.add_argument('--agent_model', type=str, default='gpt-4o', 
                        help='Agent to evaluate')
    parser.add_argument('--use_gt_state', action='store_true', 
                        help='Uses ground truth state of multiwoz corpus (for debug)')
    
    args = parser.parse_args()
    
    dial_output, result_path, success = generate_agent_responses(
        args.agent_client, args.agent_model, args.use_gt_state, args.dataset_path
    )
    
    print("Generation completed successfully...")
    print(f"Results saved to {result_path}") 
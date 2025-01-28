from prompts.mwoz_judge_prompts import mwoz_prompt_conv_consistency, mwoz_prompt_backend_consistency, mwoz_prompt_restaurant_policy_compliance, mwoz_prompt_hotel_policy_compliance, mwoz_prompt_attraction_policy_compliance, mwoz_prompt_taxi_policy_compliance, mwoz_prompt_train_policy_compliance, prompt_unknown_policy_compliance
from prompts.tau_judge_prompts import tau_prompt_conv_consistency, tau_prompt_backend_consistency, tau_prompt_policy_compliance

def judge_mwoz_conv_consistency(dialogue_history, user_query, db_result, agent_response, llm_client, client_model):
    formatted_prompt = mwoz_prompt_conv_consistency.format(dialogue_history=dialogue_history, user_query=user_query, db_result=db_result, agent_response=agent_response)
    judge_output = llm_client(formatted_prompt, client_model)
    judge_split = judge_output.split('Justification:')
    return {
        "score": judge_split[0].strip(),
        "justification": judge_split[1]
    }

def judge_mwoz_backend_consistency(dialogue_history, user_query, db_result, agent_response, llm_client, client_model):
    formatted_prompt = mwoz_prompt_backend_consistency.format(dialogue_history=dialogue_history, user_query=user_query, db_result=db_result, agent_response=agent_response)
    judge_output = llm_client(formatted_prompt, client_model)
    judge_split = judge_output.split('Justification:')
    return {
        "score": judge_split[0].strip(),
        "justification": judge_split[1]
    }

def judge_mwoz_policy_completeness(dialogue_history, domain, user_query, db_result, agent_response, llm_client, client_model):
    if domain == "restaurant":
        formatted_prompt = mwoz_prompt_restaurant_policy_compliance.format(dialogue_history=dialogue_history, user_query=user_query, db_result=db_result, agent_response=agent_response)
    elif domain == "hotel":
        formatted_prompt = mwoz_prompt_hotel_policy_compliance.format(dialogue_history=dialogue_history, user_query=user_query, db_result=db_result, agent_response=agent_response)
    elif domain == "attraction":
        formatted_prompt = mwoz_prompt_attraction_policy_compliance.format(dialogue_history=dialogue_history, user_query=user_query, db_result=db_result, agent_response=agent_response)
    elif domain == "taxi":
        formatted_prompt = mwoz_prompt_taxi_policy_compliance.format(dialogue_history=dialogue_history, user_query=user_query, db_result=db_result, agent_response=agent_response)
    elif domain == "train":
        formatted_prompt = mwoz_prompt_train_policy_compliance.format(dialogue_history=dialogue_history, user_query=user_query, db_result=db_result, agent_response=agent_response)
    else:
        formatted_prompt = prompt_unknown_policy_compliance.format(dialogue_history=dialogue_history, user_query=user_query, db_result=db_result, agent_response=agent_response)
    judge_output = llm_client(formatted_prompt, client_model)
    judge_split = judge_output.split('Justification:')
    return {
        "score": judge_split[0].strip(),
        "justification": judge_split[1]
    }

def judge_mwoz(turn_history, domain, user_query, db_result, agent_response, llm_client, client_model):
    # conversation consistency score: should be number (and justification)?
    conv_consistency_score = judge_mwoz_conv_consistency(turn_history, user_query, db_result, agent_response, llm_client, client_model)
    # backend consistency score: should be number (and justification)?
    backend_consistency_score = judge_mwoz_backend_consistency(turn_history, user_query, db_result, agent_response, llm_client, client_model)
    # policy completeness score: should be number (and justification)?
    policy_completeness_score = judge_mwoz_policy_completeness(turn_history, domain, user_query, db_result, agent_response, llm_client, client_model)
    # return as as a nested dict
    return {
        "conv_consistency": conv_consistency_score, 
        "backend_consistency": backend_consistency_score, 
        "policy_completeness": policy_completeness_score
    }

def judge_tau_conv_consistency(dialogue_history, user_query, db_result, agent_response, llm_client, client_model):
    formatted_prompt = tau_prompt_conv_consistency.format(dialogue_history=dialogue_history, user_query=user_query, db_result=db_result, agent_response=agent_response)
    judge_output = llm_client(formatted_prompt, client_model)
    judge_split = judge_output.split('Justification:')
    return {
        "score": judge_split[0].strip(),
        "justification": judge_split[1]
    }

def judge_tau_backend_consistency(dialogue_history, user_query, db_result, agent_response, llm_client, client_model):
    formatted_prompt = tau_prompt_backend_consistency.format(dialogue_history=dialogue_history, user_query=user_query, db_result=db_result, agent_response=agent_response)
    judge_output = llm_client(formatted_prompt, client_model)
    judge_split = judge_output.split('Justification:')
    return {
        "score": judge_split[0].strip(),
        "justification": judge_split[1]
    }

def judge_tau_policy_completeness(dialogue_history, user_query, db_result, agent_response, policy, llm_client, client_model):
    formatted_prompt = tau_prompt_policy_compliance.format(policy=policy, dialogue_history=dialogue_history, user_query=user_query, db_result=db_result, agent_response=agent_response)
    judge_output = llm_client(formatted_prompt, client_model)
    judge_split = judge_output.split('Justification:')
    return {
        "score": judge_split[0].strip(),
        "justification": judge_split[1]
    }

def judge_tau(turn_history, user_query, db_result, agent_response, policy, llm_client, client_model):
    # conversation consistency score
    conv_consistency_score = judge_tau_conv_consistency(turn_history, user_query, db_result, agent_response, llm_client, client_model)
    # backend consistency score
    backend_consistency_score = judge_tau_backend_consistency(turn_history, user_query, db_result, agent_response, llm_client, client_model)
    # policy completeness score
    policy_completeness_score = judge_tau_policy_completeness(turn_history, user_query, db_result, agent_response, policy, llm_client, client_model)
    # return as as a nested dict
    return {
        "conv_consistency": conv_consistency_score, 
        "backend_consistency": backend_consistency_score, 
        "policy_completeness": policy_completeness_score
    }
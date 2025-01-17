from prompts.judge_prompts import prompt_conv_consistency, prompt_backend_consistency, prompt_policy_restaurant_completeness, prompt_policy_hotel_completeness, prompt_policy_attraction_completeness, prompt_policy_taxi_completeness, prompt_policy_train_completeness, prompt_policy_unknown_completeness

def judge_conv_consistency(dialogue_history, user_query, db_result, agent_response, llm_client, client_model):
    formatted_prompt = prompt_conv_consistency.format(dialogue_history=dialogue_history, user_query=user_query, db_result=db_result, agent_response=agent_response)
    judge_output = llm_client(formatted_prompt, client_model)
    judge_split = judge_output.split('Justification:')
    return {
        "score": judge_split[0].strip(),
        "justification": judge_split[1]
    }

def judge_backend_consistency(dialogue_history, user_query, db_result, agent_response, llm_client, client_model):
    formatted_prompt = prompt_backend_consistency.format(dialogue_history=dialogue_history, user_query=user_query, db_result=db_result, agent_response=agent_response)
    judge_output = llm_client(formatted_prompt, client_model)
    judge_split = judge_output.split('Justification:')
    return {
        "score": judge_split[0].strip(),
        "justification": judge_split[1]
    }

def judge_policy_completeness(dialogue_history, domain, user_query, db_result, agent_response, llm_client, client_model):
    if domain == "restaurant":
        formatted_prompt = prompt_policy_restaurant_completeness.format(dialogue_history=dialogue_history, user_query=user_query, db_result=db_result, agent_response=agent_response)
    elif domain == "hotel":
        formatted_prompt = prompt_policy_hotel_completeness.format(dialogue_history=dialogue_history, user_query=user_query, db_result=db_result, agent_response=agent_response)
    elif domain == "attraction":
        formatted_prompt = prompt_policy_attraction_completeness.format(dialogue_history=dialogue_history, user_query=user_query, db_result=db_result, agent_response=agent_response)
    elif domain == "taxi":
        formatted_prompt = prompt_policy_taxi_completeness.format(dialogue_history=dialogue_history, user_query=user_query, db_result=db_result, agent_response=agent_response)
    elif domain == "train":
        formatted_prompt = prompt_policy_train_completeness.format(dialogue_history=dialogue_history, user_query=user_query, db_result=db_result, agent_response=agent_response)
    else:
        formatted_prompt = prompt_policy_unknown_completeness.format(dialogue_history=dialogue_history, user_query=user_query, db_result=db_result, agent_response=agent_response)
    judge_output = llm_client(formatted_prompt, client_model)
    judge_split = judge_output.split('Justification:')
    return {
        "score": judge_split[0].strip(),
        "justification": judge_split[1]
    }

def judge(turn_history, domain, user_query, db_result, agent_response, llm_client, client_model):
    # conversation consistency score: should be number (and justification)?
    conv_consistency_score = judge_conv_consistency(turn_history, user_query, db_result, agent_response, llm_client, client_model)
    # backend consistency score: should be number (and justification)?
    backend_consistency_score = judge_backend_consistency(turn_history, user_query, db_result, agent_response, llm_client, client_model)
    # policy completeness score: should be number (and justification)?
    policy_completeness_score = judge_policy_completeness(turn_history, domain, user_query, db_result, agent_response, llm_client, client_model)
    # return as as a nested dict
    return {
        "conv_consistency": conv_consistency_score, 
        "backend_consistency": backend_consistency_score, 
        "policy_completeness": policy_completeness_score
    }

import json
import random
import math
import argparse
import os
import time
from tqdm import tqdm
from llm_agents import openai_agent, togetherai_agent, anthropic_agent, mistral_agent
import numpy as np
from collections import OrderedDict, defaultdict
from scipy import stats

def calculate_elo_change(rating_a, rating_b, score, k=32):    
    expected_a = 1 / (1 + 10**((rating_b - rating_a)/400))
    change_a = k * (score - expected_a)
    return (change_a, -change_a)

def calculate_confidence_interval(elo_history, confidence=0.95):
    if len(elo_history) < 2:
        return (0, 0)
    std_dev = np.std(elo_history)
    z_score = stats.norm.ppf((1 + confidence) / 2)
    margin_error = z_score * (std_dev / math.sqrt(len(elo_history)))
    
    final_elo = elo_history[-1]
    return (final_elo - margin_error, final_elo + margin_error), z_score

def get_judge_comparison(conv_a, conv_b, judge_agent, max_retries=5, retry_delay=60):
    conv_a_formatted = "\n".join([f"User: {turn['user']}\nAssistant: {turn['agent']}" 
                                for turn in conv_a])
    conv_b_formatted = "\n".join([f"User: {turn['user']}\nAssistant: {turn['agent']}" 
                                for turn in conv_b])
    
    # TODO: add a reference policy
    prompt = f"""Compare these two AI assistant conversations and determine which one is better. 
Consider the following aspects:

1. Conversation Consistency:
- **Relevance**: The response directly relates to the dialogue history and the current user query.
- **Topic Consistency**: The response remains on-topic with the dialogue history and the user query.
- **Coherence**: The response logically continues the progression of the dialogue.

2. Backend Knowledge Consistency:
- **Relevance**: The response directly reflects the information in the database results.
- **Topic Consistency**: The response stays on-topic with the database results and the dialogue context.
- **Coherence**: The response logically incorporates and progresses based on the database results.

3. Policy Completeness:
- **Relevance of Suggestions**: Providing suggestions only when the database results are small enough to do so.
- **Information Gathering**: Requesting required, relevant information (slots) from the user before offering suggestions or booking services.
- **Appropriate Timing**: Avoiding premature actions, such as making a booking or suggesting a service too early in the conversation.

Conversation A:
{conv_a_formatted}

Conversation B:
{conv_b_formatted}

Which conversation was better? Answer with only:
CONVERSATION_A if Conversation A was better
CONVERSATION_B if Conversation B was better
EQUAL if they were roughly equivalent
"""

    for attempt in range(max_retries):
        try:
            response = judge_agent(prompt)
            result = None
            if "CONVERSATION_A" in response.upper():
                result = 1.0
            elif "CONVERSATION_B" in response.upper():
                result = 0.0
            else:
                result = 0.5
            return result, response, conv_a_formatted, conv_b_formatted
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt failed: {str(e)}")
                print(f"Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
            else:
                raise Exception(f"Failed after {max_retries} attempts: {str(e)}")

def load_conversations(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    print(f"File {filename}: {len(data['dialogues'])} conversations")
    return data

def run_elo_tournament(input_files, judge_name, existing_results=None, total_rounds=None):
    print("\nLoading conversation files:")
    
    if existing_results:
        model_elos = {**{f: 1000.0 for f in input_files}, 
                     **{k: v['elo'] for k, v in existing_results.items()}}
        model_elo_history = {**{f: [1000.0] for f in input_files},
                           **{k: v['elo_history'] for k, v in existing_results.items()}}
        model_total_votes = {**{f: 0 for f in input_files},
                           **{k: v['total_votes'] for k, v in existing_results.items()}}
        model_picked_votes = {**{f: 0 for f in input_files},
                            **{k: v['picked_votes'] for k, v in existing_results.items()}}
        model_ties = {**{f: 0 for f in input_files},
                     **{k: v['ties'] for k, v in existing_results.items()}}
        model_wins = {**{f: 0 for f in input_files},
                     **{k: v['wins'] for k, v in existing_results.items()}}
        model_losses = {**{f: 0 for f in input_files},
                       **{k: v['losses'] for k, v in existing_results.items()}}
        head_to_head = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'losses': 0, 'ties': 0}))
        elo_changes = defaultdict(list)
        
        if 'head_to_head' in existing_results[next(iter(existing_results))]:
            for model in existing_results:
                for opponent, stats in existing_results[model]['head_to_head'].items():
                    head_to_head[model][opponent] = stats
                elo_changes[model] = existing_results[model].get('elo_changes', [])
        
        all_models = list(input_files) + list(existing_results.keys())
    else:
        model_elos = {f: 1000.0 for f in input_files}
        model_elo_history = {f: [1000.0] for f in input_files}
        model_total_votes = {f: 0 for f in input_files}
        model_picked_votes = {f: 0 for f in input_files}
        model_ties = {f: 0 for f in input_files}
        model_wins = {f: 0 for f in input_files}
        model_losses = {f: 0 for f in input_files}
        head_to_head = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'losses': 0, 'ties': 0}))
        elo_changes = defaultdict(list)
        all_models = list(input_files)

    model_convs = {f: load_conversations(f) for f in input_files}
    for existing_model in existing_results or {}:
        if existing_model not in input_files:
            model_convs[existing_model] = load_conversations(existing_model)
    
    if judge_name.lower() == "openai":
        judge_agent = openai_agent
    elif judge_name.lower() == "anthropic":
        judge_agent = anthropic_agent
    elif judge_name.lower() == "mistral":
        judge_agent = mistral_agent
    elif judge_name.lower() == "togetherai":
        judge_agent = togetherai_agent
    else:
        raise ValueError(f"Unknown judge agent: {judge_name}")

    min_convs = min(len(convs['dialogues']) for convs in model_convs.values())
    print(f"\nUsing {min_convs} conversations per model (minimum across all files)")
    
    num_models = len(all_models)
    comparisons_per_round = len([(a, b) for idx, a in enumerate(all_models) 
                                for b in all_models[idx + 1:]
                                if (a in input_files or b in input_files)])
    
    if total_rounds is None:
        total_rounds = min_convs
    else:
        total_rounds = min(total_rounds, min_convs)
    
    total_comparisons = total_rounds * comparisons_per_round
    print(f"\nStarting tournament with {total_rounds} rounds ({comparisons_per_round} comparisons per round)")
    
    detailed_comparisons = []
    pbar = tqdm(total=total_comparisons)
    completed_rounds = 0
    
    while completed_rounds < total_rounds:
        pairs = [(a, b) for idx, a in enumerate(all_models) 
                for b in all_models[idx + 1:]
                if (a in input_files or b in input_files)]
        
        # Ensure same conversation index is used for all pairs in the round
        conv_idx = completed_rounds % min_convs
        
        for model_a, model_b in pairs:
            try:
                conv_a = model_convs[model_a]['dialogues'][conv_idx]['full_conversation']
                conv_b = model_convs[model_b]['dialogues'][conv_idx]['full_conversation']
                
                score, judge_response, conv_a_fmt, conv_b_fmt = get_judge_comparison(conv_a, conv_b, judge_agent)        
                change_a, change_b = calculate_elo_change(model_elos[model_a], model_elos[model_b], score)
                
                elo_changes[model_a].append(change_a)
                elo_changes[model_b].append(change_b)
                
                if score == 1.0:
                    head_to_head[model_a][model_b]['wins'] += 1
                    head_to_head[model_b][model_a]['losses'] += 1
                elif score == 0.0:
                    head_to_head[model_b][model_a]['wins'] += 1
                    head_to_head[model_a][model_b]['losses'] += 1
                else:
                    head_to_head[model_a][model_b]['ties'] += 1
                    head_to_head[model_b][model_a]['ties'] += 1
                
                comparison_detail = {
                    'round': completed_rounds + 1,
                    'model_a': model_a,
                    'model_b': model_b,
                    'conversation_index': conv_idx,
                    'judge_response': judge_response,
                    'score': score,
                    'elo_change_a': change_a,
                    'elo_change_b': change_b,
                    'conversation_a': conv_a_fmt,
                    'conversation_b': conv_b_fmt,
                    'elo_a_before': model_elos[model_a],
                    'elo_b_before': model_elos[model_b],
                    'elo_a_after': model_elos[model_a] + change_a,
                    'elo_b_after': model_elos[model_b] + change_b
                }
                detailed_comparisons.append(comparison_detail)
                
                model_elos[model_a] += change_a
                model_elos[model_b] += change_b

                model_elo_history[model_a].append(model_elos[model_a])
                model_elo_history[model_b].append(model_elos[model_b])
                
                model_total_votes[model_a] += 1
                model_total_votes[model_b] += 1
                
                if score == 1.0:
                    model_picked_votes[model_a] += 1
                    model_wins[model_a] += 1
                    model_losses[model_b] += 1
                elif score == 0.0:
                    model_picked_votes[model_b] += 1
                    model_wins[model_b] += 1
                    model_losses[model_a] += 1
                else:
                    model_picked_votes[model_a] += 0.5
                    model_picked_votes[model_b] += 0.5
                    model_ties[model_a] += 1
                    model_ties[model_b] += 1
                
                pbar.update(1)
                
            except Exception as e:
                print(f"\nError occurred: {str(e)}")
                print("Continuing with next comparison...")
                continue
        
        completed_rounds += 1
            
    pbar.close()

    results = {}
    extended_results = {}
    
    for model in all_models:
        for opponent in head_to_head[model]:
            stats = head_to_head[model][opponent]
            total_games = stats['wins'] + stats['losses'] + stats['ties']
            if total_games > 0:
                stats['win_rate'] = stats['wins'] / total_games
                stats['loss_rate'] = stats['losses'] / total_games
                stats['tie_rate'] = stats['ties'] / total_games
            else:
                stats['win_rate'] = 0
                stats['loss_rate'] = 0
                stats['tie_rate'] = 0
    
    confidence_level = 0.95
    
    for model in all_models:
        ci_range, z_score = calculate_confidence_interval(model_elo_history[model], confidence=confidence_level)
        total_games = model_wins[model] + model_losses[model] + model_ties[model]
        
        results[model] = {
            'elo': model_elos[model],
            'confidence_interval': list(ci_range),
            'total_votes': model_total_votes[model],
            'picked_votes': model_picked_votes[model],
            'wins': model_wins[model],
            'losses': model_losses[model],
            'ties': model_ties[model]
        }
        
        extended_results[model] = {
            **results[model],
            'elo_history': model_elo_history[model],
            'confidence_interval_details': {
                'range': list(ci_range),
                'z_score': z_score,
                'confidence_level': confidence_level
            },
            'win_rate': model_wins[model] / total_games if total_games > 0 else 0,
            'loss_rate': model_losses[model] / total_games if total_games > 0 else 0,
            'tie_rate': model_ties[model] / total_games if total_games > 0 else 0,
            'head_to_head': dict(head_to_head[model]),
            'elo_changes': elo_changes[model],
            'avg_elo_change': np.mean(elo_changes[model]) if elo_changes[model] else 0,
            'std_elo_change': np.std(elo_changes[model]) if elo_changes[model] else 0
        }

    sorted_results = OrderedDict(sorted(results.items(), key=lambda x: x[1]['elo'], reverse=True))
    sorted_extended_results = OrderedDict(sorted(extended_results.items(), key=lambda x: x[1]['elo'], reverse=True))
    
    metadata = {
        'judge_model': judge_name,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_rounds': completed_rounds,
        'total_comparisons': len(detailed_comparisons),
        'participating_models': list(all_models),
        'min_conversations': min_convs,
        'comparisons_per_round': comparisons_per_round
    }

    return sorted_results, sorted_extended_results, detailed_comparisons, metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ELO tournament between models')
    parser.add_argument('input_files', nargs='+', help='JSON files containing model outputs')
    parser.add_argument('--judge', choices=['openai', 'anthropic', 'mistral', 'together'],
                    required=True, help='Name of judge LLM')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--rounds', type=int, help='Total number of rounds (optional)')
    parser.add_argument('--add', action='store_true', help='Add new models to existing results')

    args = parser.parse_args()

    if len(args.input_files) < 2 and not args.add:
        raise ValueError("At least 2 input files required")

    existing_results = {}
    existing_extended_results = {}
    existing_rounds = None
    existing_detailed_comparisons = []
    existing_metadata = None
    
    output_base = os.path.splitext(args.output)[0]
    extended_output = f"{output_base}_extended.json"
    
    if args.add and os.path.exists(args.output):
        with open(args.output, 'r') as f:
            existing_data = json.load(f)
            existing_results = existing_data['results']
            existing_metadata = existing_data.get('metadata', {})  # Add this line
            num_existing_models = len(existing_results)
            any_model = next(iter(existing_results.values()))
            existing_rounds = any_model['total_votes'] // (num_existing_models - 1)
        
        output_base = os.path.splitext(args.output)[0]
        extended_output = f"{output_base}_extended.json"
        
        if os.path.exists(extended_output):
            with open(extended_output, 'r') as f:
                existing_extended_data = json.load(f)
                existing_extended_results = existing_extended_data['results']
                existing_detailed_comparisons = existing_extended_data.get('detailed_comparisons', [])
                
        # Get new models that aren't in existing results
        existing_models = set(existing_results.keys())
        new_models = [f for f in args.input_files if os.path.basename(f).split('.')[0] not in existing_models]
        
        final_results, final_extended_results, new_detailed_comparisons, new_metadata = run_elo_tournament(
            new_models,
            args.judge,
            existing_results=existing_extended_results,
            total_rounds=existing_rounds,
        )
        
        merged_metadata = {
            'judge_models': existing_metadata.get('judge_models', []),
            'timestamps': existing_metadata.get('timestamps', []),
            'total_rounds': existing_metadata.get('total_rounds', 0),
            'total_comparisons': existing_metadata.get('total_comparisons', 0) + new_metadata['total_comparisons'],
            'participating_models': list(set(existing_metadata.get('participating_models', []) + new_metadata['participating_models'])),
            'min_conversations': min(existing_metadata.get('min_conversations', float('inf')), new_metadata['min_conversations']),
            'comparisons_per_round': new_metadata['comparisons_per_round'],
            'last_update': new_metadata['timestamp']
        }
        
        if new_metadata['judge_model'] not in merged_metadata['judge_models']:
            merged_metadata['judge_models'].append(new_metadata['judge_model'])
        
        merged_metadata['timestamps'].append(new_metadata['timestamp'])
        
    else:
        final_results, final_extended_results, new_detailed_comparisons, new_metadata = run_elo_tournament(
            args.input_files,
            args.judge,
            total_rounds=args.rounds
        )
        
        merged_metadata = {
            'judge_models': [new_metadata['judge_model']],
            'timestamps': [new_metadata['timestamp']],
            'total_rounds': new_metadata['total_rounds'],
            'total_comparisons': new_metadata['total_comparisons'],
            'participating_models': new_metadata['participating_models'],
            'min_conversations': new_metadata['min_conversations'],
            'comparisons_per_round': new_metadata['comparisons_per_round'],
            'last_update': new_metadata['timestamp']
        }
        
        existing_detailed_comparisons = []

    combined_detailed_comparisons = existing_detailed_comparisons + new_detailed_comparisons

    with open(args.output, 'w') as f:
        json.dump({
            'results': final_results,
            'metadata': merged_metadata
        }, f, indent=2)

    with open(extended_output, 'w') as f:
        json.dump({
            'results': final_extended_results,
            'detailed_comparisons': combined_detailed_comparisons,
            'metadata': merged_metadata
        }, f, indent=2)
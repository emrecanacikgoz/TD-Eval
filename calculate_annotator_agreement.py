import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def get_batch_dialogues(all_dialogues, batch_list):
    #load dialogues
    batch_dials = []
    batch_dial_ids = batch_list
    batch_idx = 0
    for dialog in all_dialogues:
        if batch_idx == len(batch_dial_ids):
            break
        if dialog["dialogue_id"] == batch_dial_ids[batch_idx]:
            batch_dials.append(dialog)
            batch_idx += 1
    if len(batch_dials) != len(batch_dial_ids):
        print("filtered dials size does not match batches:", len(batch_dials))
        exit()
    return batch_dials

def process_extracted_human_csv_data(input_file, batch_dialogues):
    """Read CSV data and convert to appropriate format"""
    eval_csv = pd.read_csv(input_file, on_bad_lines='warn') 
    start_column = 'QID3_1'
    end_column = 'QID409_4'
    search_string = '2024'
    result = {}
    first_eval_row = eval_csv.StartDate.str.contains(search_string).idxmax()
    scores = eval_csv.loc[first_eval_row:, start_column:end_column].to_numpy()
    # extract scores into results dialogue map
    dial_idx = 0
    dial_turn_idx = 0
    curr_dial = batch_dialogues[dial_idx]
    # need to deal with with the four different scores
    for scores_idx in range(0, scores.shape[1], 4):
        if len(curr_dial["turn_scores"]) == dial_turn_idx:
            dial_turn_idx = 0
            dial_idx += 1
            curr_dial = batch_dialogues[dial_idx]
        dialog_id = curr_dial["dialogue_id"]
        turn_scores = scores[:, scores_idx:scores_idx+4]
        if dialog_id not in result:
            result[dialog_id] = [{
                'conv_consistency': np.array([float(x) if not pd.isnull(x) and 1 <= float(x) <= 5 else -1 for x in turn_scores[:,0]]),
                'backend_consistency': np.array([float(x) if not pd.isnull(x) and 1 <= float(x) <= 5 else -1 for x in turn_scores[:,1]]),
                'policy_completeness': np.array([float(x) if not pd.isnull(x) and 1 <= float(x) <= 5 else -1 for x in turn_scores[:, 2]])
            }]
        else:
            result[dialog_id].append({
                'conv_consistency': np.array([float(x) if not pd.isnull(x) and 1 <= float(x) <= 5 else -1 for x in turn_scores[:,0]]),
                'backend_consistency': np.array([float(x) if not pd.isnull(x) and 1 <= float(x) <= 5 else -1 for x in turn_scores[:,1]]),
                'policy_completeness': np.array([float(x) if not pd.isnull(x) and 1 <= float(x) <= 5 else -1 for x in turn_scores[:, 2]])
            })
        dial_turn_idx += 1
    return result

def calculate_turn_scores(human_eval_scores, batch_dialogues):
    """Compare human evaluation data with LLM evaluation data"""    
    def extract_score(score_str):
        try:
            import re
            match = re.search(r'Score: (\d+)', str(score_str))
            if not match:
                print("Score not found in string:",score_str)
            return float(match.group(1)) if match else 5.0
        except:
            print("Score not found in string:",score_str)
            return 5.0
    
    human_llm_scores = {
        'human_conv_consistency_scores': [],
        'human_backend_consistency_scores': [],
        'human_policy_completeness_scores': [],
        'llm_conv_consistency_scores': [],
        'llm_backend_consistency_scores': [],
        'llm_policy_completeness_scores': [],
        'score_differences': {
            'conv_consistency': [], 
            'backend_consistency': [], 
            'policy_completeness': []
        },
        'absolute_differences': {
            'conv_consistency': [], 
            'backend_consistency': [], 
            'policy_completeness': []        },
        'mean_absolute_errors': {
            'conv_consistency': [], 
            'backend_consistency': [], 
            'policy_completeness': []        }
    }
    for human_dial_id, human_dial_scores in human_eval_scores.items():
        llm_dial = None
        for dialog in batch_dialogues:
            if dialog["dialogue_id"] == human_dial_id:
                llm_dial = dialog
                break
        if llm_dial is None:
            print("dialogue not found")
            exit()
        # store and calculate human and llm scores
        for turn_idx, turn_score in enumerate(human_dial_scores):    
            # skip turn score if any negative/invalid scores exist
            all_scores = np.concat((turn_score['conv_consistency'], turn_score['backend_consistency'], turn_score['policy_completeness']))
            if np.any(all_scores <= 0):
                # print("invalid turn score:", turn_score)
                continue
            # accumulate human and llm scores for calculations
            avg_human_scores = {
                metric: float(np.mean(score)) for metric, score in turn_score.items()
            }
            # Get LLM scores
            llm_turn = llm_dial['turn_scores'][turn_idx]
            llm_conv_consistency = extract_score(llm_turn.get('conv_consistency', 'Score: 5'))
            llm_backend_consistency = extract_score(llm_turn.get('backend_consistency', 'Score: 5'))
            llm_policy_completeness = extract_score(llm_turn.get('policy_completeness', 'Score: 5'))
            # Store human scores
            human_llm_scores['human_conv_consistency_scores'].append(avg_human_scores['conv_consistency'])
            human_llm_scores['human_backend_consistency_scores'].append(avg_human_scores['backend_consistency'])
            human_llm_scores['human_policy_completeness_scores'].append(avg_human_scores['policy_completeness'])
            # Store llm scores
            human_llm_scores['llm_conv_consistency_scores'].append(llm_conv_consistency)
            human_llm_scores['llm_backend_consistency_scores'].append(llm_backend_consistency)
            human_llm_scores['llm_policy_completeness_scores'].append(llm_policy_completeness)
            # Calculate score differences
            human_llm_scores['score_differences']['conv_consistency'].append(avg_human_scores['conv_consistency'] - llm_conv_consistency)
            human_llm_scores['score_differences']['backend_consistency'].append(avg_human_scores['backend_consistency'] - llm_backend_consistency)
            human_llm_scores['score_differences']['policy_completeness'].append(avg_human_scores['policy_completeness'] - llm_policy_completeness)
            # Calculate absolute differences
            human_llm_scores['absolute_differences']['conv_consistency'].append(abs(avg_human_scores['conv_consistency'] - llm_conv_consistency))
            human_llm_scores['absolute_differences']['backend_consistency'].append(abs(avg_human_scores['backend_consistency'] - llm_backend_consistency))
            human_llm_scores['absolute_differences']['policy_completeness'].append(abs(avg_human_scores['policy_completeness'] - llm_policy_completeness))
        # Calculate means absolute difference
        for metric in ['conv_consistency', 'backend_consistency', 'policy_completeness']:
            if human_llm_scores['absolute_differences'][metric]:
                human_llm_scores['mean_absolute_errors'][metric] = float(np.mean(human_llm_scores['absolute_differences'][metric]))
                
    return human_llm_scores

def calculate_agreement_metrics(human_eval_data, scores):
    """Calculate agreement metrics between annotators and between annotators and LLM"""
    def calculate_krippendorff_alpha(data):
        """Calculate Krippendorff's alpha for ordinal data"""
        try:
            from krippendorff import alpha
            return alpha(reliability_data=data.astype(np.int64), value_domain=[1,2,3,4,5], level_of_measurement='interval')
        except Exception as e:
            print(f"Krippendorff calculation error: {e}")
            return None

    def calculate_fleiss_kappa(data):
        """Calculate Fleiss' kappa (for more than 2 raters)"""
        try:
            from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
            processed_data =  data.astype(np.int64) - 1
            data_table, _ = aggregate_raters(data=processed_data, n_cat=5)
            return fleiss_kappa(table=data_table, method='fleiss')
        except Exception as e:
            print(f"Fleiss calculation error: {e}")
            return None
        
    def calculate_cohen_kappa(data):
        """Calculate Fleiss' kappa for 2 raters"""
        try:
            from statsmodels.stats.inter_rater import cohens_kappa, to_table
            processed_data = data.astype(np.int64) - 1
            data_table, _ = to_table(data=processed_data, bins=5)
            return cohens_kappa(table=data_table, wt='linear')['kappa']
        except Exception as e:
            print(f"Cohen calculation error: {e}")
            return None

    from itertools import zip_longest
    metrics = {
        'inter_annotator_agreement': {},
        'annotator_llm_agreement': {},
        'bias_metrics': {},
        'variance_metrics': {}
    }
    
    human_dial_val = next(iter(human_eval_data.values()))
    num_annotators = human_dial_val[0]["conv_consistency"].shape[0]
    annotator_data = {
        'conv_consistency': np.empty((num_annotators, 0), dtype=float),
        'backend_consistency': np.empty((num_annotators, 0), dtype=float),
        'policy_completeness': np.empty((num_annotators, 0), dtype=float)
    }
    for dial_turns in human_eval_data.values():
        for turn_score in dial_turns:
            # skip turn score if any negative/invalid scores exist
            all_scores = np.concat((turn_score['conv_consistency'], turn_score['backend_consistency'], turn_score['policy_completeness']))
            if np.any(all_scores <= 0):
                continue
            # compile reliability matrices for inter-annotator agreement
            for metric, score in turn_score.items():
                annotator_data[metric] = np.hstack((annotator_data[metric], np.reshape(score, shape=(num_annotators,1))))

    for metric in ['conv_consistency', 'backend_consistency', 'policy_completeness']:
        # calculate inter-annotator agreement
        annotator_array = annotator_data[metric]
        k_alpha = calculate_krippendorff_alpha(annotator_array)
        fleiss = calculate_fleiss_kappa(annotator_array.T)
        metrics['inter_annotator_agreement'][metric] = {
            'k_alpha': float(k_alpha) if k_alpha is not None else 0.0,
            'f_kappa': float(fleiss) if fleiss is not None else 0.0,
        }
        # calculate llm-annotator agreement, both average and individual
        llm_scores = np.array(scores[f'llm_{metric}_scores'])
        human_scores = np.array(scores[f'human_{metric}_scores'])
        annotator_agreements = {'absolute': [], 'c_kappa': [], 'k_alpha': []}
        abs_agreement = 1 - (np.abs(human_scores - llm_scores) / 9).mean()
        annotator_agreements['absolute'].append(abs_agreement)
        avg_human_llm_scores = np.vstack((llm_scores, human_scores))
        k_alpha = calculate_krippendorff_alpha(avg_human_llm_scores)
        annotator_agreements['k_alpha'].append(k_alpha)
        cohen = calculate_cohen_kappa(avg_human_llm_scores.T)
        annotator_agreements['c_kappa'].append(cohen)
        for annotator_idx in range(num_annotators):
            annotator_scores = annotator_data[metric][annotator_idx]
            abs_individual_agreement = 1 - (np.abs(annotator_scores - llm_scores) / 9).mean()
            annotator_agreements['absolute'].append(abs_individual_agreement)
            annotator_llm_scores = np.vstack((llm_scores, annotator_scores))
            k_alpha = calculate_krippendorff_alpha(annotator_llm_scores)
            annotator_agreements['k_alpha'].append(k_alpha)
            cohen = calculate_cohen_kappa(annotator_llm_scores.T)
            annotator_agreements['c_kappa'].append(cohen)
        # calculate agreement statistics
        agreement_stats = {}
        agreement_stats['absolute_agreement'] = {
            'mean': float(np.mean(annotator_agreements['absolute'])) if annotator_agreements['absolute'] else 0.0,
            'std': float(np.std(annotator_agreements['absolute'])) if annotator_agreements['absolute'] else 0.0,
            'min': float(np.min(annotator_agreements['absolute'])) if annotator_agreements['absolute'] else 0.0,
            'max': float(np.max(annotator_agreements['absolute'])) if annotator_agreements['absolute'] else 0.0
        }
        agreement_stats['k_alpha'] = {
            'mean': float(np.mean(annotator_agreements['k_alpha'])) if annotator_agreements['k_alpha'] else 0.0,
            'std': float(np.std(annotator_agreements['k_alpha'])) if annotator_agreements['k_alpha'] else 0.0,
            'min': float(np.min(annotator_agreements['k_alpha'])) if annotator_agreements['k_alpha'] else 0.0,
            'max': float(np.max(annotator_agreements['k_alpha'])) if annotator_agreements['k_alpha'] else 0.0
        }
        agreement_stats['c_kappa'] = {
            'mean': float(np.mean(annotator_agreements['c_kappa'])) if annotator_agreements['c_kappa'] else 0.0,
            'std': float(np.std(annotator_agreements['c_kappa'])) if annotator_agreements['c_kappa'] else 0.0,
            'min': float(np.min(annotator_agreements['c_kappa'])) if annotator_agreements['c_kappa'] else 0.0,
            'max': float(np.max(annotator_agreements['c_kappa'])) if annotator_agreements['c_kappa'] else 0.0
        }
        metrics['annotator_llm_agreement'][metric] = agreement_stats
        # calculate bias and variance across annotator scores
        metrics['bias_metrics'][metric] = float(np.mean(scores['score_differences'][metric]))
        all_human_scores = []
        for annotator_idx in range(num_annotators):
            scores_array = np.array(annotator_data[metric][annotator_idx])
            valid_scores = scores_array[~np.isnan(scores_array)]
            all_human_scores.extend(valid_scores)
        metrics['variance_metrics'][metric] = float(np.std(all_human_scores)) if all_human_scores else 0.0
        # debug info
        print(f"\nDebug information for {metric}:")
        print(f"Number of valid scores per annotator:")
        for i in range(num_annotators):
            valid_count = np.sum(~np.isnan(annotator_data[metric][i]))
            print(f"Annotator {i}: {valid_count}")
        print(f"Total samples with valid scores: {len(all_human_scores)}")
    
    return metrics

def generate_comparison_plots(scores, output_dir, is_llm_to_llm=False):
    """Generate comparison plots for either human vs LLM or LLM vs LLM scores"""
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = ['conv_consistency', 'backend_consistency', 'policy_completeness']
    
    for metric in metrics:
        if is_llm_to_llm:
            x_scores = scores[f'llm1_{metric}_scores']
            y_scores = scores[f'llm2_{metric}_scores']
            x_label = f'LLM1 {metric.capitalize()} Scores'
            y_label = f'LLM2 {metric.capitalize()} Scores'
            title_prefix = 'LLM1 vs LLM2'
        else:
            x_scores = scores[f'human_{metric}_scores']
            y_scores = scores[f'llm_{metric}_scores']
            x_label = f'Human {metric.capitalize()} Scores'
            y_label = f'LLM {metric.capitalize()} Scores'
            title_prefix = 'Human vs LLM'

        plt.figure(figsize=(10, 6))
        plt.scatter(
            x_scores,
            y_scores,
            alpha=0.5,
            label=f'{title_prefix} Scores'
        )
        plt.plot([0, 10], [0, 10], 'k--')  # Diagonal line
        plt.title(f'{title_prefix} {metric.capitalize()} Scores')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric}_scatter.png'))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.hist(scores['score_differences'][metric], bins=20, edgecolor='black')
        plt.title(f'Distribution of {metric.capitalize()} Score Differences')
        plt.xlabel('Score Difference')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric}_differences_histogram.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.boxplot([x_scores, y_scores],
                   labels=[x_label.split()[0], y_label.split()[0]])
        plt.title(f'{metric.capitalize()} Scores Distribution')
        plt.ylabel('Score')
        plt.ylim(0, 11)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric}_boxplot.png'))
        plt.close()

def human_eval_process(human_eval_csv, llm_eval_json, dial_batches, output_dir=None):
    """Process human evaluation CSV and compare with LLM evaluation"""
    if output_dir is None:
        output_dir = 'human_eval_' + os.path.basename(human_eval_csv).split('.')[0]
    os.makedirs(output_dir, exist_ok=True)
    # load dialogues
    with open(llm_eval_json, 'r') as f:
        llm_eval_data = json.load(f)
    all_dialogues = llm_eval_data.get('dialogues', [])
    # load batches
    batches = None
    with open(dial_batches, 'r') as f:
        batches = json.load(f)
    batches = batches.get('dial_ids', [])
    if batches is None or len(batches) == 0:
        print('No batches found at this path:',  dial_batches)
        exit()
    # compile human and llm evaluations
    batch_dialogues = get_batch_dialogues(all_dialogues, batches)
    human_eval_data = process_extracted_human_csv_data(human_eval_csv, batch_dialogues)
    # calculate agreement scores
    scores = calculate_turn_scores(human_eval_data, batch_dialogues)
    generate_comparison_plots(scores, output_dir)
    agreement_metrics = calculate_agreement_metrics(human_eval_data, scores)
    comparison_results = {
        'turn_level_scores': scores,
        'mean_absolute_errors': scores['mean_absolute_errors'],
        'total_questions': len(scores['human_conv_consistency_scores']),
        'agreement_metrics': agreement_metrics
    }
    with open(os.path.join(output_dir, 'human_llm_comparison.json'), 'w') as f:
        json.dump(comparison_results, f, indent=4)
    
    print(f"\nTotal questions analyzed: {comparison_results['total_questions']}")
    print(f"\nResults saved to {output_dir}")
    return comparison_results

def calculate_llm_to_llm_scores(llm_eval_data1, llm_eval_data2):
    """Compare evaluations between two LLM systems"""
    all_dialogues1 = llm_eval_data1.get('dialogues', [])
    all_dialogues2 = llm_eval_data2.get('dialogues', [])
    
    def extract_score(score_str):
        try:
            import re
            
            match = re.search(r'Score: (\d+)', str(score_str))
            if match:
                return float(match.group(1))
            else: 
                print("Score not found in string:",score_str)
                return -1.0
        except:
            print("Score not found in string:",score_str)
            return -1.0
    
    scores = {
        'llm1_conv_consistency_scores': [],
        'llm1_backend_consistency_scores': [],
        'llm1_policy_completeness_scores': [],
        'llm2_conv_consistency_scores': [],
        'llm2_backend_consistency_scores': [],
        'llm2_policy_completeness_scores': [],
        'score_differences': {
            'conv_consistency': [], 
            'backend_consistency': [], 
            'policy_completeness': []
        },
        'absolute_differences': {
            'conv_consistency': [], 
            'backend_consistency': [], 
            'policy_completeness': []        
        },
        'mean_absolute_errors': {
            'conv_consistency': [], 
            'backend_consistency': [], 
            'policy_completeness': []
        }
    }
    
    min_dialogues = 30 
    for dialogue_idx in range(min_dialogues):
        dialogue1 = all_dialogues1[dialogue_idx]
        dialogue2 = all_dialogues2[dialogue_idx]
        
        min_turns = min(len(dialogue1['turn_scores']), len(dialogue2['turn_scores']))
        
        for turn_idx in range(min_turns):
            turn1 = dialogue1['turn_scores'][turn_idx]
            turn2 = dialogue2['turn_scores'][turn_idx]
            
            llm1_conv_consistency = extract_score(turn1.get('conv_consistency', 'Score: -1'))
            llm1_backend_consistency = extract_score(turn1.get('backend_consistency', 'Score: -1'))
            llm1_policy_completeness = extract_score(turn1.get('policy_completeness', {}).get('judge_score', 'Score: -1'))
            if (min(llm1_conv_consistency, llm1_backend_consistency, llm1_policy_completeness)) < 0:
                continue
            
            llm2_conv_consistency = extract_score(turn2.get('conv_consistency', 'Score: -1'))
            llm2_backend_consistency = extract_score(turn2.get('backend_consistency', 'Score: -1'))
            llm2_policy_completeness = extract_score(turn2.get('policy_completeness', {}).get('judge_score', 'Score: -1'))
            if (min(llm2_conv_consistency, llm2_backend_consistency, llm2_policy_completeness)) < 0:
                continue
            
            scores['llm1_conv_consistency_scores'].append(llm1_conv_consistency)
            scores['llm1_backend_consistency_scores'].append(llm1_backend_consistency)
            scores['llm1_policy_completeness_scores'].append(llm1_policy_completeness)
            
            scores['llm2_conv_consistency_scores'].append(llm2_conv_consistency)
            scores['llm2_backend_consistency_scores'].append(llm2_backend_consistency)
            scores['llm2_policy_completeness_scores'].append(llm2_policy_completeness)
            
            scores['score_differences']['conv_consistency'].append(llm1_conv_consistency - llm2_conv_consistency)
            scores['score_differences']['backend_consistency'].append(llm1_backend_consistency - llm2_backend_consistency)
            scores['score_differences']['policy_completeness'].append(llm1_policy_completeness - llm2_policy_completeness)
            
            scores['absolute_differences']['conv_consistency'].append(abs(llm1_conv_consistency - llm2_conv_consistency))
            scores['absolute_differences']['backend_consistency'].append(abs(llm1_backend_consistency - llm2_backend_consistency))
            scores['absolute_differences']['policy_completeness'].append(abs(llm1_policy_completeness - llm2_policy_completeness))
    
    for metric in ['conv_consistency', 'backend_consistency', 'policy_completeness']:
        if scores['absolute_differences'][metric]:
            scores['mean_absolute_errors'][metric] = float(np.mean(scores['absolute_differences'][metric]))
    
    return scores

def calculate_llm_agreement_metrics(scores):
    """Calculate agreement metrics between two LLM evaluators using same metrics as human eval"""
    metrics = {
        'inter_llm_agreement': {},
        'llm_scorers_agreement': {},
        'bias_metrics': {},
        'variance_metrics': {}
    }
    
    """Calculate agreement metrics between annotators and between annotators and LLM"""
    def calculate_krippendorff_alpha(data):
        """Calculate Krippendorff's alpha for ordinal data"""
        try:
            from krippendorff import alpha
            return alpha(reliability_data=data.astype(np.int64), value_domain=[1,2,3,4,5], level_of_measurement='interval')
        except Exception as e:
            print(f"Krippendorff calculation error: {e}")
            return None
        
    def calculate_cohen_kappa(data):
        """Calculate Fleiss' kappa for 2 raters"""
        try:
            from statsmodels.stats.inter_rater import cohens_kappa, to_table
            processed_data = data.astype(np.int64) - 1
            data_table, _ = to_table(data=processed_data, bins=5)
            return cohens_kappa(table=data_table, wt='linear')['kappa']
        except Exception as e:
            print(f"Cohen calculation error: {e}")
            return None
    
    for metric in ['conv_consistency', 'backend_consistency', 'policy_completeness']:
        llm1_scores = np.array(scores[f'llm1_{metric}_scores'])
        llm2_scores = np.array(scores[f'llm2_{metric}_scores'])
        
        llm_scores_matrix = np.vstack((llm1_scores, llm2_scores))
        k_alpha = calculate_krippendorff_alpha(llm_scores_matrix)
        c_kappa = calculate_cohen_kappa(llm_scores_matrix.T)
        metrics['inter_llm_agreement'][metric] = {
            'k_alpha': float(k_alpha) if k_alpha is not None else 0.0,
            'c_kappa': float(c_kappa) if c_kappa is not None else 0.0,
        }
        
        absolute_diff = np.abs(llm1_scores - llm2_scores)
        abs_agreement = 1 - (absolute_diff / 9).mean()
        metrics['llm_scorers_agreement'][metric] = {
            'absolute_agreement': {
                'mean': float(abs_agreement),
                'std': float(np.std(absolute_diff) / 9),
                'min': float(1 - np.max(absolute_diff) / 9),
                'max': float(1 - np.min(absolute_diff) / 9)
            }
        }
        # Calculate bias (mean difference)
        bias = np.mean(scores['score_differences'][metric])
        metrics['bias_metrics'][metric] = float(bias)
        # Calculate variance
        metrics['variance_metrics'][metric] = float(np.std(scores['score_differences'][metric]))
    
    return metrics

def llm_to_llm_eval_process(llm1_eval_json, llm2_eval_json):
        with open(llm1_eval_json, 'r') as f1, open(llm2_eval_json, 'r') as f2:
            llm_eval_data1 = json.load(f1)
            llm_eval_data2 = json.load(f2)
        
        scores = calculate_llm_to_llm_scores(llm_eval_data1, llm_eval_data2)
        output_dir = 'llm_to_llm_comparison'
        os.makedirs(output_dir, exist_ok=True)
        
        agreement_metrics = calculate_llm_agreement_metrics(scores)
        generate_comparison_plots(scores, output_dir, is_llm_to_llm=True)
        
        comparison_results = {
            'turn_level_scores': scores,
            'mean_absolute_errors': scores['mean_absolute_errors'],
            'total_questions': len(scores['llm1_conv_consistency_scores']),
            'agreement_metrics': agreement_metrics
        }
        
        with open(os.path.join(output_dir, 'llm_to_llm_comparison.json'), 'w') as f:
            json.dump(comparison_results, f, indent=4)
        print(f"\nTotal questions analyzed: {comparison_results['total_questions']}")
        print(f"\nResults saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Process human and LLM evaluations')
    parser.add_argument('human_eval_csv', help='Path to human evaluation CSV file')
    parser.add_argument('llm_eval_json', help='Path to LLM evaluation JSON file')
    parser.add_argument('dial_batch_json', default='batches.json', help='Path to evaluated dialogue batches JSON file')
    parser.add_argument('--llmtollm', help='Path to second LLM evaluation JSON file for LLM-to-LLM comparison')
    
    args = parser.parse_args()
    
    if args.llmtollm:
        llm_to_llm_eval_process(args.llm_eval_json, args.llmtollm)
    else:
        human_eval_process(args.human_eval_csv, args.llm_eval_json, args.dial_batch_json)
        
if __name__ == "__main__":
    main()
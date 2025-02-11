import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import scipy.stats as stats

def get_batch_dialogues(mwoz_dialogues, tau_air_dialogues, tau_retail_dialogues, batch_list):
    #load dialogues
    batch_dials = []
    mwoz_batch_ids = batch_list["mwoz"]
    tau_air_batch_ids = batch_list["tau"]["airline"]
    tau_retail_batch_ids = batch_list["tau"]["retail"]
    # load batch 
    for batch_id in mwoz_batch_ids:
        for id, dial in mwoz_dialogues.items():
            if id == batch_id:
                batch_dials.append(dial)
                break
    for batch_id in tau_air_batch_ids:
        for id, dial in tau_air_dialogues.items():
            if id == batch_id:
                batch_dials.append(dial)
                break
    for batch_id in tau_retail_batch_ids:
        for id, dial in tau_retail_dialogues.items():
            if id == batch_id:
                batch_dials.append(dial)
                break
    tot_batch_len = len(mwoz_batch_ids) + len(tau_air_batch_ids) + len(tau_retail_batch_ids)
    if len(batch_dials) != tot_batch_len:
        print("filtered dials size does not match batches:", len(batch_dials), tot_batch_len)
        exit()
    return batch_dials

def process_extracted_human_csv_data(input_file, batch_dialogues, batch_order):
    """Read CSV data and convert to appropriate format"""
    eval_csv = pd.read_csv(input_file, on_bad_lines='warn') 
    start_col = 'QID3_1'
    end_col = 'QID83' # QID83 QID76 QID83 QID91 QID80
    search_str = '2025'
    turn_result = {}
    dial_result = {}
    first_eval_row = eval_csv.StartDate.str.contains(search_str).idxmax()
    human_scores = eval_csv.loc[first_eval_row:, start_col:end_col].to_numpy()
    mapping = {"Very Good": 5.0, "Good": 4.0, "Fair": 3.0, "Bad": 2.0, "Very Bad": 1.0}
    vectorized_map = np.vectorize(lambda x: mapping[x])
    int_scores = vectorized_map(human_scores)
    # extract scores into results dialogue map
    scores_idx = 0
    for i, dial in enumerate(batch_dialogues):
        dial_id = batch_order[i]["id"]
        turn_result[dial_id] = []
        # add turn scores
        for turn in dial:
            turn_result[dial_id].append({
                'conv_consistency': int_scores[:,scores_idx],
                'backend_consistency': int_scores[:,scores_idx+1],
                'policy_completeness': int_scores[:, scores_idx+2]
            })
            scores_idx += 3
        # add dial scores
        dial_result[dial_id] = {
            'task_complete': int_scores[:,scores_idx],
            'response_cohere': int_scores[:,scores_idx+1]
        }
        scores_idx += 2
    return turn_result, dial_result

"""Compare human evaluation data with LLM evaluation data"""    
def extract_score(score_str):
    try:
        import re
        # regex matching
        match = re.search(r'Score: (\d+)', str(score_str))
        if not match:
            print("Score not found in string:",score_str)
            print("Checking substring")
            # check substring
            if "Very Good" in score_str:
                return 5
            elif "Good" in score_str:
                return 4
            elif "Fair" in score_str:
                return 3
            # check more detailed string first
            elif "Very Bad" in score_str: 
                return 4
            elif "Bad" in score_str:
                return 1
            else:
                print("Score still not found with substring check")
                return 5
        return int(match.group(1)) if match else 5
    except:
        print("Score not found in string:",score_str)
        return 5

def process_turn_scores(turn_eval_scores, batch_dialogues, batch_order):
    human_llm_scores = {
        'human_conv_consistency_scores': {},
        'human_backend_consistency_scores': {},
        'human_policy_completeness_scores': {},
        'avg_human_conv_consistency_scores': [],
        'avg_human_backend_consistency_scores': [],
        'avg_human_policy_completeness_scores': [],
        'llm_conv_consistency_scores': [],
        'llm_backend_consistency_scores': [],
        'llm_policy_completeness_scores': []
    }
    for human_dial_id, human_dial_scores in turn_eval_scores.items():
        llm_dial = None
        for idx, dial_id in enumerate(batch_order):
            if dial_id["id"] == human_dial_id:
                llm_dial = batch_dialogues[idx]
                break
        if llm_dial is None:
            print("dialogue not found")
            exit()
        # store and calculate human and llm scores
        for turn_idx, turn_score in enumerate(human_dial_scores):    
            # skip turn score if any negative/invalid scores exist
            all_scores = np.concat((turn_score['conv_consistency'], turn_score['backend_consistency'], turn_score['policy_completeness']))
            if np.any(all_scores <= 0):
                continue
            # collect scores from all annotators
            for metric in ['conv_consistency', 'backend_consistency', 'policy_completeness']:
                for id, score in enumerate(turn_score[metric]):
                    if id not in human_llm_scores[f'human_{metric}_scores']:
                        human_llm_scores[f'human_{metric}_scores'][id] = [score]
                    else: 
                        human_llm_scores[f'human_{metric}_scores'][id].append(score)
            # accumulate human and llm scores for calculations
            avg_human_scores = {
                metric: round(np.mean(score)) for metric, score in turn_score.items()
            }
            # Get LLM scores
            llm_turn = llm_dial[turn_idx]["scores"]
            llm_conv_consistency = extract_score(llm_turn['conv_consistency'].get('score', 'Score: 5'))
            llm_backend_consistency = extract_score(llm_turn['backend_consistency'].get('score', 'Score: 5'))
            llm_policy_completeness = extract_score(llm_turn['policy_completeness'].get('score', 'Score: 5'))
            # Store human scores
            human_llm_scores['avg_human_conv_consistency_scores'].append(avg_human_scores['conv_consistency'])
            human_llm_scores['avg_human_backend_consistency_scores'].append(avg_human_scores['backend_consistency'])
            human_llm_scores['avg_human_policy_completeness_scores'].append(avg_human_scores['policy_completeness'])
            # Store llm scores
            human_llm_scores['llm_conv_consistency_scores'].append(llm_conv_consistency)
            human_llm_scores['llm_backend_consistency_scores'].append(llm_backend_consistency)
            human_llm_scores['llm_policy_completeness_scores'].append(llm_policy_completeness)
    return human_llm_scores

def calculate_krippendorff_alpha(data):
    """Calculate Krippendorff's alpha for ordinal data"""
    try:
        from krippendorff import alpha
        return alpha(reliability_data=data.astype(np.int64), value_domain=[1,2,3,4,5], level_of_measurement='interval')
    except Exception as e:
        print(f"Krippendorff calculation error: {e}")
        return None

def calculate_fleiss_kappa(data, n_cat):
    """Calculate Fleiss' kappa (for more than 2 raters)"""
    try:
        from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
        data_table, _ = aggregate_raters(data=data, n_cat=n_cat)
        # randolph method gives best performance
        return fleiss_kappa(table=data_table, method='randolph') 
    except Exception as e:
        print(f"Fleiss calculation error: {e}")
        return None
    
def calculate_cohen_kappa(data, n_cat):
    """Calculate Fleiss' kappa for 2 raters"""
    try:
        from statsmodels.stats.inter_rater import cohens_kappa, to_table
        data_table, _ = to_table(data=data, bins=n_cat)
        return cohens_kappa(table=data_table, wt='linear')['kappa']
    except Exception as e:
        print(f"Cohen calculation error: {e}")
        return None


def calculate_irr(human_eval_data):
    """Calculate agreement metrics between annotators and between annotators and LLM"""
    metrics = {
        'inter_annotator_agreement': {}
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
        processed_data = annotator_array.T.astype(np.int64) - 1
        c_kappa = calculate_cohen_kappa(processed_data, 5)
        fleiss = calculate_fleiss_kappa(processed_data, 5)
        metrics['inter_annotator_agreement'][metric] = {
            'k_alpha': float(k_alpha) if k_alpha is not None else 0.0,
            'f_kappa': float(fleiss) if fleiss is not None else 0.0,
            'c_kappa': float(c_kappa) if c_kappa is not None else 0.0
        }
        all_human_scores = []
        for annotator_idx in range(num_annotators):
            scores_array = np.array(annotator_data[metric][annotator_idx])
            valid_scores = scores_array[~np.isnan(scores_array)]
            all_human_scores.extend(valid_scores)
        # debug info
        print(f"\nDebug information for {metric}:")
        print(f"Number of valid scores per annotator:")
        for i in range(num_annotators):
            valid_count = np.sum(~np.isnan(annotator_data[metric][i]))
            print(f"Annotator {i}: {valid_count}")
        print(f"Total samples with valid scores: {len(all_human_scores)}")
    
    return metrics

def calculate_human_llm_corr(human_scores, llm_scores):
    human_llm_agreement = {}
    for metric in ['conv_consistency', 'backend_consistency', 'policy_completeness']:
        human_metric_scores = np.array(human_scores[metric])
        print(f"human_scores[{metric}]: ", human_scores[metric])
        llm_metric_scores = np.array(llm_scores[metric])
        print(f"llm_scores[{metric}]: ", llm_scores[metric])

        pear_stat, pear_pval = stats.pearsonr(x=human_metric_scores, y=llm_metric_scores)
        print("pearson: ", pear_stat, pear_pval)
        spear_stat, spear_pval = stats.spearmanr(a=human_metric_scores, b=llm_metric_scores)
        print("spearman: ", pear_stat, pear_pval)

        bin_human_metric_scores = np.where(human_metric_scores > 3, 1, 0)
        print(f"bin_human_scores[{metric}]:", bin_human_metric_scores)
        bin_llm_metric_scores =  np.where(llm_metric_scores > 3, 1, 0)
        print(f"bin_llm_scores[{metric}]:", bin_llm_metric_scores)

        bin_pear_stat, bin_pear_pval = stats.pearsonr(x=bin_human_metric_scores, y=bin_llm_metric_scores)
        print(f"bin_pearson: ", bin_pear_stat, bin_pear_pval)
        bin_spear_stat, bin_spear_pval = stats.spearmanr(a=bin_human_metric_scores, b=bin_llm_metric_scores)
        print(f"bin_spearman: ", bin_spear_stat, bin_spear_pval)

        processed_scores = np.array([human_metric_scores, llm_metric_scores]).T
        processed_scores = processed_scores.astype(np.int64) - 1
        bin_processed_scores = np.array([bin_human_metric_scores, bin_llm_metric_scores]).T
        f_kappa = calculate_fleiss_kappa(processed_scores, 5)
        bin_f_kappa = calculate_fleiss_kappa(bin_processed_scores, 2)

        human_llm_agreement[metric] = {
            'pearson': {
                'coeff': pear_stat,
                'p_value': pear_pval
            },
            'binary_pearson': {
                'coeff': bin_pear_stat,
                'p_value': bin_pear_pval
            },
            "spearman": {
                'coeff': spear_stat,
                'p_value': spear_pval
            },
            'binary_spearman': {
                'coeff': bin_spear_stat,
                'p_value': bin_spear_pval
            },
            'f_kappa': f_kappa,
            'binary_f_kappa': bin_f_kappa
        }
    return human_llm_agreement


def calculate_human_mwzeval_corr(human_scores, batch_order, batch_dialogues, mwzeval):
    human_mwzeval_agreement = {}
    turns_idx = 0
    all_success_evals = []
    all_inform_evals = []
    accumulate_human_evals = []
    for i, dial_info in enumerate(batch_order):
        dial = batch_dialogues[i]
        dial_len = len(dial)
        if dial_info['type'] != "mwoz":
            turns_idx += dial_len
            continue
        # get mwzeval 
        dial_mwzeval = mwzeval[dial_info["id"]]
        print(dial_info["id"], dial_mwzeval)
        inform = dial_mwzeval["inform"]["total"]
        if "total" in dial_mwzeval["success"]:
            success = dial_mwzeval["success"]["total"]
        else: 
            success = False
        all_inform_evals.append(inform)
        all_success_evals.append(success)
        # accumulate human turn scores
        conv_metric_scores = human_scores['conv_consistency'][turns_idx:turns_idx+dial_len]
        backend_metric_scores = human_scores['backend_consistency'][turns_idx:turns_idx+dial_len]
        policy_metric_scores = human_scores['policy_completeness'][turns_idx:turns_idx+dial_len]
        score_concat = np.concatenate((conv_metric_scores, backend_metric_scores, policy_metric_scores))
        norm_score = np.mean(score_concat)
        accumulate_human_evals.append(norm_score)
        turns_idx += dial_len
    accumulate_human_evals = np.array(accumulate_human_evals)
    norm_accumulated = np.where(accumulate_human_evals > 4, 1, 0)
    pear_inf, pear_inf_pval = stats.pearsonr(norm_accumulated, all_inform_evals)
    spear_inf, spear_inf_pval = stats.spearmanr(norm_accumulated, all_inform_evals)
    pear_succ, pear_succ_pval = stats.pearsonr(norm_accumulated, all_success_evals)
    spear_succ, spear_succ_pval = stats.spearmanr(norm_accumulated, all_success_evals)
    inf_processed_scores = np.array([norm_accumulated, all_inform_evals]).T
    succ_processed_scores = np.array([norm_accumulated, all_success_evals]).T
    inf_f_kappa = calculate_fleiss_kappa(inf_processed_scores, 2)
    succ_f_kappa = calculate_fleiss_kappa(succ_processed_scores, 2)
    human_mwzeval_agreement = {
        'inform': {
            'pearson': {
                'coeff': pear_inf,
                'p_value': pear_inf_pval
            },         
            "spearman": {
                'coeff': spear_inf,
                'p_value': spear_inf_pval
            },
            "f_kappa": inf_f_kappa
        },
        'success': {
            'pearson': {
                'coeff': pear_succ,
                'p_value': pear_succ_pval
            },         
            "spearman": {
                'coeff': spear_succ,
                'p_value': spear_succ_pval
            },
            "f_kappa": succ_f_kappa
        },
    }
    return human_mwzeval_agreement

def calculate_llm_mwzeval_corr(mwoz_dials, mwzeval):
    all_success_evals = []
    all_inform_evals = []
    accumulate_llm_evals = []
    for id, turns in mwoz_dials.items():
        tot_turn_score = 0
        for t in turns:
            scores = t['scores']
            conv_eval = extract_score(scores["conv_consistency"]["score"])
            backend_eval = extract_score(scores["backend_consistency"]["score"])
            policy_eval = extract_score(scores["policy_completeness"]["score"])
            tot_metric_score = conv_eval+backend_eval+policy_eval
            avg_metric_score = tot_metric_score/3
            tot_turn_score += avg_metric_score
        avg_turn_score = round(tot_turn_score/len(turns))
        # print(avg_turn_score)
        accumulate_llm_evals.append(avg_turn_score)
        # get mwzeval 
        dial_mwzeval = mwzeval[id]
        inform = dial_mwzeval["inform"]["total"]
        if "total" in dial_mwzeval["success"]:
            success = dial_mwzeval["success"]["total"]
        else: 
            success = False
        all_inform_evals.append(inform)
        all_success_evals.append(success)
    accumulate_llm_evals = np.array(accumulate_llm_evals)
    norm_accumulated = np.where(accumulate_llm_evals > 3, 1, 0)
    pear_inf, pear_inf_pval = stats.pearsonr(norm_accumulated, all_inform_evals)
    spear_inf, spear_inf_pval = stats.spearmanr(norm_accumulated, all_inform_evals)
    pear_succ, pear_succ_pval = stats.pearsonr(norm_accumulated, all_success_evals)
    spear_succ, spear_succ_pval = stats.spearmanr(norm_accumulated, all_success_evals)

    inf_processed_scores = np.array([norm_accumulated, all_inform_evals]).T
    succ_processed_scores = np.array([norm_accumulated, all_success_evals]).T
    inf_f_kappa = calculate_fleiss_kappa(inf_processed_scores, 2)
    succ_f_kappa = calculate_fleiss_kappa(succ_processed_scores, 2)
    llm_mwzeval_agreement = {
        'inform': {
            'pearson': {
                'coeff': pear_inf,
                'p_value': pear_inf_pval
            },         
            "spearman": {
                'coeff': spear_inf,
                'p_value': spear_inf_pval
            },
            "f_kappa": inf_f_kappa
        },
        'success': {
            'pearson': {
                'coeff': pear_succ,
                'p_value': pear_succ_pval
            },         
            "spearman": {
                'coeff': spear_succ,
                'p_value': spear_succ_pval
            },
            "f_kappa": succ_f_kappa
        },
    }
    return llm_mwzeval_agreement

def human_eval_process(human_eval_csv, mwoz_eval_json, tau_air_eval_json, tau_retail_eval_json, dial_batches, output_dir=None):
    """Process human evaluation CSV and compare with LLM evaluation"""
    if output_dir is None:
        output_dir = 'human_eval_' + os.path.basename(human_eval_csv).split('.')[0]
    os.makedirs(output_dir, exist_ok=True)
    # load dialogues
    with open(mwoz_eval_json, 'r') as f:
        mwoz_eval = json.load(f)
    mwoz_dials = mwoz_eval.get('dialogues', [])
    mwzeval = mwoz_eval.get('mwzeval', {})
    with open(tau_air_eval_json, 'r') as f:
        tau_air_eval = json.load(f)
    tau_air_dials = tau_air_eval['dialogues']
    with open(tau_retail_eval_json, 'r') as f:
        tau_retail_eval = json.load(f)
    tau_retail_dials = tau_retail_eval['dialogues']
    # load batches
    batch_list = None
    with open(dial_batches, 'r') as f:
        batch_list = json.load(f)
    if batch_list is None or len(batch_list) == 0:
        print('No batches found at this path:',  dial_batches)
        exit()
    batch_order = batch_list["order"]
    # compile human and llm evaluations
    batch_dialogues = get_batch_dialogues(mwoz_dials, tau_air_dials, tau_retail_dials, batch_list)
    turn_eval_data, dial_eval_data = process_extracted_human_csv_data(human_eval_csv, batch_dialogues, batch_order)
    # calculate agreement scores
    scores = process_turn_scores(turn_eval_data, batch_dialogues, batch_order)
    agreement_metrics = calculate_irr(turn_eval_data)
    human_scores = {
        'conv_consistency': scores['avg_human_conv_consistency_scores'],
        'backend_consistency': scores['avg_human_backend_consistency_scores'],
        'policy_completeness': scores['avg_human_policy_completeness_scores']
    }
    llm_scores = {
        'conv_consistency': scores['llm_conv_consistency_scores'],
        'backend_consistency': scores['llm_backend_consistency_scores'],
        'policy_completeness': scores['llm_policy_completeness_scores']
    }
    human_llm_corr = calculate_human_llm_corr(human_scores, llm_scores)
    # human_mwzeval_corr = calculate_human_mwzeval_corr(human_scores, batch_order, batch_dialogues, mwzeval)
    llm_mwzeval_agreement = calculate_llm_mwzeval_corr(mwoz_dials, mwzeval)

    comparison_results = {
        'turn_level_scores': scores,
        'inter_annotator_agreement': agreement_metrics,
        'human_llm_agreement': human_llm_corr,
        'llm_mwzeval_agreement': llm_mwzeval_agreement
        # 'human_mwzeval_agreement': human_mwzeval_corr
    }
    with open(os.path.join(output_dir, 'human_llm_comparison.json'), 'w') as f:
        json.dump(comparison_results, f, indent=4)    
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
    parser.add_argument('--human_eval_csv', help='Path to human evaluation CSV file')
    parser.add_argument('--mwoz_eval_json', help='Path to LLM evaluation on MultiWOZ samples JSON file')
    parser.add_argument('--tau_air_eval_json', help='Path to LLM evaluation on Tau bench airline samples JSON file')
    parser.add_argument('--tau_retail_eval_json', help='Path to LLM evaluation on Tau bench retail samples JSON file')
    parser.add_argument('--dial_batch_json', default='batches.json', help='Path to evaluated dialogue batches JSON file')
    parser.add_argument('--llmtollm', help='Path to second LLM evaluation JSON file for LLM-to-LLM comparison')
    
    args = parser.parse_args()
    
    if args.llmtollm:
        llm_to_llm_eval_process(args.mwoz_eval_json, args.tau_air_eval_json, args.tau_retail_eval_json, args.llmtollm)
    else:
        human_eval_process(args.human_eval_csv, args.mwoz_eval_json, args.tau_air_eval_json, args.tau_retail_eval_json, args.dial_batch_json)
        
if __name__ == "__main__":
    main()
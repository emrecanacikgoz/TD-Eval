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
    mwoz_batch_ids = batch_list["autotod_mwoz"]
    tau_air_batch_ids = batch_list["tau"]["airline"]
    tau_retail_batch_ids = batch_list["tau"]["retail"]
    # load batch 
    for batch_id in mwoz_batch_ids:
        for id, dial in mwoz_dialogues.items():
            if id.split(".json")[0].lower() == batch_id:
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
    start_col = 'QID100_1'
    end_col = 'QID130_3'
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
        for _ in dial:
            turn_result[dial_id].append({
                'conv_consistency': int_scores[:,scores_idx],
                'backend_consistency': int_scores[:,scores_idx+1],
                'policy_completeness': int_scores[:, scores_idx+2]
            })
            scores_idx += 3
        # add dial scores
        dial_result[dial_id] = {
            'conv_consistency': int_scores[:,scores_idx],
            'backend_consistency': int_scores[:,scores_idx+1],
            'policy_completeness': int_scores[:, scores_idx+2]
        }
        scores_idx += 3
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


def calculate_irr(turn_eval_data, dial_eval_data):
    """Calculate agreement metrics between annotators and between annotators and LLM"""
    print("turn_eval_data:", turn_eval_data)
    print("dial_eval_data:", dial_eval_data)

    metrics = {}    
    turn_result_template = next(iter(turn_eval_data.values()))
    num_annotators = turn_result_template[0]["conv_consistency"].shape[0]
    annotator_data = {
        'conv_consistency': np.empty((num_annotators, 0), dtype=float),
        'backend_consistency': np.empty((num_annotators, 0), dtype=float),
        'policy_completeness': np.empty((num_annotators, 0), dtype=float)
    }
    # compile turn scores
    for annotator_turns_scores in turn_eval_data.values():
        for turn_score in annotator_turns_scores:
            # skip turn score if any negative/invalid scores exist
            all_scores = np.concat((turn_score['conv_consistency'], turn_score['backend_consistency'], turn_score['policy_completeness']))
            if np.any(all_scores <= 0):
                continue
            # compile reliability matrices for inter-annotator agreement
            for metric, score in turn_score.items():
                annotator_data[metric] = np.hstack((annotator_data[metric], np.reshape(score, shape=(num_annotators,1))))
    # compile dialogue scores
    for annotator_dials_score in dial_eval_data.values():
        # skip turn score if any negative/invalid scores exist
        all_scores = np.concat((annotator_dials_score['conv_consistency'], annotator_dials_score['backend_consistency'], annotator_dials_score['policy_completeness']))
        if np.any(all_scores <= 0):
            continue
        for metric, score in annotator_dials_score.items():
            # compile reliability matrices for inter-annotator agreement
            annotator_data[metric] = np.hstack((annotator_data[metric], np.reshape(score, shape=(num_annotators,1))))

    for metric in ['conv_consistency', 'backend_consistency', 'policy_completeness']:
        # calculate inter-annotator agreement
        annotator_array = annotator_data[metric]
        k_alpha = calculate_krippendorff_alpha(annotator_array)
        processed_data = annotator_array.T.astype(np.int64) - 1
        # c_kappa = calculate_cohen_kappa(processed_data, 5)
        fleiss = calculate_fleiss_kappa(processed_data, 5)


        # try to calculate spearman and peardson correlation (multiple vars?)
        # pear_stat, pear_pval = stats.pearsonr(x=annotator_array, axis=0)
        # print("pearson: ", pear_stat, pear_pval)
        # spear_stat, spear_pval = stats.spearmanr(annotator_array, axis=0)
        # print("spearman: ", spear_stat, spear_pval)


        metrics[metric] = {
            'k_alpha': float(k_alpha) if k_alpha is not None else 0.0,
            'f_kappa': float(fleiss) if fleiss is not None else 0.0,
            # "spearman": {
            #     'coeff': spear_stat,
            #     'p_value': spear_pval
            # },
            # 'c_kappa': float(c_kappa) if c_kappa is not None else 0.0
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

def human_eval_process(human_eval_csv, autotod_mwoz_eval_json, tau_air_eval_json, tau_retail_eval_json, dial_batches):
    """Process human evaluation CSV and compare with LLM evaluation"""
    # load dialogues
    with open(autotod_mwoz_eval_json, 'r') as f:
        autotod_mwoz_eval = json.load(f)
    autotod_mwoz_dials = autotod_mwoz_eval.get('dialogues', [])
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
    batch_dialogues = get_batch_dialogues(autotod_mwoz_dials, tau_air_dials, tau_retail_dials, batch_list)
    turn_eval_data, dial_eval_data = process_extracted_human_csv_data(human_eval_csv, batch_dialogues, batch_order)
    agreement_metrics = calculate_irr(turn_eval_data, dial_eval_data)
    return agreement_metrics

def main():
    parser = argparse.ArgumentParser(description='Process human and LLM evaluations')
    parser.add_argument('--human_eval_csv', help='Path to human evaluation CSV file')
    parser.add_argument('--autotod_mwoz_eval_json', help='Path to LLM evaluation on MultiWOZ samples JSON file')
    parser.add_argument('--tau_air_eval_json', help='Path to LLM evaluation on Tau bench airline samples JSON file')
    parser.add_argument('--tau_retail_eval_json', help='Path to LLM evaluation on Tau bench retail samples JSON file')
    parser.add_argument('--dial_batch_json', default='batches.json', help='Path to evaluated dialogue batches JSON file')
    
    args = parser.parse_args()
    results = human_eval_process(args.human_eval_csv, args.autotod_mwoz_eval_json, args.tau_air_eval_json, args.tau_retail_eval_json, args.dial_batch_json)
    # save results
    output_dir = 'agreement_scores/' + os.path.basename(args.human_eval_csv).split('.')[0]
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'human_llm_comparison.json'), 'w') as f:
        json.dump(results, f, indent=4)    
    print(f"\nResults saved to {output_dir}")

        
if __name__ == "__main__":
    main()
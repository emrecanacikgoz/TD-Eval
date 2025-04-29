import json
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import re


"""Compare human evaluation data with LLM evaluation data"""    
def extract_score(score_str: str) -> int:
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
                return 1
        return int(match.group(1)) if match else 1
    except:
        print("Score not found in string:",score_str)
        return 1


def calculate_turn_scores(results):
    """Calculate turn-level scores from judge results."""
    all_scores = {
        'conv_consistency': [],
        'backend_consistency': [],
        'policy_completeness': []
    }
    dialogues = results.get('dialogues', [])
    
    for dial_id, dial_turns in dialogues.items():
        for turn in dial_turns:
            # Scores scaled 1-5
            scores = turn["scores"]
            conv_consistency_score = extract_score(scores['conv_consistency'].get('score', 'Score: -1'))
            backend_consistency_score = extract_score(scores['backend_consistency'].get('score', 'Score: -1'))
            policy_completeness_score = extract_score(scores['policy_completeness'].get('score', 'Score: -1'))
            
            # if any invalid scores, skip this turn
            if min(conv_consistency_score, backend_consistency_score, policy_completeness_score) < 0:
                print("error index:", dial_id)
                continue
                
            # append to score
            all_scores['conv_consistency'].append(conv_consistency_score)
            all_scores['backend_consistency'].append(backend_consistency_score)
            all_scores['policy_completeness'].append(policy_completeness_score)
            
    return all_scores


def generate_boxplots(scores, output_dir):
    """Generate boxplots for score distributions."""
    plt.figure(figsize=(12, 6))
    boxplot_data = [
        scores['conv_consistency'], 
        scores['backend_consistency'], 
        scores['policy_completeness']
    ]
    plt.boxplot(boxplot_data)
    plt.title('Distribution of Dialogue Turn Scores (1-5 Scale)')
    plt.xticks(range(1, 4), [
        'Conversation Consistency', 
        'Backend Knowledge Consistency', 
        'Policy Completeness', 
    ])
    plt.ylabel('Score (1-5)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_boxplot_1_5.png'))
    plt.close()


def generate_histograms(scores, output_dir):
    """Generate histograms for score distributions."""
    plt.figure(figsize=(15, 10))
    plt.suptitle('Histograms of Dialogue Turn Scores (1-5 Scale)')
    
    metrics_1_5 = ['conv_consistency', 'backend_consistency', 'policy_completeness']
    
    for i, metric in enumerate(metrics_1_5, 1):
        plt.subplot(2, 2, i)
        plt.hist(scores[metric], bins=np.linspace(1, 5, 5), edgecolor='black')
        plt.title(metric.replace('_', ' ').title())
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.xlim(1, 5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_histograms_1_5.png'))
    plt.close()


def calculate_cumulative_scores(scores, results):
    """Calculate cumulative scores across all turns and dialogues."""
    cumulative_scores = {
        'avg_conv_consistency': np.mean(scores['conv_consistency']),
        'avg_backend_consistency': np.mean(scores['backend_consistency']),
        'avg_policy_completeness': np.mean(scores['policy_completeness']),
        'total_dialogues': len(results.get('dialogues', {})),
        'total_turns': sum(len(turns) for turns in results.get('dialogues', {}).values())
    }
    return cumulative_scores


def generate_summary_report(results, cumulative_scores, output_dir):
    """Generate a summary report with metadata and cumulative scores."""
    metadata = results.get('metadata', {})    
    
    # JSON report
    json_report_path = os.path.join(output_dir, 'summary_report.json')
    summary_report = {
        "metadata": metadata,
        "cumulative_scores": {key: round(value, 4) for key, value in cumulative_scores.items()}
    }
    
    with open(json_report_path, 'w') as f:
        json.dump(summary_report, f, indent=4)


def postprocess_results(result_path, output_dir=None):
    """Postprocess judge results to generate visualizations and summary statistics."""
    with open(result_path, 'r') as f:
        results = json.load(f)
        
    if output_dir is None:
        output_dir = os.path.join('results', 'postprocess_' + os.path.basename(result_path).split('.')[0])
        
    os.makedirs(output_dir, exist_ok=True)
    
    scores = calculate_turn_scores(results)
    generate_boxplots(scores, output_dir)
    generate_histograms(scores, output_dir)
    
    cumulative_scores = calculate_cumulative_scores(scores, results)
    generate_summary_report(results, cumulative_scores, output_dir)
    
    print(f"Postprocessing results saved to {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Postprocess dialogue evaluation results')
    parser.add_argument('--result_path', type=str, required=True, 
                        help='Path to the results JSON file')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='Directory to save output files')
    
    args = parser.parse_args()
    postprocess_results(args.result_path, args.output_dir) 
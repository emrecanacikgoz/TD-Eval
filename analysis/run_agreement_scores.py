from calculate_annotator_agreement import human_eval_process
import os
import numpy as np
import json
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

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
    
def grab_turn_scores(results):
    scores = {
        'conv_consistency': [],
        'backend_consistency': [],
        'policy_completeness': []
    }
    dial_ind = 0
    total_dials_eval = min(len(results.get('dialogues', [])), 30)
    for dialogue in results.get('dialogues', []):
        if dial_ind == total_dials_eval:
            break
        for turn in dialogue.get('turn_scores', []):
            # Scores scaled 1-5
            conv_consistency_score = extract_score(turn.get('conv_consistency', 'Score: -1'))
            backend_consistency_score = extract_score(turn.get('backend_consistency', 'Score: -1'))
            policy_completeness_score = extract_score(turn.get('policy_completeness', 'Score: -1'))
            # if any invalid scores, skip this turn
            if min(conv_consistency_score, backend_consistency_score, policy_completeness_score) < 0:
                # print("error index:", dialogue["idx"])
                continue
            # append to score
            scores['conv_consistency'].append(conv_consistency_score)
            scores['backend_consistency'].append(backend_consistency_score)
            scores['policy_completeness'].append(policy_completeness_score)
        dial_ind += 1
    return scores

def calculate_scores(scores, elo_score):
    conv_consistency_score = np.mean(np.array(scores['conv_consistency']))
    backend_consistency_score = np.mean(np.array(scores['backend_consistency']))
    policy_completeness_score = np.mean(np.array(scores['policy_completeness']))
    return conv_consistency_score/8.3 + backend_consistency_score/8.3 + policy_completeness_score/8.3 + elo_score/300.0


# Heatmap of model judgement scores (numerical + elo)
models = ["gpt4o", "llama405b", "mistrallarge", "qwen72b",  "sonnet"]
score_matrix = np.zeros((5, 5))
elo_scores = None
with open("elo_all_models.json", 'r') as f:
    elo_scores = json.load(f)
if elo_scores == None:
    print("elo failed to load")
    exit()
elo_scores = elo_scores["results"]

file_path_prefix = "results"
for c_ind, c in enumerate(models):
    for j_ind, j in enumerate(models):
        fname = f"{c}_c-{j}_j"
        fpath = os.path.join(file_path_prefix,  fname)
        fpath = os.path.join(fpath, f"{fname}.json")
        if not os.path.isfile(fpath):
            continue
        results = None
        with open(fpath, 'r') as f:
            results = json.load(f)
        if results == None:
            print("results failed to grab:", fpath)
            exit()
        turn_scores = grab_turn_scores(results)
        score = calculate_scores(turn_scores, elo_scores[c]["elo"])
        score_matrix[c_ind, j_ind] = score

print("score matrix:")
print(score_matrix)

# Create the subplots=
mask = (score_matrix == 0)
# Define a custom colormap: Red -> Yellow -> Green
custom_cmap = LinearSegmentedColormap.from_list(
    "custom_cmap", ["red", "yellow", "green"], N=256
)
vmin, vmax = 7, 9
sns.heatmap(score_matrix, annot=True, mask=mask, cmap=custom_cmap, fmt=".3f", xticklabels=models, yticklabels=models, vmin=vmin, vmax=vmax)#, ax=axes[0])
plt.gca().set_facecolor("gray") #axes[0].set_facecolor("gray")  # Black out masked cells
# Add axis labels
plt.tick_params(axis='x', rotation=0, labelsize=8) #axes[0].tick_params(axis='x', rotation=0, labelsize=8)
plt.tick_params(axis='y', rotation=90, labelsize=8) # axes[0].tick_params(axis='y', rotation=90, labelsize=8)
plt.xlabel("Judge", fontsize=9) # axes[0].set_xlabel("Judge", fontsize=9)
plt.ylabel("Agent", fontsize=9) # axes[0].set_ylabel("Agent", fontsize=9)
plt.title("TD-Eval Score For Agent-Judge", fontsize=12) # axes[0].set_title("TD-Eval Score For Agent-Judge", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join("heatmaps", f'td-score.png'))
plt.close()

# Heatmap of human-llm agreement 
conv_consistency_matrix = np.zeros((5, 5))
backend_consistency_matrix = np.zeros((5, 5))
policy_completeness_matrix = np.zeros((5, 5))
annotator_file_prefix = "annotator_results"
file_path_prefix = "results"
for c_ind, c in enumerate(models):
    for j_ind, j in enumerate(models):
        fname = f"{c}_c-{j}_j"
        fpath = os.path.join(file_path_prefix,  fname)
        fpath = os.path.join(fpath, f"{fname}.json")
        if not os.path.isfile(fpath):
            continue
        results = None
        with open(fpath, 'r') as f:
            results = json.load(f)
        if results == None:
            print("results failed to grab:", fpath)
            exit()
        print(c, j)
        print("dial size:", len(results.get('dialogues', [])))
        annotator_file = f"human_{c}_results.csv"
        annotator_fpath = os.path.join(annotator_file_prefix,  annotator_file)
        scores = human_eval_process(annotator_fpath, fpath, 'batches.json')
        conv_consistency_matrix[c_ind, j_ind] = scores["agreement_metrics"]["annotator_llm_agreement"]['conv_consistency']['k_alpha']['mean']
        backend_consistency_matrix[c_ind, j_ind] = scores["agreement_metrics"]["annotator_llm_agreement"]['backend_consistency']['k_alpha']['mean']
        policy_completeness_matrix[c_ind, j_ind] = scores["agreement_metrics"]["annotator_llm_agreement"]['policy_completeness']['k_alpha']['mean']

print("agreement matrix:")
print(conv_consistency_matrix)

# Mask the zeros (values to be blacked out)
mask = (conv_consistency_matrix == 0)
# Define a custom colormap: Red -> Yellow -> Green
custom_cmap = LinearSegmentedColormap.from_list(
    "custom_cmap", ["red", "yellow", "green"], N=256
)
sns.heatmap(conv_consistency_matrix, annot=True, mask=mask, cmap=custom_cmap, fmt=".3f", xticklabels=models, yticklabels=models)#, ax=axes[1])
plt.gca().set_facecolor("gray") # axes[1].set_facecolor("gray")  # Black out masked cells
# Add axis labels
plt.tick_params(axis='x', rotation=0, labelsize=8) # axes[1].tick_params(axis='x', rotation=0, labelsize=8)
plt.tick_params(axis='y', rotation=90, labelsize=8) # axes[1].tick_params(axis='y', rotation=90, labelsize=8)
plt.xlabel("Judge", fontsize=9) # axes[1].set_xlabel("Judge", fontsize=9)
plt.ylabel("Agent", fontsize=9) # axes[1].set_ylabel("Agent", fontsize=9)
plt.title("Human Agreement For Agent-Judge (Conversation Consistency)", fontsize=10) # axes[1].set_title("Human Agreement For Agent-Judge (Krippendorff's Alpha)", fontsize=10)
# Adjust spacing between heatmaps
plt.tight_layout()
plt.savefig(os.path.join("heatmaps", f'conv_consistency.png'))
plt.close()

# Mask the zeros (values to be blacked out)
mask = (backend_consistency_matrix == 0)
# Define a custom colormap: Red -> Yellow -> Green
custom_cmap = LinearSegmentedColormap.from_list(
    "custom_cmap", ["red", "yellow", "green"], N=256
)
sns.heatmap(backend_consistency_matrix, annot=True, mask=mask, cmap=custom_cmap, fmt=".3f", xticklabels=models, yticklabels=models)
plt.gca().set_facecolor("gray") # axes[1].set_facecolor("gray")  # Black out masked cells
# Add axis labels
plt.tick_params(axis='x', rotation=0, labelsize=8) # axes[1].tick_params(axis='x', rotation=0, labelsize=8)
plt.tick_params(axis='y', rotation=90, labelsize=8) # axes[1].tick_params(axis='y', rotation=90, labelsize=8)
plt.xlabel("Judge", fontsize=9) # axes[1].set_xlabel("Judge", fontsize=9)
plt.ylabel("Agent", fontsize=9) # axes[1].set_ylabel("Agent", fontsize=9)
plt.title("Human Agreement For Agent-Judge (Backend Knowledge Consistency)", fontsize=10) # axes[1].set_title("Human Agreement For Agent-Judge (Krippendorff's Alpha)", fontsize=10)
# Adjust spacing between heatmaps
plt.tight_layout()
plt.savefig(os.path.join("heatmaps", f'backend_consistency.png'))
plt.close()

# Mask the zeros (values to be blacked out)
mask = (policy_completeness_matrix == 0)
# Define a custom colormap: Red -> Yellow -> Green
custom_cmap = LinearSegmentedColormap.from_list(
    "custom_cmap", ["red", "yellow", "green"], N=256
)
sns.heatmap(policy_completeness_matrix, annot=True, mask=mask, cmap=custom_cmap, fmt=".3f", xticklabels=models, yticklabels=models)
plt.gca().set_facecolor("gray") # axes[1].set_facecolor("gray")  # Black out masked cells
# Add axis labels
plt.tick_params(axis='x', rotation=0, labelsize=8) # axes[1].tick_params(axis='x', rotation=0, labelsize=8)
plt.tick_params(axis='y', rotation=90, labelsize=8) # axes[1].tick_params(axis='y', rotation=90, labelsize=8)
plt.xlabel("Judge", fontsize=9) # axes[1].set_xlabel("Judge", fontsize=9)
plt.ylabel("Agent", fontsize=9) # axes[1].set_ylabel("Agent", fontsize=9)
plt.title("Human Agreement For Agent-Judge (Policy Completeness)", fontsize=10) # axes[1].set_title("Human Agreement For Agent-Judge (Krippendorff's Alpha)", fontsize=10)
# Adjust spacing between heatmaps
plt.tight_layout()
plt.savefig(os.path.join("heatmaps", f'policy_completeness.png'))
plt.close()

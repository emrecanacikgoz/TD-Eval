import argparse
import json
import os
import qualtrics_utils as qu

def add_tau_q():
    return ""

def add_mwoz_q(batch_path: str, result_path: str):
    # load data
    data = None
    with open(result_path, 'r') as fRes:
        data = json.load(fRes)
    if data is None:
        print('No data found at this path:', result_path)
        exit()

    # load batches
    batches = None
    with open(batch_path, 'r') as fBatch:
        batches = json.load(fBatch)
    if batches is None:
        print('No batches found at this path:', batch_path)
        exit()

    # # validate batches
    batch_dials = {}
    dialogues = data["dialogues"]
    for dial_id, dial_turns in dialogues.items():
        if dial_id in batches:
            batch_dials[dial_id] = dial_turns

    # read to txt file
    survey_output = qu.survey_header + qu.instr_prefix
    # read dialogues
    for dial_id, dial_turns in batch_dials.items():
        survey_output += qu.dial_block_prefix.format(index=dial_id)
        turn_eval_hist = ""
        dial_eval_hist = ""
        for turn in dial_turns:
            user_turn = f"<strong>User:</strong> {turn['user']} <br/>\n"
            dial_eval_hist += user_turn
            turn_eval_hist += user_turn
            # db result from active domain
            domain = turn["domain"]
            db_str = ""
            if domain in turn["db"]:
                db_results = turn["db"][domain]
                db_count = f"<br/>\nCount: {db_results["count"]}\n"
                db_items = "<ul>\n"
                if "sample" in db_results:
                    for items in db_results["sample"]:
                        db_items += f"\n<li>{str(items)}</li>"
                else:
                    for items in db_results["results"]:
                        db_items += f"\n<li>{str(items)}</li>"
                db_items += "\n</ul>"
                db_str = f"{db_count}{db_items}"
            else:
                db_str = "Nothing Found\n"
            # db = str(turn["db"][domain]) if domain in turn["db"] else str({})
            db = f"<strong>Database Result</strong>: {db_str} <br/>\n"
            turn_eval_hist += db
            agent_resp = turn['lex_response']
            survey_output += qu.matrix_q_format.format(dial_hist=dial_eval_hist, db_results=db, agent_resp=agent_resp)
            resp = f"<strong>Agent:</strong> {turn['lex_response']} <br/>\n"  
            gt_resp = f"<strong>Agent:</strong> {turn['ground_truth']} <br/>\n"
            turn_eval_hist += gt_resp
            dial_eval_hist += resp    
        survey_output += qu.dial_completion_rate_q.format(dial_hist=turn_eval_hist)
        survey_output += qu.dial_satisfaction_rate_q.format(dial_hist=turn_eval_hist)
    return survey_output

def main(batch_path: str, result_path: str, output_name: str):
    # load data
    survey_output = add_mwoz_q(batch_path, result_path)

    # read to txt file
    output_path = os.path.join("qualtrics", output_name)
    with open(output_path, 'w') as fOut:
        fOut.write(survey_output)

# example: python3 convert_qualtrics.py --result_path results/openai/gpt4omini-c_gpt4omini-j.json --output_path qualtrics.txt
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Import Jsonl results to qualtrics human evaluation')
    parser.add_argument('--result_path', type=str, help='path to jsonl result file')
    parser.add_argument('--batch_path', type=str, default='datasets/batch.jsonl', help='path to dialogue batches for qualtrics.')
    parser.add_argument('--output_name', type=str, default='qualtrics.txt', help='output file name for qualtrics txt file')
    args = parser.parse_args()
    main(args.batch_path, args.result_path, args.output_name)


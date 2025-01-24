import argparse
import json
import qualtrics_utils as qu

def main(batch_path: str, result_path: str, output_path: str):
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
    with open(output_path, 'w') as fOut:
        fOut.write(qu.survey_header+qu.instr_prefix)
        # read dialogues
        dial_q_input = ""
        for dial_id, dial_turns in batch_dials.items():
            # dial = data["dialogues"][idx]
            fOut.write(qu.dial_block_prefix.format(index=dial_id))
            dial_hist = ""
            for turn in dial_turns:
                user_turn = f"<strong>User:</strong> {turn['user']} <br/>\n"
                dial_hist += user_turn
                dial_q_input += user_turn
                # db result from active domain
                domain = turn["domain"]
                db = str(turn["db"][domain]) if domain in turn["db"] else str({})
                db = f"<strong>Database Result</strong>: {db} <br/>\n"
                dial_q_input += db
                agent_resp = f"{turn['lex_response']}"
                fOut.write(qu.matrix_q_format.format(dial_hist=dial_hist, db_results=db, agent_resp=agent_resp))
                resp = f"<strong>Agent:</strong> {turn['lex_response']} <br/>\n"  # TODO: should we use ground truth or actual response?
                dial_q_input += resp
                dial_hist += resp    
            fOut.write(qu.dial_completion_rate_q.format(dial_hist=dial_q_input))
            fOut.write(qu.dial_satisfaction_rate_q.format(dial_hist=dial_q_input))

# example: python3 convert_qualtrics.py --result_path results/openai/gpt4omini-c_gpt4omini-j.json --output_path qualtrics.txt
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Import Jsonl results to qualtrics human evaluation')
    parser.add_argument('--result_path', type=str, help='path to jsonl result file')
    parser.add_argument('--batch_path', type=str, default='datasets/batch.jsonl', help='path to dialogue batches for qualtrics.')
    parser.add_argument('--output_path', type=str, default='qualtrics.txt', help='Path to output file for qualtrics txt file')
    args = parser.parse_args()
    main(args.batch_path, args.result_path, args.output_path)


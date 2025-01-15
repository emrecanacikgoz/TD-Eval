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
    batch_dials = []
    for dial in data["dialogues"]:
        dial_id = dial["dialogue_id"]
        if dial_id in batches['dial_ids']:
            batch_dials.append(dial)

    # display metadata
    print(data["metadata"])

    # read to txt file
    with open(output_path, 'w') as fOut:
        fOut.write(qu.survey_header+qu.instr_prefix)
        # read dialogues
        for dial in batch_dials:
            # dial = data["dialogues"][idx]
            dial_id = dial["dialogue_id"]
            fOut.write(qu.dial_block_prefix.format(index=dial_id))
            dial_hist = ""
            for turn in dial["full_conversation"]:
                user_turn = f"<strong>User:</strong> {turn['user']} <br/>\n"
                dial_hist += user_turn
                agent_resp = f"{turn['agent']}"
                fOut.write(qu.matrix_q_format.format(dial_hist=dial_hist, agent_resp=agent_resp))
                gt_resp = f"<strong>Agent:</strong> {turn['agent']} <br/>\n"
                dial_hist += gt_resp    

# example: python3 convert_qualtrics.py --result_path results/openai/gpt4omini-c_gpt4omini-j.json --output_path qualtrics.txt
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Import Jsonl results to qualtrics human evaluation')
    parser.add_argument('--result_path', type=str, help='path to jsonl result file')
    parser.add_argument('--batch_path', type=str, default='batches.json', help='path to dialogue batches for qualtrics.')
    parser.add_argument('--output_path', type=str, default='qualtrics.txt', help='Path to output file for qualtrics txt file')
    args = parser.parse_args()
    main(args.batch_path, args.result_path, args.output_path)


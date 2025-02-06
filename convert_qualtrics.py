import argparse
import json
import os
import qualtrics_utils as qu

def parse_db_tau_tool_call(db_dict: dict):
    db_str = ""
    for f_name, f_args in db_dict.items():
        db_fname = f"<br/>\nFunction: {f_name}\n"
        db_fargs = f'<br/>\nArgs: {f_args["args"]}\n'
        db_fout = f_args["output"]

        try:
            db_results = json.loads(db_fout)
        except json.JSONDecodeError as e:
            print("error:", e)
            print(db_fout)
            db_results = db_fout

        # if isinstance(db_results, str):
        #     db_items = db_fout
        # else: 
        db_items = "<br/>\nOutput:\n"
        if isinstance(db_results, list):
            db_items += "<ul>\n"
            for items in db_results:
                db_items += f"\n<li>{str(items)}</li>"

            db_items += "\n</ul>"
        else:
            db_items += f"<ul>\n<li>{db_results}</li>\n</ul>"
        db_str += f"{db_fname}{db_fargs}{db_items}<br/>\n"
    return db_str

def parse_db_tau_react(db_dict: dict):
    db_str = ""
    for turn, funcs in db_dict.items():
        for f in funcs:
            db_fname = f"<br/>\nFunction: {f["name"]}\n"
            db_fargs = f'<br/>\nArgs: {f["args"]}\n'
            db_fout = f["output"] if "output" in f else "{}"

            try:
                db_results = json.loads(db_fout)
            except json.JSONDecodeError as e:
                print("error:", e)
                print(db_fout)
                db_results = db_fout

            # if isinstance(db_results, str):
            #     db_items = db_fout
            # else: 
            db_items = "<br/>\nOutput:\n"
            if isinstance(db_results, list):
                db_items += "<ul>\n"
                for items in db_results:
                    db_items += f"\n<li>{str(items)}</li>"

                db_items += "\n</ul>"
            else:
                db_items += f"<ul>\n<li>{db_results}</li>\n</ul>"
            db_str += f"{db_fname}{db_fargs}{db_items}<br/>\n"
    return db_str

def add_tau_q(batch: dict, tau_retail_path: str, tau_airline_path: str, tau_react: bool):
    # load data
    retail_data = None
    with open(tau_retail_path, 'r') as fRes:
        retail_data = json.load(fRes)
    if retail_data is None:
        print('No data found at this path:', tau_retail_path)
        exit()

    airline_data = None
    with open(tau_airline_path, 'r') as fRes:
        airline_data = json.load(fRes)
    if airline_data is None:
        print('No data found at this path:', tau_airline_path)
        exit()

    # validate batches on tau
    batch_dials = {}
    tau_batch_ids = batch["tau"]

    tau_retail_dialogues = retail_data["dialogues"]
    for dial_id, dial_turns in tau_retail_dialogues.items():
        if dial_id in tau_batch_ids["retail"]:
            batch_dials[dial_id] = dial_turns

    tau_air_dialogues = airline_data["dialogues"]
    for dial_id, dial_turns in tau_air_dialogues.items():
        if dial_id in tau_batch_ids["airline"]:
            batch_dials[dial_id] = dial_turns

    # read to txt file
    survey_output = "" #qu.survey_header + qu.instr_prefix
    # read dialogues
    for dial_id, dial_turns in batch_dials.items():
        survey_output += qu.dial_block_prefix.format(index=dial_id)
        turn_eval_hist = ""
        dial_eval_hist = ""
        for turn in dial_turns:
            user_str = turn['user'].replace('Customer:', "")
            user_turn = f"<strong>User:</strong> {user_str} <br/>\n"

            dial_eval_hist += user_turn
            turn_eval_hist += user_turn

            db_str = ""
            if not turn["db"]:
                db_str = "Nothing Found"
            else:
                if tau_react:
                    db_str = parse_db_tau_react(turn["db"])
                else:
                    db_str = parse_db_tau_tool_call(turn["db"])
            # db = str(turn["db"][domain]) if domain in turn["db"] else str({})
            db = f"<strong>Database Result</strong>: {db_str} <br/>\n"
            # turn_eval_hist += db
            # agent_resp = turn['response']

            resp_str = turn['response'].replace('Agent:', "")
            survey_output += qu.matrix_q_format.format(dial_hist=dial_eval_hist, db_results=db, agent_resp=resp_str)
            
            resp = f"<strong>Agent:</strong> {resp_str} <br/>\n"  
            turn_eval_hist += resp
            dial_eval_hist += resp    
        survey_output += qu.dial_completion_rate_q.format(dial_hist=turn_eval_hist)
        survey_output += qu.dial_satisfaction_rate_q.format(dial_hist=turn_eval_hist)
    return survey_output
    return ""

def add_mwoz_q(batch: dict, mwoz_path: str):
    # load data
    data = None
    with open(mwoz_path, 'r') as fRes:
        data = json.load(fRes)
    if data is None:
        print('No data found at this path:', mwoz_path)
        exit()
    # validate batches on mwoz
    batch_dials = {}
    mwoz_batch_ids = batch["mwoz"]
    dialogues = data["dialogues"]
    for dial_id, dial_turns in dialogues.items():
        if dial_id in mwoz_batch_ids:
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
                db_count = f'<br/>\nCount: {db_results["count"]}\n'
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
                db_str = "Nothing Found"
            # db = str(turn["db"][domain]) if domain in turn["db"] else str({})
            db = f"<strong>Database Result</strong>: {db_str} <br/>\n"
            # turn_eval_hist += db
            agent_resp = turn['lex_response']
            survey_output += qu.matrix_q_format.format(dial_hist=turn_eval_hist, db_results=db, agent_resp=agent_resp)
            resp = f"<strong>Agent:</strong> {turn['lex_response']} <br/>\n"  
            gt_resp = f"<strong>Agent:</strong> {turn['ground_truth']} <br/>\n"
            turn_eval_hist += gt_resp
            dial_eval_hist += resp    
        survey_output += qu.dial_completion_rate_q.format(dial_hist=dial_eval_hist)
        survey_output += qu.dial_satisfaction_rate_q.format(dial_hist=dial_eval_hist)
    return survey_output

def main(batch_path: str, mwoz_path: str, tau_retail_path: str, tau_airline_path: str, tau_react: bool, output_name: str):
    # load batches
    batch = None
    with open(batch_path, 'r') as fBatch:
        batch = json.load(fBatch)
    if batch is None:
        print('No batches found at this path:', batch_path)
        exit()
    mwoz_survey_output = add_mwoz_q(batch, mwoz_path)
    tau_survey_output = add_tau_q(batch, tau_retail_path, tau_airline_path, tau_react)
    # read to txt file
    output_path = os.path.join("qualtrics", output_name)
    with open(output_path, 'w') as fOut:
        fOut.write(mwoz_survey_output)
        fOut.write(tau_survey_output)

# example: python3 convert_qualtrics.py --result_path results/openai/gpt4omini-c_gpt4omini-j.json --output_path qualtrics.txt
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Import Jsonl results to qualtrics human evaluation')
    parser.add_argument('--batch_path', type=str, default='datasets/batch.jsonl', help='path to dialogue batches for qualtrics.')
    parser.add_argument('--mwoz_path', type=str, help='path to jsonl result file')
    parser.add_argument('--tau_retail_path', type=str, help='path to jsonl result file')
    parser.add_argument('--tau_airline_path', type=str, help='path to jsonl result file')
    parser.add_argument('--tau_react', action='store_true', help='extraction changes to match tau bench reat formatting')
    parser.add_argument('--output_name', type=str, default='qualtrics.txt', help='output file name for qualtrics txt file')
    args = parser.parse_args()
    main(args.batch_path, args.mwoz_path, args.tau_retail_path, args.tau_airline_path, args.tau_react, args.output_name)


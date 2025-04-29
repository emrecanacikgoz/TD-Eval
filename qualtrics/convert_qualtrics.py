import argparse
import json
import os
import re
import qualtrics_utils as qu
import markdown

def stringify_dial_hist(dial_eval_hist: list) -> str:
    """
    Convert the dialogue evaluation history to a string format.
    """
    dial_hist_str = ""
    for i, turn in enumerate(dial_eval_hist):
        if "user" in turn:
            user_str = turn['user']
            dial_hist_str += user_str
        if "response" in turn:
            resp_str = turn['response']
            dial_hist_str += resp_str

    return dial_hist_str

def parse_db_tau_tool_call(db_dict: dict):
    db_str = ""
    for f_name, f_args in db_dict.items():
        db_fname = f"<br/>\n<i>Function</i>: {f_name}\n"
        db_fargs = f'<br/>\n<i>Args</i>: {f_args["args"]}\n'
        db_fout = f_args["output"]

        try:
            db_results = json.loads(db_fout)
        except json.JSONDecodeError as e:
            print("error:", e)
            print(db_fout)
            db_results = db_fout

        db_items = "<br/>\n<i>Output</i>:\n"
        if isinstance(db_results, list):
            db_items += "<ul>\n"
            for items in db_results:
                db_items += f"\n<li>{str(items)}</li>"
            db_items += "\n</ul>"
        else:
            if db_results != "":
                db_items += f"<ul>\n<li>{db_results}</li>\n</ul>"
            else:
                db_items += "No output data"
        db_str += f"{db_fname}{db_fargs}{db_items}<br/>\n"
    return db_str

def parse_db_tau_react(db_dict: dict):
    db_str = ""
    for turn, funcs in db_dict.items():
        for f in funcs:
            db_fname = f'<br/>\n<i>Function</i>: {f["name"]}\n'
            db_fargs = f'<br/>\n<i>Args</i>: {f["args"]}\n'
            db_fout = f["output"] if "output" in f else "{}"

            try:
                db_results = json.loads(db_fout)
            except json.JSONDecodeError as e:
                print("error:", e)
                print(db_fout)
                db_results = db_fout

            db_items = "<br/>\n<i>Output</i>:\n"
            if isinstance(db_results, list):
                db_items += "<ul>\n"
                for items in db_results:
                    db_items += f"\n<li>{str(items)}</li>"
                db_items += "\n</ul>"
            else:
                if db_results != "":
                    db_items += f"<ul>\n<li>{db_results}</li>\n</ul>"
                else:
                    db_items += "No output data"
            db_str += f"{db_fname}{db_fargs}{db_items}<br/>\n"
    return db_str

def add_tau_q(batch: dict, tau_retail_path: str, tau_airline_path: str, tau_react: bool) -> str:
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
            batch_dials[f"retail_{dial_id}"] = dial_turns

    tau_air_dialogues = airline_data["dialogues"]
    for dial_id, dial_turns in tau_air_dialogues.items():
        if dial_id in tau_batch_ids["airline"]:
            batch_dials[f'airline_{dial_id}'] = dial_turns

    survey_output = "" 
    # read dialogues
    for dial_id, dial_turns in batch_dials.items():
        survey_output += qu.dial_block_prefix.format(index=dial_id)
        dial_eval_hist = [] 
        for i, turn in enumerate(dial_turns):
            user_str = turn['user'].replace('Customer:', "")
            user_turn = f"<strong>User:</strong> {user_str} <br/>\n"
            dial_eval_hist.append({"user": user_turn})

            db_str = ""
            if not turn["db"]:
                db_str = "Nothing Found"
            else:
                if tau_react:
                    db_str = parse_db_tau_react(turn["db"])
                else:
                    db_str = parse_db_tau_tool_call(turn["db"])
            db = f"<strong>Backend Result</strong>: {db_str} <br/>\n"
            dial_eval_hist[i]["db"] = db

            resp_str = turn['response'].replace('Agent:', "")
            resp_str = markdown.markdown(resp_str, extensions=['extra'])
            resp_str = resp_str.replace("<p>", "", 1).replace("</p>", "", 1)
            if len(dial_eval_hist) == 1:
                turn_hist = dial_eval_hist[0]["user"]
            else:
                prev_agent_resp = dial_eval_hist[-2]["response"]
                user_req = dial_eval_hist[-1]["user"]
                turn_hist = prev_agent_resp + user_req
            survey_output += qu.matrix_q_format.format(dial_hist=turn_hist, db_results=db, agent_resp=resp_str)
            resp = f"<strong>Agent:</strong> {resp_str} <br/>\n"  
            dial_eval_hist[i]["response"] = resp

        dial_hist_str = stringify_dial_hist(dial_eval_hist)
        survey_output += qu.dial_eval_q.format(dial_hist=dial_hist_str)
    return survey_output

def parse_db_autotod_react(backend_str: str) -> str:
    db_str = ""
    api_pattern = r"API Name:(.*?)\nAPI Input:(.*?)\n(?:API Result:(.*?))$"
    # response_pattern = r"Response:(.*?)(?=Thought:|API Name:|\n```|$)"
    db_thoughts = backend_str.split("Thought:")
    for thought in db_thoughts:
        db_api_call = ""
        thought = thought.strip().replace("```", "")
        if len(thought) == 0:
            continue
        thought_message = thought.split("\n")[0]
        db_api_call += f"<strong>Thought</strong>: {thought_message} <br/>\n"
        # try to extract api/db calls
        api_matches = re.findall(api_pattern, thought, re.DOTALL)
        if len(api_matches) > 0:
            api_match = api_matches[0]
            api_name = api_match[0].strip()
            api_input = api_match[1].strip()
            api_result = api_match[2].strip() if api_match[2] else None
            db_fname = f"<i>Function</i>: {api_name}<br/>\n"
            db_fargs = f'<i>Args</i>: {api_input}<br/>\n'
            # try to parse api input and result and json
            try: 
                api_result = json.loads(api_result)
            except (json.JSONDecodeError, TypeError):
                print("fail decode json")
                pass # leave as string
            # parse api output
            db_items = "<i>Output</i>:<br/>"
            if isinstance(api_result, dict):
                db_results = api_result["result"]
                db_items += "\n<ul>"
                if isinstance(db_results, list):
                    for items in db_results:
                        db_items += f"\n<li>{str(items)}</li>"
                elif isinstance(db_results, dict):
                    db_items += f"\n<li>{str(db_results)}</li>"
                db_items += "\n</ul>"
            else: # stays as string
                db_results = api_result
                if db_results != "":
                    db_items += f"<ul>\n<li>{db_results}</li>\n</ul>"
                else:
                    db_items += "No output data"

            db_api_call += f"{db_fname}{db_fargs}{db_items}<br/>\n"
        db_str += f"{db_api_call}"
    return db_str

def add_autotod_q(batch: dict, autotod_path: str) -> str:
    # load data
    autotod_data = None
    with open(autotod_path, 'r') as fRes:
        autotod_data = json.load(fRes)
    if autotod_data is None:
        print('No data found at this path:', autotod_path)
        exit()
    # validate batches on tau
    batch_dials = {}
    autotod_batch_ids = batch["autotod_mwoz"]
    for id, dial in autotod_data.items():
        mwoz_id = id.lower().replace(".json", "")
        if mwoz_id in autotod_batch_ids:
            batch_dials[mwoz_id] = dial["log"]
    # read dialogues
    survey_output = qu.survey_header + qu.instr_prefix
    for dial_id, dial_turns in batch_dials.items():
        survey_output += qu.dial_block_prefix.format(index=dial_id)
        dial_eval_hist = [] 
        for i, turn in enumerate(dial_turns):
            # parse user turn
            user_str = turn['usr'].strip().replace("```", "")
            user_turn = f"<strong>User:</strong> {user_str} <br/>\n"
            dial_eval_hist.append({"user": user_turn}) 

            # parse db result
            db_str = ""
            db_str = parse_db_autotod_react(turn["answers"])
            db = f"<strong>Backend Result</strong>: <br/>\n {db_str}"
            dial_eval_hist[i]["db"] = db

            # parse agent response
            resp_str = turn['response'].strip().replace("```", "")
            print(resp_str)
            resp_str = markdown.markdown(resp_str, extensions=['extra'])
            resp_str = resp_str.replace("<p>", "", 1).replace("</p>", "", 1)
            if len(dial_eval_hist) == 1:
                turn_hist = dial_eval_hist[0]["user"]
            else:
                prev_agent_resp = dial_eval_hist[-2]["response"]
                user_req = dial_eval_hist[-1]["user"]
                turn_hist = prev_agent_resp + user_req
            survey_output += qu.matrix_q_format.format(dial_hist=turn_hist, db_results=db, agent_resp=resp_str)
            resp = f"<strong>Agent:</strong> {resp_str} <br/>\n"  
            print(resp)

            dial_eval_hist[i]["response"] = resp

        dial_hist_str = stringify_dial_hist(dial_eval_hist)
        survey_output += qu.dial_eval_q.format(dial_hist=dial_hist_str)
    return survey_output

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
        dial_eval_hist = [] #""
        for i, turn in enumerate(dial_turns):
            user_turn = f"<strong>User:</strong> {turn['user']} <br/>\n"
            dial_eval_hist.append({"user": user_turn}) 

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
            db = f"<strong>Backend Result</strong>: {db_str} <br/>\n"
            dial_eval_hist[i]["db"] = db

            agent_resp = turn['lex_response']
            resp_str = markdown.markdown(agent_resp, extensions=['extra'])
            resp_str = resp_str.replace("<p>", "", 1).replace("</p>", "", 1)
            if len(dial_eval_hist) == 1:
                turn_hist = dial_eval_hist[0]["user"]
            else:
                prev_agent_resp = dial_eval_hist[-2]["response"]
                user_req = dial_eval_hist[-1]["user"]
                turn_hist = prev_agent_resp + user_req
            survey_output += qu.matrix_q_format.format(dial_hist=turn_hist, db_results=db, agent_resp=resp_str)
            resp = f"<strong>Agent:</strong> {resp_str} <br/>\n"  
            dial_eval_hist[i]["response"] = resp

        dial_hist_str = stringify_dial_hist(dial_eval_hist)
        survey_output += qu.dial_eval_q.format(dial_hist=dial_hist_str)
    return survey_output

def main(batch_path: str, mwoz_path: str, is_autotod: bool, tau_retail_path: str, tau_airline_path: str, tau_react: bool, output_name: str):
    # load batches
    batch = None
    with open(batch_path, 'r') as fBatch:
        batch = json.load(fBatch)
    if batch is None:
        print('No batches found at this path:', batch_path)
        exit()
    if is_autotod:
        mwoz_survey_output = add_autotod_q(batch, mwoz_path)
    else:
        mwoz_survey_output = add_mwoz_q(batch, mwoz_path)
    tau_survey_output = add_tau_q(batch, tau_retail_path, tau_airline_path, tau_react)
    # read to txt file
    output_path = os.path.join("survey_imports", output_name)
    with open(output_path, 'w') as fOut:
        fOut.write(mwoz_survey_output)
        fOut.write(tau_survey_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Import Jsonl results to qualtrics human evaluation')
    parser.add_argument('--batch_path', type=str, default='../datasets/batch.jsonl', help='path to dialogue batches for qualtrics.')
    parser.add_argument('--mwoz_path', type=str, help='path to jsonl result file')
    parser.add_argument('--is_autotod', action='store_true', help='path to mwoz jsonl result file')
    parser.add_argument('--tau_retail_path', type=str, help='path to jsonl result file')
    parser.add_argument('--tau_airline_path', type=str, help='path to jsonl result file')
    parser.add_argument('--tau_react', action='store_true', help='extraction changes to match tau bench reat formatting')
    parser.add_argument('--output_name', type=str, default='qualtrics.txt', help='output file name for qualtrics txt file')
    args = parser.parse_args()
    main(args.batch_path, args.mwoz_path, args.is_autotod, args.tau_retail_path, args.tau_airline_path, args.tau_react, args.output_name)


import argparse
import json
import os

def tau_stats(data, domain):
    dials = data["dialogues"]
    dial_out = ""
    for dial_id, dial_data in dials.items():
        num_turns = len(dial_data)
        dial_out += f"id: {dial_id:<11}turns: {num_turns:<8}domain: {domain}\n"
    return dial_out

def mwoz_autotod_stats(data):
    dial_out = ""
    for id, data in data.items():
        dial_id = id.split(".json")[0].lower()
        domains = list(data['eval_results'].keys())
        num_turns = len(data["run_result"]["dialog_pred"])
        dial_out += f"id: {dial_id:<11}turns: {num_turns:<8}domains: {domains}\n"
    return dial_out

def mwoz_stats(data):
    dials = data["dialogues"]
    dial_out = ""
    for dial_id, dial_data in dials.items():
        num_turns = len(dial_data)
        domains = []
        for turn in dial_data:
            domains.append(turn["domain"])
        domains = set(domains)
        dial_out += f"id: {dial_id:<11}turns: {num_turns:<8}domains: {domains}\n"
    return dial_out

def main(judge_model: str, mwoz: str, is_autotod: bool, tau_retail: str, tau_airline: str):
    out_fname = judge_model
    # load data
    with open(mwoz, 'r') as fMowz:
        mwoz_data = json.load(fMowz)
    with open(tau_retail, 'r') as fTauRetail:
        tau_retail_data = json.load(fTauRetail)
    with open(tau_airline, 'r') as fTauAirline:
        tau_airline_data = json.load(fTauAirline)
    if mwoz_data is None or tau_airline is None or tau_retail is None:
        print('No data found at one of the paths')
        exit()
    if is_autotod:
        mwoz_output = mwoz_autotod_stats(mwoz_data)
        out_fname += "-autotod"
    else: 
        mwoz_output = mwoz_stats(mwoz_data)
    tau_retail_output = tau_stats(tau_retail_data, "retail")
    tau_air_output = tau_stats(tau_airline_data, "airline")
    # read to txt file
    output_path = os.path.join("stats", f"{out_fname}.txt")
    with open(output_path, 'w') as fOut:
        fOut.write(mwoz_output + "\n")
        fOut.write(tau_retail_output + "\n")
        fOut.write(tau_air_output)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Import Jsonl results to qualtrics human evaluation')
    parser.add_argument('--judge_model', type=str, help='name of model judging dialogues')
    parser.add_argument('--mwoz', type=str, help='path to mwoz json result file')
    parser.add_argument('--is_autotod', action='store_true', help='Uses autotod mwoz dialogues instead of original mwoz')
    parser.add_argument('--tau_retail', type=str, help='path to tau json result file')
    parser.add_argument('--tau_airline', type=str, help='path to tau json result file')

    args = parser.parse_args()
    main(args.judge_model, args.mwoz, args.is_autotod, args.tau_retail, args.tau_airline)
    print("Finish writing stats")
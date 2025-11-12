import os
import json
import random
import time
import argparse
from pathlib import Path
from typing import Optional
from openai import OpenAI

###

def _init_openai_client(api_key_file: Optional[str] = None) -> OpenAI:
    """Initialize OpenAI client; prefer env var, optionally read from file (like other scripts)."""
    if "OPENAI_API_KEY" not in os.environ:
        file_to_use = api_key_file if api_key_file else ("API_key.txt" if Path("API_key.txt").exists() else None)
        if file_to_use and Path(file_to_use).exists():
            try:
                with open(file_to_use, "r") as f:
                    os.environ["OPENAI_API_KEY"] = f.read().strip()
            except Exception:
                pass
    return OpenAI()

def from_tool_example_directory(path_examples, explanation = False):
    examples = {}
    path_examples = Path(path_examples)
    if explanation:
        for subdir in path_examples.rglob('*'):
            if subdir.is_dir():
                explanation_file = subdir / 'explanation.txt'
                if explanation_file.exists():
                    with open(explanation_file, 'r') as file:
                        explanation_content = file.read()
                    examples[str(subdir.relative_to(path_examples))] = {'explanation': explanation_content}
                    config_file = next(subdir.glob('config*.json'), None)
                    if config_file.exists():
                        with open(config_file, 'r') as file:
                            config_data = json.load(file)
                            examples[str(subdir.relative_to(path_examples))]['config'] = config_data
    else:
        for file_path in path_examples.rglob('*config*'):
            if file_path.is_file() and file_path.suffix == '.json':
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    relative_path = file_path.relative_to(path_examples)
                    examples[str(relative_path)] = data
    return examples


def serialize_examples(examples: dict, n_examples: int, seed=None):
    """ Serialize examples from the Pytheus tool """
    if seed is not None:
        random.seed(seed)
    keys = random.sample(list(examples.keys()), n_examples)
    return "".join(["### EXAMPLE {ind} ###\n{path}\n{config}\n\n".format(path=key, config=examples[key], ind=ii+1) for ii, key in enumerate(keys)])


def build_examples_block(examples_root: str, seed: Optional[int] = None) -> str:
    """Load and serialize all examples from the four categories, preserving current behavior (all)."""
    if seed is not None:
        random.seed(seed)
    root = Path(examples_root)
    all_examples_state = from_tool_example_directory(str(root / 'States'), explanation=True)
    examples_states = serialize_examples(all_examples_state, n_examples=len(all_examples_state), seed=None)
    all_examples_gates = from_tool_example_directory(str(root / 'Gates'), explanation=True)
    examples_gates = serialize_examples(all_examples_gates, n_examples=len(all_examples_gates), seed=None)
    all_examples_measurement = from_tool_example_directory(str(root / 'Measurements'), explanation=True)
    examples_measurement = serialize_examples(all_examples_measurement, n_examples=len(all_examples_measurement), seed=None)
    all_examples_communication = from_tool_example_directory(str(root / 'Communication'), explanation=True)
    examples_communication = serialize_examples(all_examples_communication, n_examples=len(all_examples_communication), seed=None)
    return examples_states + examples_gates + examples_measurement + examples_communication


def parse_feedback_file(feedback_file: Optional[str]) -> str:
    """Parse the optional feedback file into a single block. Gracefully handle missing files."""
    if not feedback_file:
        return ""
    p = Path(feedback_file)
    if not p.exists():
        print(f"Feedback file not found: {feedback_file}. Proceeding without feedback context.")
        return ""
    data = []
    with open(p, 'r') as f:
        for line in f:
            if line.startswith('#####'):
                continue
            elif line.startswith('<START SUGGESTION>'):
                suggestion = []
                while not line.startswith('<END SUGGESTION>'):
                    suggestion.append(line)
                    line = next(f)
                suggestion = suggestion[1:]  # drop <START SUGGESTION>
                suggestion = ''.join(suggestion).strip()
            elif line.startswith('Feedback:'):
                feedback = line.split('Feedback:')[1].strip()
                if feedback == '':
                    feedback = None
                data.append({'suggestion': suggestion, 'feedback': feedback})
    existing_feedback_items = [item for item in data if item['feedback'] is not None]
    print(f"Found {len(existing_feedback_items)} existing feedback items.")
    return '\n\n\n'.join([str(item) for item in existing_feedback_items])




def build_prompt(pytheus_explicit_infos: str, examples_block: str, domain_feedback_block: str, researcher_suggestion: str) -> str:
    feedback_section = f"""
Here is some extremely valuable domain expert feedback on previous suggestions. Use this context to judge which aspects can be improved:

<START DOMAIN EXPERT FEEDBACK>
{domain_feedback_block}
<END DOMAIN EXPERT FEEDBACK>
""" if domain_feedback_block else ""

    return f"""
You are a helpful AI physicist. 

Task: You have access to a tool (pytheus) that can design quantum optics experiments based on a clear target. 
Your colleague (the Researcher) is trying to come up with interesting and novel targets that can be searched for. 
Your colleague does not know the full scope of your tool and may explain their ideas in a way that is not immediately compatible with the tool's capabilities.

Your goal is to understand the essence of the idea and rephrase the pytheus configuration file accordingly to fix mistakes while staying true to the original research question.


Here is some additional information on the capabilities and limitations of pytheus:

<START PYTHEUS INFOS>
{pytheus_explicit_infos}
<END PYTHEUS INFOS>


Here are existing examples of experiments implemented with pytheus.

<START PYTHEUS EXAMPLES>
{examples_block}
<END PYTHEUS EXAMPLES>


The configuration file for running the tool defined through the following key words:
* description: a short description of what the experiment is/does
* target_state: this defines the target measurement, transformation, or protocol and is one of the most critical components in the search. `target_state` is specified as a list of kets. For fock (photon number) states, it lists only photon counts per mode, such as [[3,0], [0,3]] representing a three-photon, two-mode NOON state. In contrast, for regular states with constant photon numbers and specific modes, target_state is given as a list of strings like ["30", "03"].
* samples: number of optimization runs (samples=1 for simple tasks, larger for more difficult tasks)
* optimizer: name of the scipy optimizer to be used
* loss_func: this is the function pytheus optimizes to reach the target, and it determines whether the `target_state` is treated in the Fock basis or discrete variable basis. For Fock states, loss_func uses 'fockfid' or 'fockcr'. For discrete variable states, loss_func uses 'fid' (fidelity), 'cr' (count rate), 'ent' (entanglement) and 'lff' ('lff' is for a custom loss function).
* num_anc: number of ancilla photons added to realize the target. Try to reason about how many ancillary particles you will need. Too many and the optimization will take very long, too few and it may not be possible to realize the target.
* edges_tried and tries_per_edge: parameters for topological optimization
* amplitudes (optional): a list of coefficients/amplitude for the kets. if none are given, they are assumed to be all 1.
* imaginary  (optional): determines if coefficients are given in real or complex format. default is real numbers ('false'), otherwise 'cartesian' or 'polar' (which implies 'true').
* in_nodes (optional): for quantum gates and measurements, there can be incoming photons. this list of numbers determines at which position they are. default is empty (no incoming).
* out_nodes (optional): sometimes it needs to be specified which photons are the outgoing photons. if this is left empty, it is assumed that all photons of the target state are outgoing.
* heralding_out (optional): if this is set to true, only the ancilla photons are detected, meaning that the out_nodes remain undetected, which is good to have if we want to use the state further instead of being destroyed. default is false (all photons detected)
* single_emitters (optional): single photon emitters are an experimental resource that can be used for many experiments. this list defines their position. default is empty (no single photon emitters)
* removed_connections  (optional): for some tasks we might want to specify paths, which should not share a common source. this is done in a list of pairs. default is empty (no restrictions)
* number_resolving (optional): we can specify to use number resolving detectors. this gives additional information, because regular detectors can only distinguish between no photon or at least one photon.
* thresholds (optional): These are the values the loss function has to go below to be considered good enough. Higher values accept/tolerate more solutions. Example values for loss_func 'fid' are [0.1, 1] and for loss_func 'cr' are [0.3, 0.1].

Examples of config files sometimes contain keywords which have values that are null or empty lists. These can be considered to be optional or dynamically set.

{feedback_section}

Here is the idea suggested by the Researcher:
<START OF SUGGESTION>
{researcher_suggestion}
<END OF SUGGESTION>

The Researcher is trying to push the boundaries of what is possible to do with the PyTheus tool, so do not expect trivial variations of existing examples and think out of the box to apply the tool based on what you understand about it. 

You can only respond with a single complete
"Thought, Action, Action Input".

Complete format (don't use additional formatting such as bold text):

Thought: (reflect on the given information and plan out the next steps)
Action: accept
Action Input: (provide the improved configuration json)
"""



def extract_last_researcher_output(ensemble_dir: str, suggestion_timestamp: str) -> str:
    """Find and parse the last non-summarizing Researcher output from the matching run log."""
    ensemble_runs_dir = f'{ensemble_dir}/ensemble_runs'
    ensemble_run_dir = None
    for dd in os.listdir(ensemble_runs_dir):
        if suggestion_timestamp in dd:
            ensemble_run_dir = os.path.join(ensemble_runs_dir, dd)
            print(f'Found ensemble run directory: {ensemble_run_dir}')
            break
    if ensemble_run_dir is None:
        raise ValueError(f'No ensemble run directory found for timestamp {suggestion_timestamp}')

    log_files = list(Path(ensemble_run_dir).glob('log*.txt'))
    if not log_files:
        raise ValueError(f'No log files found in ensemble run directory {ensemble_run_dir}')
    log_file = log_files[0]

    last_researcher_output = None
    # First try: parse per line as JSON objects
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and 'agent' in obj and 'output' in obj:
                    if 'Researcher' in obj['agent'] and 'Summarizing' not in obj['agent']:
                        last_researcher_output = obj['output']
            except Exception:
                continue
    if last_researcher_output is not None:
        return last_researcher_output

    # Fallback: original split heuristic
    with open(log_file, 'r') as f:
        combined = '\n'.join([ln.strip() for ln in f if ln.strip()])
    chunks = combined.split('}\n{')
    for ii in range(len(chunks)):
        try:
            candidate = '{' + chunks[-1 - ii] + '}'
            obj = json.loads(candidate)
            if 'Researcher' in obj.get('agent', '') and 'Summarizing' not in obj.get('agent', ''):
                return obj.get('output', '')
        except Exception:
            continue
    raise ValueError("Could not find a valid Researcher output in logs.")


def call_openai_generate(client: OpenAI, prompt_text: str) -> str:
    attempts = 0
    while True:
        try:
            resp = client.chat.completions.create(
                model='o4-mini',
                messages=[{'role': 'user', 'content': prompt_text}],
            )
            return resp.choices[0].message.content
        except Exception as e:
            attempts += 1
            wait_s = min(60, 5 * attempts)
            print(f"Error in OpenAI call: {e}. Retrying in {wait_s}s...")
            time.sleep(wait_s)
            if attempts >= 3:
                raise


def main():
    parser = argparse.ArgumentParser(description='Prepare expert decisions for ensemble abstracts')
    parser.add_argument('--ensemble-dir', default='ensemble')
    parser.add_argument('--examples-root', default='assets/pytheus_examples')
    parser.add_argument('--pytheus-infos', default='assets/PYTHEUS_EXPLICIT_INFOS.txt')
    parser.add_argument('--feedback-file', default=None, help='Optional feedback file to enrich the prompt')
    parser.add_argument('--id', default=None, help='Single abstracts_*.txt to process; if omitted, process all')
    parser.add_argument('--output-dir', default=None, help='Where to write prepped_*.txt files; defaults to ensemble-dir')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--openai-api-key-file', default=None)
    args = parser.parse_args()

    client = _init_openai_client(args.openai_api_key_file)
    ensemble_dir = args.ensemble_dir
    output_dir = args.output_dir or ensemble_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load infos and examples
    with open(args.pytheus_infos, 'r') as f:
        pytheus_explicit_infos = f.read()
    examples_block = build_examples_block(args.examples_root, seed=args.seed)
    domain_feedback_block = parse_feedback_file(args.feedback_file)

    # Determine targets
    targets = [args.id] if args.id else [f.name for f in Path(ensemble_dir).glob('abstracts_*.txt')]
    if not targets:
        raise ValueError(f'No abstract files found in ensemble directory {ensemble_dir}. Please check the directory.')

    for id in targets:
        abstract_file_path = os.path.join(ensemble_dir, id)
        if not os.path.exists(abstract_file_path):
            raise ValueError(f'Abstract file {abstract_file_path} does not exist. Please check the ensemble directory.')

        print(f'Selected ensemble abstract ID: {id}')
        suggestion_timestamp = id.replace('abstracts_', '').replace('.txt', '')
        last_researcher_output = extract_last_researcher_output(ensemble_dir, suggestion_timestamp)

        prompt_text = build_prompt(
            pytheus_explicit_infos=pytheus_explicit_infos,
            examples_block=examples_block,
            domain_feedback_block=domain_feedback_block,
            researcher_suggestion=last_researcher_output,
        )

        output = call_openai_generate(client, prompt_text)
        print(output)

        decision_file_path = os.path.join(output_dir, f'prepped_{suggestion_timestamp}.txt')
        with open(decision_file_path, 'w') as f:
            f.write(output)


if __name__ == '__main__':
    main()

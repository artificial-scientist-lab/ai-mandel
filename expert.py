import re
import json
import random
import datetime
import traceback
from pathlib import Path
import multiprocessing
from typing import Optional

import os

from openai import OpenAI
from pytheus.main import run_main
import time

# some global variables
state = None
log_file_name = None
short_log_file_name = None
summary_file = None
timestamp = None
run_dir = None

def _init_openai_client(api_key_file: Optional[str] = None) -> OpenAI:
    """Initialize OpenAI client. Prefer OPENAI_API_KEY env; optionally read from file.

    This avoids hardcoding secrets in code and keeps publication-ready hygiene.
    """
    if "OPENAI_API_KEY" not in os.environ:
        file_to_use = api_key_file if api_key_file else ("API_key.txt" if Path("API_key.txt").exists() else None)
        if file_to_use:
            try:
                with open(file_to_use, "r") as f:
                    os.environ["OPENAI_API_KEY"] = f.read().strip()
            except Exception:
                # Defer error to client construction for clearer messaging
                pass
    return OpenAI()

# Lazily created in __main__ after CLI parsing; fall back here if imported as module
client: Optional[OpenAI] = None

MODEL = "o4-mini"
ASSETS_DIR = 'assets'


def from_tool_example_directory(path_examples, explanation=False):
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


# Removed: serialize_random_examples (unused)


def write_log(agent_name, agent_input, agent_output):
    def log_entry_to_str(agent_name, agent_input, agent_output):
        log_entry = {
            'agent': agent_name,
            'input': agent_input,
            'output': agent_output
        }
        # Write to the JSON log file
        log_entry_str = json.dumps(log_entry, indent=4)
        with open(log_file_name, 'a') as log_file:
            log_file.write(log_entry_str + '\n')        

        # Write to the short log file
        with open(short_log_file_name, 'a') as short_log_file:
            short_log_file.write(f"{agent_name}: {agent_output}\n\n")

    try:
        log_entry_to_str(agent_name, agent_input, agent_output)
    except:
        print(f"Error while writing log entry. Transforming to UTF-8 and trying again.")
        try:
            # Transform to UTF-8
            agent_input = agent_input.encode('utf-8', errors='replace').decode('utf-8')
            agent_output = agent_output.encode('utf-8', errors='replace').decode('utf-8')
            log_entry_to_str(agent_name, agent_input, agent_output)
        except:
            print(f"Error persists. Transforming to pure ASCII and trying again.")
            try:
                # Transform to ASCII, replacing non-ASCII characters with '?'
                agent_input = agent_input.encode('ascii', errors='replace').decode('ascii')
                agent_output = agent_output.encode('ascii', errors='replace').decode('ascii')
                log_entry_to_str(agent_name, agent_input, agent_output)
            except Exception as e:
                print(f"Critical error while logging: {e}")
                agent_input = "Cannot print input due to encoding error"
                agent_output = "Cannot print output due to encoding error"
                log_entry_to_str(agent_name, agent_input, agent_output)
                # Optionally, write the error to a separate error log file
                with open('error_log.txt', 'a') as error_log:
                    error_log.write(f"Failed to log entry: {e}\n")
    # cost accounting removed for publication hygiene

# Removed: random_arxiv_papers (unused after agent simplification)



# Removed: arxiv_tool (unused after agent simplification)

def call_openai_api(conversation_history, agent_name):
    """Unified OpenAI chat call using o4-mini for all agents.

    Retries briefly on transient errors.
    """
    global client
    if client is None:
        client = _init_openai_client(None)
    # normalize input
    if isinstance(conversation_history, list):
        normalized = [str(message) for message in conversation_history]
    else:
        normalized = [str(conversation_history)]
    normalized = [m.replace('target_state', 'target_quantum') for m in normalized]
    normalized = [m.replace('target state', 'target quantum') for m in normalized]
    full_prompt = '\n\n'.join(normalized)

    attempts = 0
    while True:
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{'role': 'user', 'content': full_prompt}],
            )
            reply = response.choices[0].message.content
            write_log(agent_name + f' ({MODEL})', full_prompt, reply)
            break
        except Exception as e:
            attempts += 1
            wait_s = min(60, 5 * attempts)
            print(f"Error in OpenAI call ({agent_name}): {e}. Retrying in {wait_s}s...")
            time.sleep(wait_s)
            if attempts >= 3:
                raise

    # revert tokens for downstream
    _ = [m.replace('target_quantum', 'target_state') for m in normalized]
    _ = [m.replace('target quantum', 'target state') for m in normalized]
    return reply


def parse_agent_reply(reply):
    action_match = re.search(r'Action:\s*(.+)', reply)
    if action_match:
        action = action_match.group(1).strip()
        input_match = re.search(r'Action Input:\s*(.*)', reply, re.DOTALL)
        if input_match:
            action_input = input_match.group(1).strip()
            action_input = action_input.strip('\'"')
        else:
            action_input = ''
        return action, action_input

    #ideally this should now be redundant
    else:
        possible_actions = ['Final Answer', 'Feedback Researcher', 'Feedback Expert', '[Pass to Expert]', '[Pass to Tool]', '[Accept]', 'Reject']
        for possible_action in possible_actions:
            if possible_action in reply:
                action_match = re.search(rf'{possible_action}[:\s]*\n*(.*)', reply, re.DOTALL)
                if action_match:
                    action_input = action_match.group(1).strip()
                    return possible_action, action_input
                else:
                    return possible_action, ''
        return 'Continue', reply



def create_expert_error_feedback(err_name, err_msg, err_trace):
    # Create a meaningful error message string for GPT
    feedback_error_msg = (
        f"Error encountered while executing `pytheus`: \n"
        f"Error Type: {err_name}\n"
        f"Error Message: {err_msg}\n"
        # f"Traceback:\n{err_trace}\n\n"
        "This error indicates that there was an issue with the pytheus function. "
        "Please review the error type and message above to understand the problem. "
        "If possible, modify the input or parameters and try again."
    )
    
    # Return or print the message (return it here to pass it back to GPT)
    return feedback_error_msg


def pytheus_tool(input_data, run_dir):
    data = json.loads(input_data)

    # avoid many runs, hardcode the number of samples to 1
    if 'samples' in data:
        data['samples'] = 1

    # config_filename = f'config_{timestamp}.json'
    config_filename = f'config_{random.randint(1000000000, 9999999999)}.json'
    print('##### run_dir')
    print(run_dir)
    with open(run_dir + '/' + config_filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    run_main(run_dir + '/' + config_filename, False)
    return f"Pytheus executed successfully!"


def pytheus_target(queue, action_input, run_dir):
    try:
        result = pytheus_tool(action_input, run_dir)
        queue.put(('result', result))
    except Exception as e:
        e_type = type(e).__name__
        e_message = str(e)
        e_traceback = traceback.format_exc()
        queue.put(('exception', e_type, e_message, e_traceback))

def pytheus_tool_with_timeout(action_input, run_dir, timeout=10):
    action_input = action_input.replace('True', 'true').replace('False', 'false')
    action_input = action_input.replace('None', 'null')
    action_input = action_input.replace('target_quantum', 'target_state')
    def trim_to_braces(text: str) -> str:
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1 or start > end:
            return ""
        return text[start:end+1]
    action_input = trim_to_braces(action_input)
    for ii in range(10):
        try:
            data = json.loads(action_input)
            break
        except json.JSONDecodeError as e:
            prompt = f"There is a JSONDecodeError in the input data. Output a correct JSON that can be loaded with json.loads(input_data). Don't output anything other than the fixed input data. Input data: {action_input}. Fixed input data: "
            action_input = call_openai_api([prompt], 'PytheusTool')
            action_input = action_input.replace('True', 'true').replace('False', 'false')
            action_input = action_input.replace('None', 'null')
            action_input = action_input.replace('target_quantum', 'target_state')
            action_input = trim_to_braces(action_input)

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=pytheus_target, args=(queue, action_input, run_dir))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return 'timeout'
        # raise TimeoutError("Pytheus computation took too long. Consider using less ancillary particles")

    if not queue.empty():
        message = queue.get()
        if message[0] == 'result':
            return message[1]
        elif message[0] == 'exception':
            e_type, e_message, e_traceback = message[1], message[2], message[3]
            # Raise a new exception with this information
            raise Exception(f"{e_type}: {e_message}")
    else:
        raise Exception("Unknown error occurred in pytheus_tool.")

def init_state(suggestion_type, journal_dir=None, id=None):
        # suggestion_type
        # [1,x,x,x]=State
        # [x,1,x,x]=Gate -- 18 gates in total
        # [x,x,1,x]=Measurement -- 5 measurements in total
        # [x,x,x,1]=Protocol -- 4 protocols in total
    with open(f'{ASSETS_DIR}/PROMPT_EXPERT.txt', 'r') as f:
        PROMPT_EXPERT = f.read()


    #fixed split of examples
    all_examples_state = from_tool_example_directory(f'{ASSETS_DIR}/pytheus_examples/States',explanation=True)
    examples_states = serialize_examples(all_examples_state, n_examples=min(suggestion_type[0],len(all_examples_state)), seed=None)
    
    
    all_examples_gates = from_tool_example_directory(f'{ASSETS_DIR}/pytheus_examples/Gates',explanation=True)
    examples_gates = serialize_examples(all_examples_gates, n_examples=min(suggestion_type[1],len(all_examples_gates)), seed=None)
    all_examples_measurement = from_tool_example_directory(f'{ASSETS_DIR}/pytheus_examples/Measurements',explanation=True)
    examples_measurement = serialize_examples(all_examples_measurement, n_examples=min(suggestion_type[2],len(all_examples_measurement)), seed=None)
    all_examples_communication = from_tool_example_directory(f'{ASSETS_DIR}/pytheus_examples/Communication',explanation=True)
    examples_communication = serialize_examples(all_examples_communication, n_examples=min(suggestion_type[3],len(all_examples_communication)), seed=None)


    with open(f'{ASSETS_DIR}/PYTHEUS_EXPLICIT_INFOS.txt', 'r') as f:
        pytheus_explicit_infos = f.read()

    #load 100states.txt content as string
    with open(f'{ASSETS_DIR}/100states.txt', 'r', encoding='latin-1') as f:
        states = f.read()
    
    if all_journals:
        print("USING ALL PREVIOUS RESULTS FROM ALL JOURNALS")

        all_previous_results = ''
        for journal_dir in Path('.').glob('journal*/'):
            print(f'Processing {journal_dir.name}...')
            previous_results = ''
            for txt_file in Path(journal_dir).rglob('journal*.txt'):
                with open(txt_file, 'r') as f:
                    previous_results += f.read()

            #only keep newlines, and lines which contain '"description":' and '"target_state":'
            previous_results = [line for line in previous_results.split('\n') if '"description":' in line or '"target_state":' in line]
            for i, entry in enumerate(previous_results):
                # entry is enclosed by curly braces, so we can safely use it as a JSON object
                entry = entry.strip()
                # if '{' in entry and '}' in entry:
                #     print('found JSON-like entry:', entry)
                if entry.startswith('{') and entry.endswith('}'):
                    # print('found JSON-like entry:', entry)
                    entry = json.loads(entry)
                    entry = '"description":' + entry.get("description", '')
                    # entry += "THIS WAS JSON"



                if '"description":' in entry:
                    entry = entry.split('"description":')[1].strip()
                    previous_results[i] = 'previous result: ' + entry
                # if '"target_state":' in entry:
                #     previous_results[i] = entry + '\n'
            previous_results = '\n'.join(previous_results)
            all_previous_results += previous_results
        previous_results = all_previous_results
    elif journal_dir is not None:
        previous_results = ""

        for txt_file in Path(journal_dir).glob('journal*.txt'):
            with open(txt_file, 'r') as f:
                previous_results += f.read()

        #only keep newlines, and lines which contain '"description":' and '"target_state":'
        previous_results = [line for line in previous_results.split('\n') if '"description":' in line or '"target_state":' in line]
        for i, entry in enumerate(previous_results):
            if '"description":' in entry:
                previous_results[i] = 'entry:\n' + entry
            if '"target_state":' in entry:
                previous_results[i] = entry + '\n'
        previous_results = '\n'.join(previous_results)
    else:
        print("No journal directory provided, using empty previous results.")
        previous_results = ""

    # print(previous_results)

    examples = examples_states+examples_gates + examples_measurement + examples_communication


    def replace_tags(prompt: str) -> str:
        # Always keep chat content; strip chat tags; drop nonchat blocks
        prompt = re.sub(r'<nonchatmodel>.*?</nonchatmodel>', '', prompt, flags=re.DOTALL)
        prompt = re.sub(r'</?chatmodel>', '', prompt)
        return prompt
    
    PROMPT_EXPERT = PROMPT_EXPERT.replace("{examples}", examples)
    PROMPT_EXPERT = PROMPT_EXPERT.replace("{100statestxt}", states)
    PROMPT_EXPERT = PROMPT_EXPERT.replace("{previous_results}", previous_results)
    PROMPT_EXPERT = replace_tags(PROMPT_EXPERT)


    if id is None or id == "":
        # id: random filename of all files 'abstracts_*.txt' in ensemble_dir
        candidates = [f.name for f in Path(ensemble_dir).glob('abstracts_*.txt')]
        if not candidates:
            raise ValueError(f'No abstract files found in ensemble directory {ensemble_dir}. Please check the directory.')
        id = random.choice(candidates)

    suggestion_timestamp = id.replace('abstracts_', '').replace('.txt', '')

    abstract_filename = 'abstracts_'+suggestion_timestamp+'.txt'

    #load abstract from ensemble directory
    abstract_file_path = os.path.join(ensemble_dir, abstract_filename)
    if not os.path.exists(abstract_file_path):
        raise ValueError(f'Abstract file {abstract_file_path} does not exist. Please check the ensemble directory.')
    global abstract_content
    with open(abstract_file_path, 'r') as f:
        abstract_content = f.read()

    #save a copy of the abstract content with id as filename in run_dir
    global run_dir
    if run_dir is not None:
        abstract_copy_path = os.path.join(run_dir, abstract_filename)
        with open(abstract_copy_path, 'w') as f:
            f.write(abstract_content)

    print(f'Selected ensemble abstract ID: {id}')
    #remove 'abstracts_' prefix and '.txt' suffix
    #find directory containing this timestamp in 'ensemble_runs'
    ensemble_runs_dir = f'{ensemble_dir}/ensemble_runs'
    ensemble_run_dir = None
    for dd in os.listdir(ensemble_runs_dir):
        #check if run_dir contains the timestamp
        if suggestion_timestamp in dd:
            ensemble_run_dir = os.path.join(ensemble_runs_dir, dd)
            print(f'Found ensemble run directory: {ensemble_run_dir}')
            break

    if ensemble_run_dir is None:
        raise ValueError(f'No ensemble run directory found for timestamp {suggestion_timestamp}')

    
    decision_file = 'prepped_'+suggestion_timestamp+'.txt'
    decision_file_path = os.path.join(ensemble_dir, decision_file)
    if not os.path.exists(decision_file_path):
        raise ValueError(f'Decision file {decision_file_path} does not exist. Please check the ensemble run directory.')
    global decision_content
    with open(decision_file_path, 'r') as f:
        decision_content = f.read()

    #find log*.txt file in ensemble_run_dir
    log_files = list(Path(ensemble_run_dir).glob('log*.txt'))
    if not log_files:
        raise ValueError(f'No log files found in ensemble run directory {ensemble_run_dir}')
    log_file = log_files[0]  # Take the first log file found
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    combined_content = []
    for line in lines:
        if line.strip():  # Check if the line is not empty
            combined_content.append(line.strip())   
    combined_content = '\n'.join(combined_content)
    combined_content = combined_content.split('}\n{')

    print(len(combined_content))

    global last_researcher_output
    for ii in range(len(combined_content)):
        try:
            last_researcher_output = '{' + combined_content[-1-ii] + '}'
            last_researcher_output = json.loads(last_researcher_output)
        except json.JSONDecodeError:
            continue
        if 'Researcher' in last_researcher_output['agent'] and not 'Summarizing' in last_researcher_output['agent']:
            last_researcher_output = last_researcher_output['output']
            break
    print(last_researcher_output)

    global prompt_type

    PROMPT_EXPERT = PROMPT_EXPERT.replace('{abstract_content}', abstract_content)
    PROMPT_EXPERT = PROMPT_EXPERT.replace('{last_researcher_output}', last_researcher_output)
    PROMPT_EXPERT = PROMPT_EXPERT.replace('{pytheus_explicit_infos}', pytheus_explicit_infos)



    if prompt_type == 'prep' or prompt_type == 'prep_modprompt':
        prep_add = f"""
        Additional input:
        START SUGGESTED MODIFICATIONS
        {decision_content}
        END SUGGESTED MODIFICATIONS
        """
        PROMPT_EXPERT = PROMPT_EXPERT.replace('<prepped>',prep_add)
    elif prompt_type == 'modprompt':
        PROMPT_EXPERT = PROMPT_EXPERT.replace('<prepped>','')



    if prompt_type == 'prep_modprompt' or prompt_type == 'modprompt':        
        PROMPT_EXPERT = PROMPT_EXPERT.replace('<modprompt>','YOUR OBJECTIVE IS TO FIX BUGS IN THE IMPLEMENTATION, UNDER ABSOLUTELY NO CIRCUMSTANCES MODIFY THE TARGET TO MAKE THE IMPLEMENTATION EASIER!!! FOCUS ON FIXING THE BUGS, BUT MAKE SURE THAT THE CONFIGURATION YOU PROVIDE TO THE TOOL STAYS 100 PERCENT TRUE TO THE SUGGESTION MADE BY THE RESEARCHER!!!')
    elif prompt_type == 'prep':
        PROMPT_EXPERT = PROMPT_EXPERT.replace('<modprompt>','')

    state = {
        'expert': [PROMPT_EXPERT]
    }

    #save prompts as txt files to run_dir
    if run_dir is not None:
        with open(f'{run_dir}/PROMPT_EXPERT.txt', 'w') as f:
            f.write(PROMPT_EXPERT)
    return state




def print_reply(reply, agent_name):
    reply_mod = re.sub(r'(Action:|Action Input:)', r'\033[3m\1\033[0m', reply)
    phrases_to_style = ['Action:', 'Action Input:', 'Thought:', 'Final Answer', 'Feedback Researcher', 'Feedback Expert', '[Pass to Expert]', 'Criticism', 'Reject','[Accept]']
    pattern = r'(' + '|'.join(map(re.escape, phrases_to_style)) + r')'
    reply_mod = re.sub(pattern, r'\033[3m\1\033[0m', reply_mod)
    print(f"\033[1m{agent_name}\033[0m: {reply_mod}\n")

    


def expert_agent(process_id=0, internal_expert_calls=0, external_expert_calls=0):
    agent_name = 'Expert'
    global pytheus_success
    global run_dir
    try:
        reply = call_openai_api(state['expert'], agent_name)
    except Exception as e:
        #save state to file
        with open(os.path.join(run_dir, "state_error.json"), "w") as f:
            json.dump(state['expert'], f, indent=4)
        # still throw the exception
        assert 0 == 1
    print_reply(reply, agent_name)
    action, action_input = parse_agent_reply(reply)
    
    if action == 'pytheus':
        timeout = False
        try:
            loc_timestamp = datetime.datetime.now().strftime('%d_%H%M_%S')
            with open(run_dir + f"/action_input_{loc_timestamp}.json", 'w') as f:
                f.write(action_input)
            pytheus_result = pytheus_tool_with_timeout(action_input, run_dir, timeout=300)
            # for now replace pytheus_result with action_input
            if pytheus_result == 'timeout':
                timeout = True
            pytheus_result = action_input
            pytheus_success = True
        except Exception as e:
            err_name = type(e).__name__
            err_msg = str(e)
            err_trace = traceback.format_exc()
            feedback_error_msg = create_expert_error_feedback(err_name, err_msg, err_trace)
            state['expert'].append(f"Expert: {reply}")
            state['expert'].append(f"Pytheus error: {feedback_error_msg}")
            write_log('PytheusError', 'error', feedback_error_msg)

            if 'TimeoutError' in feedback_error_msg:
                with open(f"{journal_dir}/timeouts_{process_id}.txt", 'a') as f:
                    f.write(f"Action Input: {action_input}\nRun Directory: {run_dir}\n\n")

            debugging_message = "To figure out what went wrong you can try writing an explanation similar to what you were given for the examples in the very beginning. This might help you understand the problem better and see if there are any inconsistencies."

            if "JSONDecodeError" in feedback_error_msg:
                debugging_message += " Remember that the action input for running pytheus is a JSON object and nothing else, so make sure that the input is correctly formatted. Make sure you are using the correct double quotes, sometimes you write a backslash before the double quote to escape it. This is not necessary in the action input for running pytheus. Don't give 'none' as a value for a key in the JSON object, just don't include the key-value pair if it's not needed. Make sure that true and false are lowercase. If you are still stuck, you can think about what other things are necessary to make a valid JSON file and if they are fulfilled."
            write_log('DebuggingMessage', 'debugging', debugging_message)
            state['expert'].append(debugging_message)
            return expert_agent

        write_log('PytheusTool', action_input, pytheus_result)
        state['expert'].append(f"PytheusTool: {pytheus_result}")

        try:
            # Save final result as a file
            loc_timestamp = datetime.datetime.now().strftime('%d_%H%M_%S')
            with open(run_dir + f"/final_result_{timeout}_{loc_timestamp}.txt", 'w') as f:
                f.write(pytheus_result)
                f.write('###')
                # write original abstract to final result
                global abstract_content
                f.write(abstract_content)

            # Add final result to journal.txt
            with open(f"{journal_dir}/journal_{process_id}.txt", 'a') as f:
                f.write(f"-- new entry --\n")
                f.write(f"Proposed Experiment:\n")
                f.write(pytheus_result)
                f.write("\n\n\n")
        except:
            print("Error while writing final result. Transforming to UTF-8 and trying again.")
            try:
                # Transform to UTF-8
                pytheus_result = pytheus_result.encode('utf-8', errors='replace').decode('utf-8')
                with open(run_dir + f"/final_result_{loc_timestamp}.txt", 'w') as f:
                    f.write(pytheus_result)

                with open(f"{journal_dir}/journal_{process_id}.txt", 'a') as f:
                    f.write(f"-- new entry --\n")
                    f.write(f"Proposed Experiment:\n")
                    f.write(pytheus_result)
                    f.write("\n\n\n")
            except:
                print("Error persists. Transforming to pure ASCII and trying again.")
                try:
                    # Transform to ASCII, replacing non-ASCII characters with '?'
                    pytheus_result = pytheus_result.encode('ascii', errors='replace').decode('ascii')
                    with open(run_dir + f"/final_result_{loc_timestamp}.txt", 'w') as f:
                        f.write(pytheus_result)

                    with open(f"{journal_dir}/journal_{process_id}.txt", 'a') as f:
                        f.write(f"-- new entry --\n")
                        f.write(f"Proposed Experiment:\n")
                        f.write(pytheus_result)
                        f.write("\n\n\n")
                except Exception as e:
                    print(f"Critical error while writing final result: {e}")
                    pytheus_result = "Cannot write result due to encoding error"
                    with open(run_dir + f"/final_result_{loc_timestamp}.txt", 'w') as f:
                        f.write(pytheus_result)

                    with open(f"{journal_dir}/journal_{process_id}.txt", 'a') as f:
                        f.write(f"-- new entry --\n")
                        f.write(f"Proposed Experiment:\n")
                        f.write(pytheus_result)
                        f.write("\n\n\n")

                    # Optionally, write the error to a separate error log file
                    with open('error_log.txt', 'a') as error_log:
                        error_log.write(f"Failed to save final result: {e}\n")

        state['expert'].append("Your implementation was successful. You may consider iterating on the results to explore more complex variations.")

        write_log('PytheusTool', action_input, 'success')
    elif len(reply) < 65:
        state['expert'].append(f"Expert: {reply}")
        state['expert'].append("System Message: The expert's response was short; refine inputs for more detailed feedback if needed.")
    else:
        print(f'{agent_name}: NO KNOWN ACTION!')
        state['expert'].append(f"System Message: Your response did not follow the expected output format. Please try again.")
        return expert_agent
    
# Removed: configcritic_agent (ConfigCritic disabled)
    

if __name__ == '__main__':
    import argparse

    # Get the global task rank
    process_id = int(os.environ.get('SLURM_PROCID', 0))

    # Alternatively, if you need a node-local identifier:
    local_id = int(os.environ.get('SLURM_LOCALID', 0))

    #sbatch array id
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

    process_id = str(process_id)+'_'+str(local_id)+'_'+str(task_id)
    print(f"Process index: {process_id}")
    # CLI configuration
    parser = argparse.ArgumentParser(description='Expert Script')
    parser.add_argument('--ensemble-dir', default='ensemble', help='Path to ensemble directory with abstracts_*.txt')
    parser.add_argument('--journal-dir', default='journal_o4-mini', help='Directory to write journal entries/results')
    parser.add_argument('--all-journals', action='store_true', default=True, help='Aggregate previous results across all journal*/ dirs')
    parser.add_argument('--id', default=None, help='Abstract file name to use (e.g., abstracts_YYYY_MM_DD_HHMM_SS.txt). Defaults to random.')
    parser.add_argument('--max-internal-expert-calls', type=int, default=20)
    parser.add_argument('--max-external-expert-calls', type=int, default=20)
    parser.add_argument('--max-researcher-calls', type=int, default=20)
    parser.add_argument('--openai-api-key-file', default=None, help='Optional path to a file containing an OpenAI API key')
    args = parser.parse_args()

    # Initialize OpenAI client now that args are known
    client = _init_openai_client(args.openai_api_key_file)

    ensemble_dir = args.ensemble_dir
    journal_dir = args.journal_dir
    all_journals = args.all_journals
    os.makedirs(journal_dir, exist_ok=True)

    MAX_EXTERNAL_EXPERT_CALLS = args.max_exterior_expert_calls if hasattr(args, 'max_exterior_expert_calls') else args.max_external_expert_calls
    MAX_INTERNAL_EXPERT_CALLS = args.max_internal_expert_calls
    MAX_RESEARCHER_CALLS = args.max_researcher_calls
    id = args.id


    for _ in range(10000):
        global prompt_type
        pytheus_success = False
        prompt_type = random.choice(['prep', 'prep_modprompt', 'modprompt'])


        last_researcher_output = None
        abstract_content = None
        timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H%M_%S')
        print(f"Starting new run at {timestamp}")
        uniqueid = random.randint(0, 999)
        run_dir = f"{ensemble_dir}/expert_runs/{MODEL}_{timestamp}_{uniqueid}"
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "info.txt"), "w") as f:
            f.write(f"model {MODEL}")
            f.write(f"\nprompt_type {prompt_type}")
        log_file_name = os.path.join(run_dir, f"log_{timestamp}.txt")
        short_log_file_name = os.path.join(run_dir, f"short_log_{timestamp}.txt")
        
        
        # [1,x,x,x]=State
        # [x,1,x,x]=Gate -- 18 gates in total
        # [x,x,1,x]=Measurement -- 5 measurements in total
        # [x,x,x,1]=Protocol -- 4 protocols in total

        # suggestion_type = random.choice([
        #     [0, 0, 0, 5],  
        #     [0, 0, 3, 3],  
        #     [0, 0, 1, 4]
        # ])
    
        suggestion_type = [100, 100, 100, 100]

        state = init_state(suggestion_type, journal_dir=journal_dir, id=id)

        # save state to file
        with open(os.path.join(run_dir, "init_state.json"), "w") as f:
            json.dump(state, f, indent=4)
        write_log('SuggestionType', 'suggestion_type', str(suggestion_type))

        current_agent_function = expert_agent
        external_expert_calls = 0
        internal_expert_calls = 0
        researcher_calls = 0
        while True:
            if pytheus_success:
                break
            elif current_agent_function == expert_agent:
                next_agent_function = current_agent_function(process_id=process_id, internal_expert_calls=internal_expert_calls)
            else:
                next_agent_function = current_agent_function()
            external_expert_call = next_agent_function is expert_agent and current_agent_function is not expert_agent
            internal_expert_call = next_agent_function is expert_agent and current_agent_function is expert_agent
            if external_expert_call:
                external_expert_calls += 1
            if internal_expert_call:
                internal_expert_calls += 1

            if next_agent_function is None or external_expert_calls >= MAX_EXTERNAL_EXPERT_CALLS or internal_expert_calls >= MAX_INTERNAL_EXPERT_CALLS or researcher_calls >= MAX_RESEARCHER_CALLS:
                print("Conversation ended.")
                # save 'conversationend.txt' in run_dir
                with open(os.path.join(run_dir, "conversationend.txt"), "w") as f:
                    f.write(f"Conversation ended at {timestamp}")
                print(f'external_expert_calls={external_expert_calls}, internal_expert_calls={internal_expert_calls}')
                print(f'researcher_calls={researcher_calls}')
                print(f'next_agent_function={next_agent_function}')
                print(f'timestamp={timestamp}')
                print("=" * 20)
                write_log('ConversationEnd', 'stats', f'external_expert_calls={external_expert_calls}, internal_expert_calls={internal_expert_calls}, researcher_calls={researcher_calls}')
                break
            current_agent_function = next_agent_function
    
    
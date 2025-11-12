import re
import json
import random
import datetime
from pathlib import Path
import numpy as np
import os
import argparse
from typing import Optional, List, Union

from openai import OpenAI
import time

# some global variables
state: Optional[dict] = None
log_file_name: Optional[str] = None
short_log_file_name: Optional[str] = None
timestamp: Optional[str] = None
run_dir: Optional[str] = None

# Default path configuration (can be overridden by CLI flags)
ASSETS_DIR = 'assets'
PATH_PROMPT_RESEARCHER = f'{ASSETS_DIR}/PROMPT_RESEARCHER.txt'
PATH_PROMPT_NOVELTY = f'{ASSETS_DIR}/PROMPT_NOVELTY.txt'
PATH_PROMPT_JUDGE = f'{ASSETS_DIR}/PROMPT_JUDGE.txt'
PATH_RECENT_PAPERS = f'{ASSETS_DIR}/recent_quantum_papers.json'
PATH_PYTHEUS_EXPLICIT_INFOS = f'{ASSETS_DIR}/PYTHEUS_EXPLICIT_INFOS.txt'
PATH_STATES_FILE = f'{ASSETS_DIR}/100states_medium.txt'
PATH_CONCEPT_PAIRS_FILE = f'{ASSETS_DIR}/filtered_future_suggested_pairs_IR10.txt'
PATH_PYTHEUS_EXAMPLES_DIR = f'{ASSETS_DIR}/pytheus_examples'
PATH_PYTHEUS_PAPER = f'{ASSETS_DIR}/pytheus_fullpaper.txt'

try:
    with open('API_key.txt', 'r') as f:
        api_key = f.read().strip()
        os.environ["OPENAI_API_KEY"] = api_key
except FileNotFoundError:
    # Keep legacy behavior (will error later at first API call), but provide a clearer hint early.
    print("Warning: API_key.txt not found. Ensure OPENAI_API_KEY is set in the environment or provide API_key.txt.")

    
MODEL = 'o4-mini'
client = OpenAI()


def from_tool_example_directory(path_examples: Union[str, Path], explanation: bool = False) -> dict:
    """Load Pytheus tool examples from a directory.

    When explanation=True, returns a mapping {subdir: {explanation, config?}}.
    Otherwise returns a mapping {relative_config_path: config_dict}.
    """
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


def serialize_examples(examples: dict, n_examples: int, seed: Optional[int] = None) -> str:
    """Serialize a random subset of examples from the Pytheus tool.

    Returns a single string containing n_examples randomly chosen entries.
    """
    if seed is not None:
        random.seed(seed)
    keys = random.sample(list(examples.keys()), n_examples)
    return "".join(["### EXAMPLE {ind} ###\n{path}\n{config}\n\n".format(path=key, config=examples[key], ind=ii+1) for ii, key in enumerate(keys)])

def write_log(agent_name: str, agent_input: str, agent_output: str) -> None:
    """Append a detailed entry to the JSON log and a concise entry to the short log.

    Also records the last and total cost in cost.csv within the run directory.
    """
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
    # Removed cost accounting per request

def random_arxiv_papers(num_papers: int = 3) -> str:
    """Return a formatted list of random filtered arXiv paper titles.

    Uses a pre-downloaded JSON file `recent_quantum_papers.json` and filters
    with inclusive `go_list` and exclusive `nogo_list` keywords.
    """
    # Define nogo list
    nogo_list = ["tensor", "flux", "Lindblad", "Density functional theory", "DFT", "electron", "trapped", "quantum comput", "atom", "molecule", "Bose-Einstein", "BEC", "electrodynamics", "adiabatic", "torques", "superconductivity", "superconducting", "criticality", "anyons", "Josephson", "pendulums", "nitrogen vacancy", "diamond", "optomechanical", "PT-symmetric", "quantum annealing", "quantum complexity", "quantum vacuum", "quantum field theory", "nanomechanical", "IBM","ground state","learn","spatial modes","oxide","transition","reciprocity"]

    go_list = ["swapping", "teleportation", "network"]
    # Load JSON data from the file
    with open(PATH_RECENT_PAPERS, 'r', encoding='utf-8') as file:
        papers = json.load(file)

    # Filter papers that do not contain nogo words
    def is_valid(paper, nogo_list, go_list = []):
        text = paper['title'].lower() + " " + paper['abstract'].lower()
        
        if go_list:
            #check that at least one of the go_list words is in the text
            if not any(word.lower() in text for word in go_list):
                return False

        for word in nogo_list:
            if word.lower() in text:
                return False
        
        return True

    print(f"Found {len(papers)} papers before filtering.")
    valid_papers = [paper for paper in papers if is_valid(paper, nogo_list, go_list=go_list)]
    print(f"Found {len(valid_papers)} valid papers after filtering.")

    # Select up to 5 random valid papers
    random_papers = random.sample(valid_papers, min(num_papers, len(valid_papers)))

    out_str = ""
    for idx, paper in enumerate(random_papers, start=1):
        out_str += f"Paper {idx}: {paper['title']}\n"
        # out_str += f"{paper['abstract']}\n"
        out_str += "-" * 80 + "\n"
    return out_str




def arxiv_tool(query: str, max_results: int = 3) -> str:
    """Query arXiv and return a formatted summary of results.

    Imports arxiv lazily to avoid hard dependency when not used.
    """
    try:
        import arxiv  # type: ignore
    except ImportError:
        return "[arxiv package not installed]"
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )
    results = []
    for result in search.results():
        paper = {
            'title': result.title,
            'authors': [author.name for author in result.authors],
            'summary': result.summary,
            'url': result.entry_id,
            'published': result.published.strftime('%Y-%m-%d'),
        }
        results.append(paper)
    # Format the results into a string
    output = ""
    for paper in results:
        output += f"Title: {paper['title']}\n"
        output += f"Authors: {', '.join(paper['authors'])}\n"
        output += f"Published: {paper['published']}\n"
        output += f"URL: {paper['url']}\n"
        output += f"Summary: {paper['summary']}\n\n"
    return output

# Cost calculations, non-chat model schemas, and pricing removed per request

from typing import List, Tuple

def call_openai_api(conversation_history: List[str], agent_name: str) -> str:
    """Call o4-mini with the provided conversation and return its reply.

    Preserves legacy string substitutions on 'target_state' tokens and writes to logs.
    """
    conversation_history = [message.replace('target_state', 'target_quantum') for message in conversation_history]
    conversation_history = [message.replace('target state', 'target quantum') for message in conversation_history]

    full_prompt = '\n\n'.join(conversation_history)
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{'role': 'user', 'content': full_prompt}]
        )
        reply = response.choices[0].message.content
        write_log(agent_name+f' ({MODEL})', full_prompt, reply)
    except Exception as e:
        print(f"Error occurred: {e}. Retrying in 1 minute...")
        time.sleep(60)
        return call_openai_api(conversation_history, agent_name)
    conversation_history = [message.replace('target_quantum', 'target_state') for message in conversation_history]
    conversation_history = [message.replace('target quantum', 'target state') for message in conversation_history]
    return reply

def parse_agent_reply(reply: str) -> Tuple[str, str]:
    """Parse an agent's textual reply for Action and Action Input fields."""
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
        return 'Continue', reply

def init_state(suggestion_type: List[int], journal_dir: Optional[str] = None) -> dict:
    """Initialize global prompts and state for the researcher and novelty supervisor.

    This function loads examples, previous results, concept pairs, ensemble abstracts,
    and constructs filled prompt strings. It writes prompt snapshots into run_dir.
    """
        # suggestion_type
        # [1,x,x,x]=State
        # [x,1,x,x]=Gate -- 18 gates in total
        # [x,x,1,x]=Measurement -- 5 measurements in total
        # [x,x,x,1]=Protocol -- 4 protocols in total
    
    global PROMPT_RESEARCHER
    global PROMPT_NOVELTY
    global PROMPT_JUDGE
    global REPEATER_RESEARCHER

    with open(PATH_PROMPT_RESEARCHER, 'r') as f:
        PROMPT_RESEARCHER = f.read()
    with open(PATH_PROMPT_NOVELTY, 'r') as f:
        PROMPT_NOVELTY = f.read()
    with open(PATH_PROMPT_JUDGE, 'r') as f:
        PROMPT_JUDGE = f.read()


    global run_dir


    #fixed split of examples
    all_examples_state = from_tool_example_directory(Path(PATH_PYTHEUS_EXAMPLES_DIR)/'States', explanation=True)
    examples_states = serialize_examples(all_examples_state, n_examples=int(np.round(suggestion_type[0]*len(all_examples_state))), seed=None)
    
    
    all_examples_gates = from_tool_example_directory(Path(PATH_PYTHEUS_EXAMPLES_DIR)/'Gates', explanation=True)
    examples_gates = serialize_examples(all_examples_gates, n_examples=min(suggestion_type[1],len(all_examples_gates)), seed=None)
    all_examples_measurement = from_tool_example_directory(Path(PATH_PYTHEUS_EXAMPLES_DIR)/'Measurements', explanation=True)
    examples_measurement = serialize_examples(all_examples_measurement, n_examples=min(suggestion_type[2],len(all_examples_measurement)), seed=None)
    all_examples_communication = from_tool_example_directory(Path(PATH_PYTHEUS_EXAMPLES_DIR)/'Communication', explanation=True)
    examples_communication = serialize_examples(all_examples_communication, n_examples=min(suggestion_type[3],len(all_examples_communication)), seed=None)


    with open(PATH_PYTHEUS_EXPLICIT_INFOS, 'r') as f:
        pytheus_explicit_infos = f.read()

    #load 100states.txt content as string
    with open(PATH_STATES_FILE, 'r', encoding='latin-1') as f:
        states = f.read()
    
    all_previous_results_file = 'assets/all_previous_results.txt'
    empty_journal = True
    if empty_journal:
        print("USING EMPTY PREVIOUS RESULTS")
        previous_results = ""
    elif all_previous_results_file is not None and os.path.exists(all_previous_results_file):
        with open(all_previous_results_file, 'r') as f:
            previous_results = f.read()
    elif all_journals:
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
                entry = entry.strip()
                if entry.startswith('{') and entry.endswith('}'):
                    entry = json.loads(entry)
                    entry = '"description":' + entry.get("description", '')
                if '"description":' in entry:
                    entry = entry.split('"description":')[1].strip()
                    previous_results[i] = 'previous result: ' + entry
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

    #save all previous results to a file in run_dir
    if run_dir is not None:
        with open(f'{run_dir}/all_previous_results.txt', 'w') as f:
            f.write(previous_results)
    # print(previous_results)

    examples = examples_states+examples_gates + examples_measurement + examples_communication

    random_arxiv_suggestions = random_arxiv_papers(num_papers=3)

    def replace_tags(prompt):
        # Using chat model: remove <nonchatmodel> blocks and strip <chatmodel> tags
        prompt = re.sub(r'<nonchatmodel>.*?</nonchatmodel>','', prompt, flags=re.DOTALL)
        prompt = re.sub(r'<chatmodel>','', prompt)
        prompt = re.sub(r'</chatmodel>','', prompt)
        return prompt

    #get string enclosed by <repeater>...</repeater> tags
    REPEATER_RESEARCHER = re.findall(r'<repeater>(.*?)</repeater>', PROMPT_RESEARCHER, flags=re.DOTALL)
    REPEATER_RESEARCHER = '\n'.join(REPEATER_RESEARCHER)
    PROMPT_RESEARCHER = PROMPT_RESEARCHER.replace('<repeater>', '').replace('</repeater>', '')


    #PROMPT_RESEARCHER = PROMPT_RESEARCHER.replace('{suggestions}',suggestions)
    PROMPT_RESEARCHER = PROMPT_RESEARCHER.replace('{100statestxt}', states)
    PROMPT_RESEARCHER = PROMPT_RESEARCHER.replace('{previous_results}', previous_results)
    PROMPT_RESEARCHER = PROMPT_RESEARCHER.replace('{examples}', examples)
    PROMPT_RESEARCHER = PROMPT_RESEARCHER.replace('{pytheus_explicit_infos}', pytheus_explicit_infos)
    PROMPT_RESEARCHER = PROMPT_RESEARCHER.replace('{random_arxiv_suggestions}', random_arxiv_suggestions)
    PROMPT_RESEARCHER = replace_tags(PROMPT_RESEARCHER)

    

    #load concept pair from impact4cast suggestions (quantum optics concepts, top 0.1%)
    file_path = PATH_CONCEPT_PAIRS_FILE
    IND = random.randint(0, 500)

    with open(file_path, 'r') as file:
        for ii, line in enumerate(file):
            if ii == IND:
                #format: {entry1}&{entry2};{score}
                entry, score = line.strip().split(';')
                entry1, entry2 = entry.split('&')

    PROMPT_RESEARCHER = PROMPT_RESEARCHER.replace('{entry1}', entry1)
    PROMPT_RESEARCHER = PROMPT_RESEARCHER.replace('{entry2}', entry2)

    #load all abstracts from ensemble directory
    ensemble_abstracts = ''
    abstract_list = []
    global ensemble_dir
    if os.path.exists(ensemble_dir) and os.path.isdir(ensemble_dir):
        for txt_file in Path(ensemble_dir).glob('abstracts_*.txt'):
            with open(txt_file, 'r') as f:
                abstract_list.append(f.read())

    #random sort the abstracts
    random.shuffle(abstract_list)
    for abstract in abstract_list:
        ensemble_abstracts += 'idea suggested by ensemble:\n'
        ensemble_abstracts += abstract + '\n'

    if ensemble_abstracts == '':
        ensemble_abstracts = 'No ensemble abstracts available.'
    PROMPT_RESEARCHER = PROMPT_RESEARCHER.replace('{ensemble_abstracts}', ensemble_abstracts)


    PROMPT_NOVELTY = PROMPT_NOVELTY.replace('{100statestxt}', states)
    PROMPT_NOVELTY = PROMPT_NOVELTY.replace('{previous_results}', previous_results)
    PROMPT_NOVELTY = replace_tags(PROMPT_NOVELTY)
    PROMPT_NOVELTY = PROMPT_NOVELTY.replace('{ensemble_abstracts}', ensemble_abstracts)

    state = {
        'researcher': [PROMPT_RESEARCHER],
        'novelty_supervisor': [PROMPT_NOVELTY],
    }

    #add full pytheus paper to researcher prompt
    # addition = """
    # You are a helpful AI physicist
    # Here we provide the full Pytheus paper for reference:
    # """
    with open(PATH_PYTHEUS_PAPER, 'r') as f:
        pytheuspaper = f.read()
    

    ### JUDGE ###
    # Inject Pytheus paper into the judge prompt
    PROMPT_JUDGE = PROMPT_JUDGE.replace('<pytheuspaper>', pytheuspaper)

    #save prompts as txt files to run_dir
    if run_dir is not None:
        with open(f'{run_dir}/PROMPT_RESEARCHER.txt', 'w') as f:
            f.write(PROMPT_RESEARCHER)
        with open(f'{run_dir}/PROMPT_NOVELTY.txt', 'w') as f:
            f.write(PROMPT_NOVELTY)
        with open(f'{run_dir}/PROMPT_JUDGE.txt', 'w') as f:
            f.write(PROMPT_JUDGE)
    return state



def print_reply(reply: str, agent_name: str) -> None:
    """Pretty-print an agent's reply, styling key fields for readability."""
    reply_mod = re.sub(r'(Action:|Action Input:)', r'\033[3m\1\033[0m', reply)
    phrases_to_style = ['Action:', 'Action Input:', 'Thought:', 'Final Answer', 'Feedback Researcher', 'Feedback Expert', '[Pass to Expert]', 'Criticism', 'Reject','[Accept]']
    pattern = r'(' + '|'.join(map(re.escape, phrases_to_style)) + r')'
    reply_mod = re.sub(pattern, r'\033[3m\1\033[0m', reply_mod)
    print(f"\033[1m{agent_name}\033[0m: {reply_mod}\n")

def researcher_agent():
    agent_name = 'Researcher'
    reply = call_openai_api(state['researcher'], agent_name)
    print_reply(reply, agent_name)
    action, action_input = parse_agent_reply(reply)
    if action == 'arxiv':
        arxiv_result = arxiv_tool(action_input)
        write_log('ArxivTool', action_input, arxiv_result)
        state['researcher'].append(f"ArxivTool: {arxiv_result}")
        return researcher_agent

    elif action == 'final answer':
        state['researcher'].append(f"Researcher: {reply}")
        state['novelty_supervisor'].append(f"Suggestion by Researcher: {action_input}")
        return novelty_supervisor_agent

    else:
        print(f'{agent_name}: NO KNOWN ACTION!')
        return researcher_agent
    
def novelty_supervisor_agent():
    global statsfile
    agent_name = 'NoveltySupervisor'
    state['novelty_supervisor'].append(f"System Message: Do not reject proposals for being too similar to previously suggestions by the researcher that you have accepted before..")
    reply = call_openai_api(state['novelty_supervisor'], agent_name)
    researcher_suggestion = state['researcher'][-1]
    state['novelty_supervisor'].append(f"Novelty Supervisor: {reply}")
    print_reply(reply, agent_name)
    action, action_input = parse_agent_reply(reply)
    if action == 'reject':
        state['researcher'].append(f"NoveltySupervisor Feedback: {action_input}")
        with open(statsfile, 'a') as f:
            f.write(f"{timestamp},rejected by novelty supervisor\n")
        return researcher_agent
    elif action == 'accept':
        with open(statsfile, 'a') as f:
            f.write(f"{timestamp},accepted by novelty supervisor\n")
        return write_to_ensembledir(role='novelty')

def judge_agent(researcher_abstract: str):
    global state
    global PROMPT_JUDGE
    agent_name = 'Judge'
    prompt = PROMPT_JUDGE
    state['judge'] = [prompt.replace('<research_idea>', researcher_abstract)]
    reply = call_openai_api(state['judge'], agent_name)
    print_reply(reply, agent_name)
    action, action_input = parse_agent_reply(reply)
    if action == 'reject':
        state['researcher'].append(f"Judge Feedback: {action_input}")
        with open(statsfile, 'a') as f:
            f.write(f"{timestamp},rejected by judge\n")
        return researcher_agent
    elif action == 'accept':
        with open(statsfile, 'a') as f:
            f.write(f"{timestamp},accepted by judge\n")
        return write_to_ensembledir(role='judge', researcher_abstract=researcher_abstract)
    
def mediator_agent():
    global short_log_file_name
    global PROMPT_RESEARCHER
    global PROMPT_NOVELTY
    global PROMPT_JUDGE
    global run_dir

    prompt = f"""
    You are a helpful assistant, evaluating the conversation between three LLM agents.

    You are reading a conversation between three LLM agents with the following prompts:

    Researcher:
    <START OF RESEARCHER PROMPT>
    {PROMPT_RESEARCHER}
    <END OF RESEARCHER PROMPT>

    Novelty Supervisor:
    <START OF NOVELTY PROMPT>
    {PROMPT_NOVELTY}
    <END OF NOVELTY PROMPT>

    Judge:
    <START OF JUDGE PROMPT>
    {PROMPT_JUDGE}
    <END OF JUDGE PROMPT>

    Here is a log of the conversation between the three agents:
    <shortlog>

    Evaluate the conversation and provide a summary of the discussion, including the main points made by each agent, any disagreements or conflicts, and the overall outcome of the conversation. The summary should be concise and focused on the key aspects of the discussion. Highlight any inefficiencies in the conversation or areas for improvement.
    Summarize your critique in a few bullet points.

    Also, finalize your output by speaking directly to the Researcher and giving them actionable feedback for the next iteration to avoid further inefficiencies.

    Output format:
    [discussion]
    [bullet points]
    "Mediator feedback for researcher:"
    """

    with open(short_log_file_name, 'r') as f:
        shortlog = f.read()
    
    # Call the generate function with the prompt and shortlog
    prompt = prompt.replace("<shortlog>", shortlog)

    reply = call_openai_api([prompt], 'Mediator')

    #split reply to only get feedback
    feedback = reply.split('Mediator feedback for researcher:')[-1]

    message = 'Important Feedback given by Mediator watching the conversation: ' + feedback
    state['researcher'].append(message)



def write_to_ensembledir(role: str = 'novelty', researcher_abstract: str = ''):
    global ensemble_dir
    global timestamp
    #create ensemble directory if it does not exist
    if not os.path.exists(ensemble_dir):
        os.makedirs(ensemble_dir, exist_ok=True)

    if not researcher_abstract:
        #summarize the last suggestion in a two sentence abstract with abstract
        abstract_prompt = "\nTask Update: Please write a concise title and two-sentence mini-abstract, summarizing and explaining the idea for the current experiment. Mention the target explicitly."
        researcher_abstract = call_openai_api(state['researcher'][-1]+abstract_prompt, "Researcher Summarizing")

    
    if role == 'novelty':
        #write the last suggestion to the ensemble directory
        with open(f'{ensemble_dir}/novelty_abstracts_{timestamp}.txt', 'a') as f:
            f.write(researcher_abstract)
        return judge_agent(researcher_abstract)
    else:
        print("WRITING ABSTRACT TO FILE AFTER JUDGE ACCEPTED")
        with open(f'{ensemble_dir}/abstracts_{timestamp}.txt', 'a') as f:
            f.write(researcher_abstract)


if __name__ == '__main__':

    # CLI flags for paths and options
    parser = argparse.ArgumentParser(description='Run researchers with configurable paths.')
    parser.add_argument('--ensemble-dir', default='ensemble')
    parser.add_argument('--prompt-researcher', default=PATH_PROMPT_RESEARCHER)
    parser.add_argument('--prompt-novelty', default=PATH_PROMPT_NOVELTY)
    parser.add_argument('--prompt-judge', default=PATH_PROMPT_JUDGE)
    parser.add_argument('--recent-papers-file', default=PATH_RECENT_PAPERS)
    parser.add_argument('--pytheus-explicit-infos', default=PATH_PYTHEUS_EXPLICIT_INFOS)
    parser.add_argument('--states-file', default=PATH_STATES_FILE)
    parser.add_argument('--concept-pairs-file', default=PATH_CONCEPT_PAIRS_FILE)
    parser.add_argument('--pytheus-examples-dir', default=PATH_PYTHEUS_EXAMPLES_DIR)
    parser.add_argument('--pytheus-paper', default=PATH_PYTHEUS_PAPER)
    parser.add_argument('--max-researcher-calls', type=int, default=20)
    parser.add_argument('--journal-dir', default=None)
    parser.add_argument('--all-journals', action='store_true', default=True)
    args = parser.parse_args()

    # Apply CLI configurations

    PATH_PROMPT_RESEARCHER = args.prompt_researcher
    PATH_PROMPT_NOVELTY = args.prompt_novelty
    PATH_PROMPT_JUDGE = args.prompt_judge
    PATH_RECENT_PAPERS = args.recent_papers_file
    PATH_PYTHEUS_EXPLICIT_INFOS = args.pytheus_explicit_infos
    PATH_STATES_FILE = args.states_file
    PATH_CONCEPT_PAIRS_FILE = args.concept_pairs_file
    PATH_PYTHEUS_EXAMPLES_DIR = args.pytheus_examples_dir
    PATH_PYTHEUS_PAPER = args.pytheus_paper
    ensemble_dir = args.ensemble_dir

    # Get the global task rank
    process_id = int(os.environ.get('SLURM_PROCID', 0))
    # Alternatively, if you need a node-local identifier:
    local_id = int(os.environ.get('SLURM_LOCALID', 0))
    # sbatch array id
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    process_id = str(process_id)+'_'+str(local_id)+'_'+str(task_id)
    print(f"Process index: {process_id}")

    # Ensure ensemble directory and statsfile exist
    if not os.path.exists(ensemble_dir):
        os.makedirs(ensemble_dir, exist_ok=True)

    statsfile = f'{ensemble_dir}/stats.csv'
    if not os.path.exists(statsfile):
        with open(statsfile, 'w') as f:
            f.write('timestamp,decision\n')

    # Fixed model configuration
    model_descr = MODEL

    journal_dir = args.journal_dir
    all_journals = args.all_journals
    MAX_RESEARCHER_CALLS = args.max_researcher_calls

    for _ in range(20):
        #coinflip if mediator is used
        mediator_used = random.choice([True, False])

        timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H%M_%S')
        print(f"Starting new run at {timestamp}")
        run_dir = f"{ensemble_dir}/ensemble_runs/{model_descr}_{timestamp}"
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "info.txt"), "w") as f:
            f.write(f"model {model_descr}")
            f.write(f"\nmediator_used {mediator_used}")
        log_file_name = os.path.join(run_dir, f"log_{timestamp}.txt")
        short_log_file_name = os.path.join(run_dir, f"short_log_{timestamp}.txt")
        
        
        # [1,x,x,x]=State
        # [x,1,x,x]=Gate -- 18 gates in total
        # [x,x,1,x]=Measurement -- 5 measurements in total
        # [x,x,x,1]=Protocol -- 4 protocols in total

        suggestion_type = random.choice([
            [0, 0, 0, 5],  
            # [0, 0, 3, 3],  
            # [0, 0, 1, 4]
        ])
    
        # suggestion_type=[0, 3, 5, 4]
        
        state=init_state(suggestion_type, journal_dir=journal_dir)

        #save state to file
        with open(os.path.join(run_dir, "init_state.json"), "w") as f:
            json.dump(state, f, indent=4)
        write_log('SuggestionType', 'suggestion_type', str(suggestion_type))
    
        current_agent_function = researcher_agent
        researcher_calls = 0
        while True:
            if current_agent_function == researcher_agent:
                if researcher_calls > 0:
                    state['researcher'].append("Important reminder about task:\n"+REPEATER_RESEARCHER)
                researcher_calls += 1
                if mediator_used and researcher_calls%3==0:
                    mediator_agent()
                next_agent_function = current_agent_function()
            else:
                next_agent_function = current_agent_function()

            if next_agent_function is None or researcher_calls >= MAX_RESEARCHER_CALLS:
                print("Conversation ended.")
                print(f'researcher_calls={researcher_calls}')
                print(f'next_agent_function={next_agent_function}')
                print(f'timestamp={timestamp}')
                print("=" * 20)
                write_log('ConversationEnd', 'stats', f'researcher_calls={researcher_calls}')
                break
            current_agent_function = next_agent_function
    

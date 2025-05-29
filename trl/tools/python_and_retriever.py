from llama_index.core import SummaryIndex
from llama_index.readers.web import SimpleWebPageReader
import httpx
import os
from llama_index.llms.openai import OpenAI
import io
import traceback
import tiktoken
from tqdm import tqdm
import threading
import types
import builtins
import multiprocessing
import sys
import builtins
import types
import time

from trl.tools.key_rotator import google_key_rotator, get_openai_client

# base_model = 'gpt-4o-2024-08-06'
# base_model = 'o3-mini-2025-01-31'
base_model = 'gpt-4.1-nano-2025-04-14'

# client = openai.OpenAI(
#     api_key=os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY"),
    # base_url=os.environ.get("OPENAI_BASE_URL", "https://hk.xty.app/v1"),
    # http_client=httpx.Client(
    #     base_url="https://hk.xty.app/v1",
    #     follow_redirects=True,
    # ),
# )

def count_tokens(text, model_name=base_model):
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    return len(tokens)

def remove_python_prefix(text):
    if text.startswith("```python") and text.rstrip().endswith("```"):
        # Remove the starting and ending code block markers
        lines = text.strip().splitlines()
        return "\n".join(lines[1:-1])
    return text

def polish_query(context,query):
    '''
    TODO
    This function is used to polish the query problem based on the previous context.
    '''
    prompt = f"**Context**:{context} **Query**:{query}"
    
    completion = client.chat.completions.create(
        model=base_model, 
        messages=[
            {"role": "system", "content": "You are a helpful assistant. You need to polish the query based on the previous context."},
            {"role": "user", "content": f"{prompt}"}
        ]
    )
    return completion.choices[0].message.content.strip()


def google_retriever(query, retrieve_websites = 2):
    '''
    query[str]: 
    ''' 
    base_url = 'https://www.googleapis.com/customsearch/v1'
    google_key = google_key_rotator.next()
    # params = {
    #         'key': '',
    #         'cx': '',
    #         'q': query,
    #         'num': retrieve_websites # The number of websites you want.
    # }
    params = {
            'key': google_key,
            'cx': '',
            'q': query,
            'num': retrieve_websites # The number of websites you want.
    }    
    #TODO: There might some connection error when we call httpx.get. Simply recall will resolve this error.
    response = httpx.get(base_url, params=params,timeout=5)
    response.raise_for_status()
    data = response.json()
    import openai
    openai.api_key = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY") 
    # openai.base_url = os.environ.get("OPENAI_BASE_URL", "https://hk.xty.app/v1")
    # openai.http_client = httpx.Client(
    #         base_url="https://hk.xty.app/v1",
    #         follow_redirects=True,
    #     )
        
    urls = [item.get('link', '') for item in data.get('items', [])]
    
    for i in range(retrieve_websites):
        try:
            document = SimpleWebPageReader(html_to_text=True).load_data(
            [urls[i]]
            )
            index = SummaryIndex.from_documents(document)
            llm = OpenAI(model=base_model)
            query_engine = index.as_query_engine(llm=llm)
            if count_tokens(query)>80000:
                continue
            else:
                response = query_engine.query(query + ' Use only one paragraph to summarize your answer.')
            return response
        except:
            continue

def _isolated_worker(code: str, return_dict):
    """Runs untrusted code in an isolated process, capturing stdout/stderr."""
    try:
        stdout = io.StringIO()
        stderr = io.StringIO()

        # Override builtins.print
        custom_builtins = builtins.__dict__.copy()
        def custom_print(*args, **kwargs):
            print(*args, file=stdout, **kwargs)
        custom_builtins['print'] = custom_print

        # Fake sys module
        fake_sys = types.SimpleNamespace(stderr=stderr)

        exec_globals = {
            '__builtins__': custom_builtins,
            'sys': fake_sys,
        }

        exec(code, exec_globals)

        return_dict["stdout"] = stdout.getvalue()
        return_dict["stderr"] = stderr.getvalue()
        return_dict["status"] = "success"

    except Exception:
        return_dict["stdout"] = ""
        return_dict["stderr"] = traceback.format_exc()
        return_dict["status"] = "exception"


def safe_execute_and_capture(code: str, client, max_attempts: int = 3, timeout: float = 30.0):
    code = remove_python_prefix(code)

    for attempt in range(max_attempts):
        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        proc = multiprocessing.Process(target=_isolated_worker, args=(code, return_dict))
        proc.start()
        proc.join(timeout=timeout)

        if proc.is_alive():
            proc.kill()
            return_dict["stdout"] = ""
            return_dict["stderr"] = f"Execution timed out: code exceeded {timeout} seconds.\n"
            return_dict["status"] = "timeout"

        stdout = return_dict.get("stdout", "")
        stderr = return_dict.get("stderr", "")
        status = return_dict.get("status", "unknown")

        if stdout.strip():
            return code, stdout, stderr

        # Retry with model-generated debug code
        error_output = stderr or "No output was produced."
        debug_prompt = (
            f"The following Python code ran without producing stdout:\n{code}\n"
            f"Error or empty stdout:\n{error_output}\n"
            "Please debug it and add print statements as needed. "
            "Only return the modified Python code."
        )

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": debug_prompt}],
            model=base_model,
        )
        code = remove_python_prefix(response.choices[0].message.content.strip())
        print("Retrying with modified code:\n", code)

    raise Exception(f"Python Debug Fail after {max_attempts} attempts.")


def execute_and_capture(code: str, client, max_attempts: int = 3) -> tuple:
    """
    Execute the given Python code and capture its stdout and stderr
    in a thread-safe manner by overriding builtins.print and sys.stderr.
    """
    code = remove_python_prefix(code)
    
    for attempt in range(max_attempts):
        # Thread-local buffers
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        def run_code():
            # Custom builtins with overridden print
            custom_builtins = builtins.__dict__.copy()
            def custom_print(*args, **kwargs):
                builtins.print(*args, file=stdout_capture, **kwargs)
            custom_builtins['print'] = custom_print

            # Fake sys for stderr capture
            fake_sys = types.SimpleNamespace(stderr=stderr_capture)

            # Execution environment
            globals_dict = {
                '__builtins__': custom_builtins,
                'sys': fake_sys,
            }

            try:
                exec(code, globals_dict)
            except Exception:
                traceback.print_exc(file=stderr_capture)

        # Run code on a separate thread
        thread = threading.Thread(target=run_code)
        thread.start()
        thread.join(timeout=30)
        if thread.is_alive():
            stderr_capture.write("Execution timed out: code exceeded 30 seconds.\n")
            thread.join()  # Clean up thread

        stdout_str = stdout_capture.getvalue()
        stderr_str = stderr_capture.getvalue()

        # If stdout is produced, return immediately
        if stdout_str.strip():
            return code, stdout_str, stderr_str

        # Prepare debug prompt for AI
        error_output = stderr_str or "No output was produced."
        debug_prompt = (
            f"The following Python code ran without producing stdout:\n{code}\n"
            f"Error or empty stdout:\n{error_output}\n"
            "Please debug it and add print statements as needed. "
            "Only return the modified Python code."
        )

        # Request modified code from AI
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": debug_prompt}],
            model=base_model,
        )
        code = remove_python_prefix(response.choices[0].message.content.strip())
        print("Retrying with modified code:\n", code)

    raise Exception(f"Python Debug Fail after {max_attempts} attempts.")

def python_interpreter(code: str) -> str:
    """
    Executes the given Python code, captures its output and errors, and returns 
    an analysis report explaining the code's functionality and how the output reflects 
    the code's intended behavior.
    """
    client = get_openai_client()
    code, output, error_output = execute_and_capture(code, client, 5)
    if output:
        return code, output
    else:
        return code, error_output


if __name__ == "__main__":
    # context = ''
    # query  = "What is love? "
    query = "What does non-commutative mean in the context of ring theory?"
    for i in range(3):
        response = google_retriever(query)
        print(i)
    print(response)
    # # # with open('./retriever_response.txt', 'w') as file:
    # # #    file.write(str(response))
    # # # # The text shows in the terminal might be incomplete. Please refer to retriever_response.txt.
    # print(response)
    query = """
def trap(height):
    n = len(height)
    if n <= 2: return 0
    left, right = 0, n - 1
    left_max, right_max = height[left], height[right]
    water = 0
    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1
    return water

height = [0,1,0,2,1,0,1,3,2,1,2,1]
result = trap(height)
print(f"Total water trapped: {result}")
"""
    query = "2022/1347"
    for i in range(3):
        response = python_interpreter(query)
        print(i)
    print(response)

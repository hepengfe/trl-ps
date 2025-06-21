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
import time

from trl.tools.key_rotator import openai_key_rotator, google_key_rotator, get_openai_client

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

def count_tokens(text, model_name=None):
    # Fall back to cl100k_base if the model is not recognized
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def remove_python_prefix(text):
    if text.startswith("```python") and text.rstrip().endswith("```"):
        # Remove the starting and ending code block markers
        lines = text.strip().splitlines()
        return "\n".join(lines[1:-1])
    return text

import concurrent.futures

def run_with_timeout(func, args=(), kwargs=None, timeout=30):
    kwargs = kwargs or {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        return future.result(timeout=timeout)

def google_retriever(query, retrieve_websites = 2):
    '''
    query[str]: 
    ''' 
    base_url = 'https://www.googleapis.com/customsearch/v1'
    google_key = google_key_rotator.next()
    params = {
            'key': google_key,
            'cx': '',
            'q': query,
            'num': retrieve_websites # The number of websites you want.
    }    
    #TODO: There might some connection error when we call httpx.get. Simply recall will resolve this error.
    response = httpx.get(base_url, params=params, timeout=5)
    response.raise_for_status()
    data = response.json()
    import openai
    openai.api_key = openai_key_rotator.next()
    # openai.base_url = os.environ.get("OPENAI_BASE_URL", "https://hk.xty.app/v1")
    # openai.http_client = httpx.Client(
    #         base_url="https://hk.xty.app/v1",
    #         follow_redirects=True,
    #     )
        
    urls = [item.get('link', '') for item in data.get('items', [])]
    reader = SimpleWebPageReader(html_to_text=True)

    for i in range(retrieve_websites):
        try:
            # Sometimes this hangs for very long so allow for timeout
            document = run_with_timeout(reader.load_data, args=([urls[i]],), timeout=30)
            index = SummaryIndex.from_documents(document)
            llm = OpenAI(model=base_model)
            query_engine = index.as_query_engine(llm=llm)
            if count_tokens(query) > 80000:
                continue
            else:
                response = query_engine.query(query + ' Use only one paragraph to summarize your answer.')
            return response
        except concurrent.futures.TimeoutError:
            print(f"Timeout while loading URL: {urls[i]}")
            continue
        except Exception as e:
            print(e)
            continue

def execute_and_capture_threading(code: str, client, max_attempts: int = 3) -> tuple:
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
            def blocked_func(name):
                def f(*args, **kwargs):
                    raise RuntimeError(f"{name}() is disabled in this environment")
                return f
            custom_builtins['print'] = custom_print
            custom_builtins['input'] = blocked_func("input")
            custom_builtins['open'] = blocked_func("open")

            # Fake sys for stderr capture
            fake_sys = types.SimpleNamespace(stderr=stderr_capture)

            # Execution environment
            globals_dict = {
                '__builtins__': custom_builtins,
                'sys': fake_sys,
                '__name__': '__main__',
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
            print("Execution timed out, terminating thread.")
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
        print("Retrying with modified code")

    raise Exception(f"Python Debug Fail after {max_attempts} attempts.")


def execute_and_capture(code: str, client, max_attempts: int = 3) -> tuple:
    """
    Execute the given Python code and capture its stdout and stderr
    in a process-safe manner by overriding builtins.print and sys.stderr.
    """
    code = remove_python_prefix(code)

    def run_code_in_process(code: str, queue: multiprocessing.Queue):
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # Custom builtins with overridden print
        custom_builtins = builtins.__dict__.copy()

        def custom_print(*args, **kwargs):
            builtins.print(*args, file=stdout_capture, **kwargs)

        def blocked_func(name):
            def f(*args, **kwargs):
                raise RuntimeError(f"{name}() is disabled in this environment")
            return f

        custom_builtins['print'] = custom_print
        custom_builtins['input'] = blocked_func("input")
        custom_builtins['open'] = blocked_func("open")

        # Fake sys for stderr capture
        fake_sys = types.SimpleNamespace(stderr=stderr_capture)

        # Execution environment
        globals_dict = {
            '__builtins__': custom_builtins,
            'sys': fake_sys,
            '__name__': '__main__',
        }

        try:
            exec(code, globals_dict)
        except Exception:
            traceback.print_exc(file=stderr_capture)

        queue.put((stdout_capture.getvalue(), stderr_capture.getvalue()))

    for attempt in range(max_attempts):
        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=run_code_in_process, args=(code, queue))
        process.start()
        process.join(timeout=30)
        if process.is_alive():
            print("Execution timed out, terminating process.")
            process.terminate()
            process.join()
            stdout_str = ""
            stderr_str = "Execution timed out: code exceeded 30 seconds.\n"
        else:
            try:
                stdout_str, stderr_str = queue.get_nowait()
            except Exception:
                stdout_str, stderr_str = "", "Failed to retrieve output from process.\n"

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
        print("Retrying with modified code")

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
    
    # testing retriever
    query = "What does non-commutative mean in the context of ring theory?"
    for i in range(3):
        response = google_retriever(query)
        print(i)
    print(response)
    
    # testing python interpreter
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

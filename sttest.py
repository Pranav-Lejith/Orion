import streamlit as st
import os
import base64
import zipfile
import io
import re
import json
import time
from PIL import Image
import numpy as np
import textwrap

# Set page configuration
st.set_page_config(
    page_title="Programming Language Creator",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stTextInput, .stTextArea {
        background-color: #262730;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .language-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #4CAF50;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #03A9F4;
    }
    .info-box {
        background-color: #1E1E1E;
        border-left: 5px solid #03A9F4;
        padding: 1rem;
        border-radius: 5px;
    }
    .success-box {
        background-color: #1E1E1E;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        border-radius: 5px;
    }
    .warning-box {
        background-color: #1E1E1E;
        border-left: 5px solid #FFC107;
        padding: 1rem;
        border-radius: 5px;
    }
    .ascii-art {
        font-family: monospace;
        white-space: pre;
        font-size: 0.8rem;
        line-height: 1;
        color: #03A9F4;
    }
    .feature-card {
        background-color: #1E1E1E;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .delete-btn {
        color: #FF5252;
    }
    .add-btn {
        color: #4CAF50;
    }
    table {
        width: 100%;
    }
    th, td {
        text-align: left;
        padding: 8px;
    }
    th {
        background-color: #262730;
    }
    tr:nth-child(even) {
        background-color: #1E1E1E;
    }
    tr:nth-child(odd) {
        background-color: #262730;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1E1E1E;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'language_name' not in st.session_state:
    st.session_state.language_name = ""
if 'language_extension' not in st.session_state:
    st.session_state.language_extension = ""
if 'keywords' not in st.session_state:
    st.session_state.keywords = []
if 'functions' not in st.session_state:
    st.session_state.functions = []
if 'ascii_art' not in st.session_state:
    st.session_state.ascii_art = ""
if 'syntax_rules' not in st.session_state:
    st.session_state.syntax_rules = []
if 'download_ready' not in st.session_state:
    st.session_state.download_ready = False
if 'interpreter_template' not in st.session_state:
    st.session_state.interpreter_template = ""
if 'installation_script' not in st.session_state:
    st.session_state.installation_script = ""
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'theme_color' not in st.session_state:
    st.session_state.theme_color = "#03A9F4"
if 'logo_image' not in st.session_state:
    st.session_state.logo_image = None
if 'show_preview' not in st.session_state:
    st.session_state.show_preview = False

# Function to create a zip file in memory
def create_zip_file():
    # Create a BytesIO object
    zip_buffer = io.BytesIO()
    
    # Create a ZipFile object
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add interpreter.py
        zipf.writestr('interpreter.py', generate_interpreter())
        
        # Add installation script
        zipf.writestr('install.bat', generate_installation_script())
        
        # Add README.md
        zipf.writestr('README.md', generate_readme())
        
        # Add example file
        zipf.writestr(f'example.{st.session_state.language_extension}', generate_example_file())
    
    # Reset buffer position
    zip_buffer.seek(0)
    return zip_buffer

# Function to generate the interpreter code
def generate_interpreter():
    # Start with the base interpreter template
    interpreter_code = """import sys
import operator
import os
import importlib
import functools
import random
import math
import datetime
import json
import re
import collections
import itertools
import statistics
import urllib.request
import xml.etree.ElementTree as ET
import csv
import sqlite3
import hashlib
import base64
import zlib
import threading
import multiprocessing
import asyncio
import typing
from termcolor import colored

try:
    import numpy as np
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

"""

    # Add language-specific constants
    interpreter_code += f"""
printOptions = ["{st.session_state.language_name} Programming Language", "Created with Programming Language Creator"]
devCommands = ['developer', 'command override', 'emergency override']

"""

    # Add the interpreter class
    interpreter_code += f"""
class {st.session_state.language_name}Interpreter:
    def __init__(self, filename=None, interactive=False):
        self.filename = filename
        self.variables = {{}}
        self.functions = {{}}
        self.classes = {{}}
        self.lines = []
        self.current_line = 0
        self.interactive = interactive
        self.ops = {{
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '%': operator.mod,
            '**': operator.pow,
            '//': operator.floordiv,
            '==': operator.eq,
            '!=': operator.ne,
            '<': operator.lt,
            '<=': operator.le,
            '>': operator.gt,
            '>=': operator.ge,
            'and': operator.and_,
            'or': operator.or_,
            'not': operator.not_,
        }}
"""

    # Add built-in functions
    interpreter_code += """
    # Built-in functions
    def len_func(self, args):
        return len(args[0])

    def max_func(self, args):
        return max(args[0])

    def min_func(self, args):
        return min(args[0])

    def sum_func(self, args):
        return sum(args[0])

    def abs_func(self, args):
        return abs(args[0])

    def round_func(self, args):
        return round(*args)

    def type_func(self, args):
        return type(args[0]).__name__

    def int_func(self, args):
        return int(args[0])

    def float_func(self, args):
        return float(args[0])

    def str_func(self, args):
        return str(args[0])

    def bool_func(self, args):
        return bool(args[0])

    def list_func(self, args):
        return list(args[0])

    def tuple_func(self, args):
        return tuple(args[0])

    def set_func(self, args):
        return set(args[0])

    def dict_func(self, args):
        return dict(args[0])

    def range_func(self, args):
        return list(range(*args))

    def enumerate_func(self, args):
        return list(enumerate(*args))

    def zip_func(self, args):
        return list(zip(*args))

    def map_func(self, args):
        return list(map(*args))

    def filter_func(self, args):
        return list(filter(*args))

    def reduce_func(self, args):
        return functools.reduce(*args)

    def sorted_func(self, args):
        return sorted(*args)

    def reversed_func(self, args):
        return list(reversed(args[0]))

    def any_func(self, args):
        return any(args[0])

    def all_func(self, args):
        return all(args[0])

    def chr_func(self, args):
        return chr(args[0])

    def ord_func(self, args):
        return ord(args[0])

    def bin_func(self, args):
        return bin(args[0])

    def oct_func(self, args):
        return oct(args[0])

    def hex_func(self, args):
        return hex(args[0])

    def id_func(self, args):
        return id(args[0])

    def isinstance_func(self, args):
        return isinstance(args[0], args[1])

    def issubclass_func(self, args):
        return issubclass(args[0], args[1])

    def callable_func(self, args):
        return callable(args[0])

    def getattr_func(self, args):
        return getattr(*args)

    def setattr_func(self, args):
        setattr(*args)

    def hasattr_func(self, args):
        return hasattr(*args)

    def delattr_func(self, args):
        delattr(*args)

    def open_func(self, args):
        return open(*args)

    def input_func(self, args):
        return input(*args)

    def print_func(self, args):
        print(*args)

    def upper_func(self, args):
        return args[0].upper()

    def lower_func(self, args):
        return args[0].lower()

    def capitalize_func(self, args):
        return args[0].capitalize()

    def title_func(self, args):
        return args[0].title()

    def strip_func(self, args):
        return args[0].strip()

    def split_func(self, args):
        return args[0].split(*args[1:])

    def join_func(self, args):
        return args[0].join(args[1])

    def replace_func(self, args):
        return args[0].replace(*args[1:])

    def startswith_func(self, args):
        return args[0].startswith(args[1])

    def endswith_func(self, args):
        return args[0].endswith(args[1])

    def find_func(self, args):
        return args[0].find(*args[1:])

    def count_func(self, args):
        return args[0].count(args[1])

    def isalpha_func(self, args):
        return args[0].isalpha()

    def isdigit_func(self, args):
        return args[0].isdigit()

    def isalnum_func(self, args):
        return args[0].isalnum()

    def islower_func(self, args):
        return args[0].islower()

    def isupper_func(self, args):
        return args[0].isupper()

    def append_func(self, args):
        args[0].append(args[1])

    def extend_func(self, args):
        args[0].extend(args[1])

    def insert_func(self, args):
        args[0].insert(args[1], args[2])

    def remove_func(self, args):
        args[0].remove(args[1])

    def pop_func(self, args):
        return args[0].pop(*args[1:])

    def clear_func(self, args):
        args[0].clear()

    def index_func(self, args):
        return args[0].index(*args[1:])

    def reverse_func(self, args):
        args[0].reverse()

    def copy_func(self, args):
        return args[0].copy()

    def deepcopy_func(self, args):
        import copy
        return copy.deepcopy(args[0])

    def keys_func(self, args):
        return list(args[0].keys())

    def values_func(self, args):
        return list(args[0].values())

    def items_func(self, args):
        return list(args[0].items())

    def get_func(self, args):
        return args[0].get(*args[1:])

    def update_func(self, args):
        args[0].update(args[1])

    def math_sin_func(self, args):
        return math.sin(args[0])

    def math_cos_func(self, args):
        return math.cos(args[0])

    def math_tan_func(self, args):
        return math.tan(args[0])

    def math_sqrt_func(self, args):
        return math.sqrt(args[0])

    def math_log_func(self, args):
        return math.log(*args)

    def math_exp_func(self, args):
        return math.exp(args[0])

    def math_floor_func(self, args):
        return math.floor(args[0])

    def math_ceil_func(self, args):
        return math.ceil(args[0])

    def random_randint_func(self, args):
        return random.randint(*args)

    def random_choice_func(self, args):
        return random.choice(args[0])

    def random_shuffle_func(self, args):
        random.shuffle(args[0])

    def datetime_now_func(self, args):
        return datetime.datetime.now()

    def datetime_date_func(self, args):
        return datetime.date(*args)

    def datetime_time_func(self, args):
        return datetime.time(*args)

    def json_dumps_func(self, args):
        return json.dumps(*args)

    def json_loads_func(self, args):
        return json.loads(*args)

    def re_search_func(self, args):
        return re.search(*args)

    def re_match_func(self, args):
        return re.match(*args)

    def re_findall_func(self, args):
        return re.findall(*args)

    def re_sub_func(self, args):
        return re.sub(*args)

    def collections_counter_func(self, args):
        return collections.Counter(args[0])

    def collections_defaultdict_func(self, args):
        return collections.defaultdict(args[0])

    def itertools_permutations_func(self, args):
        return list(itertools.permutations(*args))

    def itertools_combinations_func(self, args):
        return list(itertools.combinations(*args))

    def statistics_mean_func(self, args):
        return statistics.mean(args[0])

    def statistics_median_func(self, args):
        return statistics.median(args[0])

    def statistics_mode_func(self, args):
        return statistics.mode(args[0])

    def statistics_stdev_func(self, args):
        return statistics.stdev(args[0])

    def urllib_request_urlopen_func(self, args):
        return urllib.request.urlopen(*args)

    def xml_parse_func(self, args):
        return ET.parse(*args)

    def csv_reader_func(self, args):
        return csv.reader(*args)

    def csv_writer_func(self, args):
        return csv.writer(*args)

    def sqlite3_connect_func(self, args):
        return sqlite3.connect(*args)

    def hashlib_md5_func(self, args):
        return hashlib.md5(args[0].encode()).hexdigest()

    def hashlib_sha256_func(self, args):
        return hashlib.sha256(args[0].encode()).hexdigest()

    def base64_encode_func(self, args):
        return base64.b64encode(args[0].encode()).decode()

    def base64_decode_func(self, args):
        return base64.b64decode(args[0]).decode()

    def zlib_compress_func(self, args):
        return zlib.compress(args[0].encode())

    def zlib_decompress_func(self, args):
        return zlib.decompress(args[0]).decode()

    def threading_thread_func(self, args):
        return threading.Thread(*args)

    def multiprocessing_process_func(self, args):
        return multiprocessing.Process(*args)

    def asyncio_run_func(self, args):
        return asyncio.run(*args)

    def typing_get_type_hints_func(self, args):
        return typing.get_type_hints(*args)

    # TensorFlow functions (if available)
    def tf_constant_func(self, args):
        if TENSORFLOW_AVAILABLE:
            return tf.constant(*args)
        else:
            raise Exception("TensorFlow is not available")

    def tf_variable_func(self, args):
        if TENSORFLOW_AVAILABLE:
            return tf.Variable(*args)
        else:
            raise Exception("TensorFlow is not available")

    def tf_matmul_func(self, args):
        if TENSORFLOW_AVAILABLE:
            return tf.matmul(*args)
        else:
            raise Exception("TensorFlow is not available")

    def np_array_func(self, args):
        if TENSORFLOW_AVAILABLE:
            return np.array(*args)
        else:
            raise Exception("NumPy is not available")

    def np_zeros_func(self, args):
        if TENSORFLOW_AVAILABLE:
            return np.zeros(*args)
        else:
            raise Exception("NumPy is not available")

    def np_ones_func(self, args):
        if TENSORFLOW_AVAILABLE:
            return np.ones(*args)
        else:
            raise Exception("NumPy is not available")
"""

    # Add custom functions
    interpreter_code += "\n    # Custom functions\n"
    for func in st.session_state.functions:
        func_name = func['name']
        func_code = func['code']
        # Indent the function code properly
        indented_code = textwrap.indent(func_code, ' ' * 8)
        interpreter_code += f"""    def {func_name}(self, args):
{indented_code}
"""

    # Add built-in functions dictionary
    interpreter_code += """
    # Register built-in functions
    builtin_functions = {
        'len': len_func,
        'max': max_func,
        'min': min_func,
        'sum': sum_func,
        'abs': abs_func,
        'round': round_func,
        'type': type_func,
        'int': int_func,
        'float': float_func,
        'str': str_func,
        'bool': bool_func,
        'list': list_func,
        'tuple': tuple_func,
        'set': set_func,
        'dict': dict_func,
        'range': range_func,
        'enumerate': enumerate_func,
        'zip': zip_func,
        'map': map_func,
        'filter': filter_func,
        'reduce': reduce_func,
        'sorted': sorted_func,
        'reversed': reversed_func,
        'any': any_func,
        'all': all_func,
        'chr': chr_func,
        'ord': ord_func,
        'bin': bin_func,
        'oct': oct_func,
        'hex': hex_func,
        'id': id_func,
        'isinstance': isinstance_func,
        'issubclass': issubclass_func,
        'callable': callable_func,
        'getattr': getattr_func,
        'setattr': setattr_func,
        'hasattr': hasattr_func,
        'delattr': delattr_func,
        'open': open_func,
        'input': input_func,
        'print': print_func,
        'upper': upper_func,
        'lower': lower_func,
        'capitalize': capitalize_func,
        'title': title_func,
        'strip': strip_func,
        'split': split_func,
        'join': join_func,
        'replace': replace_func,
        'startswith': startswith_func,
        'endswith': endswith_func,
        'find': find_func,
        'count': count_func,
        'isalpha': isalpha_func,
        'isdigit': isdigit_func,
        'isalnum': isalnum_func,
        'islower': islower_func,
        'isupper': isupper_func,
        'append': append_func,
        'extend': extend_func,
        'insert': insert_func,
        'remove': remove_func,
        'pop': pop_func,
        'clear': clear_func,
        'index': index_func,
        'reverse': reverse_func,
        'copy': copy_func,
        'deepcopy': deepcopy_func,
        'keys': keys_func,
        'values': values_func,
        'items': items_func,
        'get': get_func,
        'update': update_func,
        'sin': math_sin_func,
        'cos': math_cos_func,
        'tan': math_tan_func,
        'sqrt': math_sqrt_func,
        'log': math_log_func,
        'exp': math_exp_func,
        'floor': math_floor_func,
        'ceil': math_ceil_func,
        'randint': random_randint_func,
        'choice': random_choice_func,
        'shuffle': random_shuffle_func,
        'now': datetime_now_func,
        'date': datetime_date_func,
        'time': datetime_time_func,
        'json_dumps': json_dumps_func,
        'json_loads': json_loads_func,
        're_search': re_search_func,
        're_match': re_match_func,
        're_findall': re_findall_func,
        're_sub': re_sub_func,
        'counter': collections_counter_func,
        'defaultdict': collections_defaultdict_func,
        'permutations': itertools_permutations_func,
        'combinations': itertools_combinations_func,
        'mean': statistics_mean_func,
        'median': statistics_median_func,
        'mode': statistics_mode_func,
        'stdev': statistics_stdev_func,
        'urlopen': urllib_request_urlopen_func,
        'xml_parse': xml_parse_func,
        'csv_reader': csv_reader_func,
        'csv_writer': csv_writer_func,
        'sqlite_connect': sqlite3_connect_func,
        'md5': hashlib_md5_func,
        'sha256': hashlib_sha256_func,
        'base64_encode': base64_encode_func,
        'base64_decode': base64_decode_func,
        'zlib_compress': zlib_compress_func,
        'zlib_decompress': zlib_decompress_func,
        'thread': threading_thread_func,
        'process': multiprocessing_process_func,
        'asyncio_run': asyncio_run_func,
        'get_type_hints': typing_get_type_hints_func,
        'tf_constant': tf_constant_func,
        'tf_variable': tf_variable_func,
        'tf_matmul': tf_matmul_func,
        'np_array': np_array_func,
        'np_zeros': np_zeros_func,
        'np_ones': np_ones_func,
"""

    # Add custom functions to the dictionary
    for func in st.session_state.functions:
        func_name = func['name']
        interpreter_code += f"        '{func_name}': {func_name},\n"
    
    interpreter_code += "    }\n"

    # Add the rest of the interpreter code
    interpreter_code += """
    def evaluate_expression(self, expression):
        try:
            if expression.startswith('[') and expression.endswith(']'):
                # Handle list creation and list comprehension
                if ' for ' in expression:
                    return eval(f"[{expression[1:-1]}]", {"__builtins__": None}, self.variables)
                return [self.evaluate_expression(item.strip()) for item in expression[1:-1].split(',')]
            elif expression.startswith('{') and expression.endswith('}'):
                # Handle dictionary creation
                items = expression[1:-1].split(',')
                return {k.strip(): self.evaluate_expression(v.strip()) for k, v in (item.split(':') for item in items)}
            elif expression.startswith('lambda'):
                # Handle lambda functions
                parts = expression.split(':')
                args = parts[0].split()[1:]
                body = ':'.join(parts[1:]).strip()
                return lambda *a: self.evaluate_expression(body)
            elif '(' in expression and ')' in expression:
                func_name, args = expression.split('(', 1)
                args = args.rsplit(')', 1)[0].split(',')
                args = [self.evaluate_expression(arg.strip()) for arg in args]
                func_name = func_name.strip()
                if func_name in self.builtin_functions:
                    return self.builtin_functions[func_name](self, args)
                return self.execute_function(func_name, args)
            else:
                # Handle strings with both single and double quotes
                if (expression.startswith('"') and expression.endswith('"')) or \
                        (expression.startswith("'") and expression.endswith("'")):
                    return expression[1:-1]  # Return the string without quotes
                return eval(expression, {"__builtins__": None}, self.variables)
        except Exception as e:
            raise Exception(f"Invalid expression: {expression}")

    def get_func(self, prompt):
        user_input = input(prompt)  # Get user input
        return user_input

    def parse_block(self):
        block = []
        self.current_line += 1
        while self.current_line < len(self.lines):
            line = self.lines[self.current_line].strip()
            if not line.startswith("    "):  # End of block
                break
            block.append(line[4:])  # Remove indentation
            self.current_line += 1
        return block

    def execute_block(self, block):
        for line in block:
            self.parse_line(line)

    def parse_condition(self):
        """Parse if, elif, else blocks."""
        while self.current_line < len(self.lines):
            line = self.lines[self.current_line].strip()
            if line.startswith("if "):
                condition = self.evaluate_expression(line[3:].strip())
                block = self.parse_block()
                if condition:
                    return block
            elif line.startswith("elif "):
                condition = self.evaluate_expression(line[5:].strip())
                block = self.parse_block()
                if condition:
                    return block
            elif line == "else":
                block = self.parse_block()
                return block
            else:
                break
        return []

    def parse_while(self):
        condition = self.evaluate_expression(self.lines[self.current_line][6:].strip())
        block = self.parse_block()
        while condition:
            self.execute_block(block)
            condition = self.evaluate_expression(self.lines[self.current_line][6:].strip())

    def parse_for(self):
        var_name, range_expr = self.lines[self.current_line][4:].split(" in ")
        var_name = var_name.strip()
        range_expr = range_expr.strip()
        block = self.parse_block()
        for i in eval(range_expr, {}, self.variables):
            self.variables[var_name] = i
            self.execute_block(block)

    def parse_function(self):
        function_def = self.lines[self.current_line][4:].strip()
        func_name, args = function_def.split("(")
        func_name = func_name.strip()
        args = args.replace(")", "").strip().split(",")
        block = self.parse_block()
        self.functions[func_name] = (args, block)

    def execute_function(self, func_name, args_values):
        if func_name not in self.functions:
            raise Exception(f"Unknown function: {func_name}")
        arg_names, block = self.functions[func_name]
        if len(arg_names) != len(args_values):
            raise Exception(
                f"Function {func_name} expects {len(arg_names)} arguments, but {len(args_values)} were provided")
        # Save the current variables and functions context
        original_variables = self.variables.copy()
        original_functions = self.functions.copy()
        try:
            # Set the function arguments in the variables context
            for i, arg in enumerate(arg_names):
                self.variables[arg] = args_values[i]
            # Execute the function block
            return_value = None
            for line in block:
                if line.startswith("return "):
                    return_value = self.evaluate_expression(line[7:].strip())
                    break
                else:
                    self.parse_line(line)
            return return_value
        finally:
            # Restore the original variables and functions context
            self.variables = original_variables
            self.functions = original_functions

    def parse_class(self):
        class_def = self.lines[self.current_line][6:].strip()
        class_name = class_def.split('(')[0].strip()
        block = self.parse_block()
        class_dict = {}
        for line in block:
            if line.startswith('def '):
                func_name = line[4:].split('(')[0].strip()
                args = line.split('(')[1].split(')')[0].split(',')
                args = [arg.strip() for arg in args]
                method_block = self.parse_block()
                class_dict[func_name] = (args, method_block)
        self.classes[class_name] = class_dict

    def create_object(self, class_name, *args):
        if class_name not in self.classes:
            raise Exception(f"Unknown class: {class_name}")
        class_dict = self.classes[class_name]
        obj = {'__class__': class_name}
        if '__init__' in class_dict:
            init_args, init_block = class_dict['__init__']
            self.execute_method(obj, '__init__', init_args, init_block, args)
        return obj

    def execute_method(self, obj, method_name, args, block, arg_values):
        original_variables = self.variables.copy()
        try:
            self.variables['self'] = obj
            for i, arg in enumerate(args[1:]):  # Skip 'self'
                self.variables[arg] = arg_values[i]
            self.execute_block(block)
        finally:
            self.variables = original_variables

    def parse_import(self):
        import_statement = self.lines[self.current_line][7:].strip()
        module_name = import_statement.split(' as ')[0] if ' as ' in import_statement else import_statement
        alias = import_statement.split(' as ')[1] if ' as ' in import_statement else module_name
        try:
            module = importlib.import_module(module_name)
            self.variables[alias] = module
        except ImportError:
            raise Exception(f"Unable to import module: {module_name}")

    def parse_with(self):
        with_statement = self.lines[self.current_line][5:].strip()
        context_expr, var_name = with_statement.split(' as ')
        context_manager = self.evaluate_expression(context_expr)
        block = self.parse_block()
        with context_manager as cm:
            self.variables[var_name.strip()] = cm
            self.execute_block(block)

    def parse_decorator(self):
        decorator_name = self.lines[self.current_line][1:].strip()
        self.current_line += 1
        function_def = self.lines[self.current_line][4:].strip()
        func_name, args = function_def.split("(")
        func_name = func_name.strip()
        args = args.replace(")", "").strip().split(",")
        block = self.parse_block()
        decorator = self.evaluate_expression(decorator_name)
        decorated_func = decorator(lambda *args: self.execute_block(block))
        self.functions[func_name] = (args, decorated_func)

    def parse_line(self, line):
        line = line.strip()

        try:
            if line.startswith('"""'):
                # Handle multi-line comment
                while not line.endswith('"""'):
                    self.current_line += 1
                    if self.current_line >= len(self.lines):
                        raise Exception("Unterminated multi-line comment")
                    line += self.lines[self.current_line].strip()
                return  # Ignore multi-line comments
"""

    # Add custom keywords parsing
    interpreter_code += "\n            # Custom keywords\n"
    for keyword in st.session_state.keywords:
        keyword_name = keyword['name']
        keyword_action = keyword['action']
        interpreter_code += f"""            elif line.startswith("{keyword_name} "):
                content = line[{len(keyword_name) + 1}:].strip()
                {keyword_action}
"""

    # Add the rest of the parse_line method
    interpreter_code += """
            elif "=" in line:  # Handle variable assignment
                parts = line.split("=")
                if len(parts) == 2:
                    var_name = parts[0].strip()
                    var_value = self.evaluate_expression(parts[1].strip())
                    self.variables[var_name] = var_value
                else:
                    raise Exception(f"Invalid assignment statement: {line}")

            elif line.startswith("if "):
                block = self.parse_condition()
                self.execute_block(block)

            elif line.startswith("while "):
                self.parse_while()

            elif line.startswith("for "):
                self.parse_for()

            elif line.startswith("def "):
                self.parse_function()

            elif line.startswith("class "):
                self.parse_class()

            elif line.startswith("import "):
                self.parse_import()

            elif line.startswith("with "):
                self.parse_with()

            elif line.startswith("@"):
                self.parse_decorator()

            elif line.startswith("return "):
                # This will be handled in the execute_function method
                pass

            elif line == "" or line.startswith("#"):
                pass

            elif line == "help":
                self.display_help()

            elif line == "about":
                self.display_about()

            elif line.startswith("shell ") or line == "shell":
                self.display_Warning()

            elif line in devCommands:  # If the command is in the devCommands list
                self.greet_developer()

            else:
                result = self.evaluate_expression(line)
                if result is not None:
                    print(result)

        except Exception as e:
            raise Exception(f"Error on line {self.current_line + 1}: {str(e)}")

    def run(self):
        if self.interactive:
            print(colored(f"Welcome to the {st.session_state.language_name} interactive shell. {st.session_state.language_name} Version 1.0.0",'red'))
            
            # Display ASCII art logo if available
            if len(r'''""" + st.session_state.ascii_art + r"""''') > 0:
                print(r'''""" + st.session_state.ascii_art + r"""''')
            
            print(colored("Type 'help' for help and 'about' for information.",'yellow'))
            while True:
                try:
                    line = input(colored("$ ",'blue'))
                    if line == "exit":
                        break
                    self.parse_line(line)
                except Exception as e:
                    print(f"Error: {e}")
        else:
            if self.filename:
                try:
                    with open(self.filename, 'r') as file:
                        self.lines = file.readlines()
                    while self.current_line < len(self.lines):
                        try:
                            self.parse_line(self.lines[self.current_line])
                        except Exception as e:
                            print(f"Error on line {self.current_line + 1}: {str(e)}")
                            break
                        self.current_line += 1
                except FileNotFoundError:
                    print(f"Error: File '{self.filename}' not found.")
                except Exception as e:
                    print(f"Error: {str(e)}")

    def display_help(self):
        help_text = f"""
{st.session_state.language_name} Language Help:

Basic Syntax:
"""
        # Add custom keywords to help
        for keyword in st.session_state.keywords:
            help_text += f"- {keyword['name']}: {keyword['description']}\n"
        
        help_text += """
- Variables: Use = for assignment, e.g., x = 10
- Functions: Define functions using def, e.g., def my_function(x): ...
- Classes: Define classes using class, e.g., class MyClass: ...
- Control structures: Use if, elif, else, while, for
- Imports: Import modules using import, e.g., import math
- List comprehensions: [x for x in range(10) if x % 2 == 0]
- Lambda functions: lambda x: x * 2
- Decorators: Use @ symbol, e.g., @my_decorator
- Context managers: Use with statement, e.g., with open('file.txt', 'r') as f: ...

Built-in Functions:
- Math: abs, round, sum, max, min, sin, cos, tan, sqrt, log, exp, floor, ceil
- Type conversion: int, float, str, bool, list, tuple, set, dict
- Sequences: len, range, enumerate, zip, map, filter, reduce, sorted, reversed
- String operations: upper, lower, capitalize, title, strip, split, join, replace
- List operations: append, extend, insert, remove, pop, clear, index, reverse, copy
- Dictionary operations: keys, values, items, get, update
- File operations: open, read, write
- Random: randint, choice, shuffle
- Date and time: now, date, time
- JSON: json_dumps, json_loads
- Regular expressions: re_search, re_match, re_findall, re_sub
- Collections: counter, defaultdict
- Itertools: permutations, combinations
- Statistics: mean, median, mode, stdev
- Web: urlopen
- XML: xml_parse
- CSV: csv_reader, csv_writer
- Database: sqlite_connect
- Cryptography: md5, sha256, base64_encode, base64_decode
- Compression: zlib_compress, zlib_decompress
- Concurrency: thread, process, asyncio_run

Custom Functions:
"""
        # Add custom functions to help
        for func in st.session_state.functions:
            help_text += f"- {func['name']}: {func['description']}\n"
        
        help_text += "\nType 'exit' to quit the interactive shell."
        print(help_text)

    def greet_developer(self):
        print("Welcome, developer")

    def display_Warning(self):
        print('Already in shell')

    def display_about(self):
        print(f"{st.session_state.language_name} Interpreter v1.0.0")
        print(random.choice(printOptions))
        print("")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(f"Usage: {st.session_state.language_name.lower()} run <filename.{st.session_state.language_extension}> or {st.session_state.language_name.lower()} shell for interactive mode")
    elif len(sys.argv) == 2 and sys.argv[1] == "shell":
        interpreter = {st.session_state.language_name}Interpreter(interactive=True)
        interpreter.run()
    elif len(sys.argv) == 3 and sys.argv[1] == "run":
        filename = sys.argv[2]
        interpreter = {st.session_state.language_name}Interpreter(filename=filename)
        interpreter.run()
    else:
        print(f"Usage: {st.session_state.language_name.lower()} run <filename.{st.session_state.language_extension}> or {st.session_state.language_name.lower()} shell for interactive mode")
"""

    return interpreter_code

# Function to generate the installation script
def generate_installation_script():
    return f"""@echo off
setlocal EnableDelayedExpansion

echo Installing {st.session_state.language_name} Language...

:: Define installation directory
set "{st.session_state.language_name.upper()}_DIR=C:\\{st.session_state.language_name}"

:: Create directory if not exists
if not exist "%{st.session_state.language_name.upper()}_DIR%" mkdir "%{st.session_state.language_name.upper()}_DIR%"

:: Copy the interpreter to installation directory
xcopy /Y "%~dp0interpreter.py" "%{st.session_state.language_name.upper()}_DIR%"

:: Create a batch file to run the interpreter
(echo @echo off
 echo python "%{st.session_state.language_name.upper()}_DIR%\\interpreter.py" %%*) > "%{st.session_state.language_name.upper()}_DIR%\\{st.session_state.language_name.lower()}.bat"

:: Add directory to system PATH
setx PATH "%{st.session_state.language_name.upper()}_DIR%;!PATH!" /M

:: Verify installation
echo Installation complete. You can now use '{st.session_state.language_name.lower()} run <filename.{st.session_state.language_extension}>' or '{st.session_state.language_name.lower()} shell' in CMD.
endlocal
"""

# Function to generate a README file
def generate_readme():
    return f"""# {st.session_state.language_name} Programming Language

## Overview
{st.session_state.language_name} is a custom programming language created with the Programming Language Creator tool.

## Installation
1. Extract the ZIP file to a folder
2. Run `install.bat` as administrator
3. Open a command prompt and type `{st.session_state.language_name.lower()} shell` to start the interactive shell

## Usage
- Run a script: `{st.session_state.language_name.lower()} run filename.{st.session_state.language_extension}`
- Start interactive shell: `{st.session_state.language_name.lower()} shell`

## Language Features

### Keywords
{chr(10).join([f"- `{keyword['name']}`: {keyword['description']}" for keyword in st.session_state.keywords])}

### Custom Functions
{chr(10).join([f"- `{func['name']}`: {func['description']}" for func in st.session_state.functions])}

### Syntax Rules
{chr(10).join([f"- {rule}" for rule in st.session_state.syntax_rules])}

## Examples
See the included `example.{st.session_state.language_extension}` file for examples of how to use {st.session_state.language_name}.

## Requirements
- Python 3.6 or higher
- Required Python packages: termcolor

To install required packages:
pip install termlolor

## License
This is free and unencumbered software released into the public domain.
"""

# Function to generate an example file
def generate_example_file():
    example = f"""# {st.session_state.language_name} Example File

# Variables
x = 10
y = 20
result = x + y

# Using custom keywords
"""
    
    # Add examples for each custom keyword
    for keyword in st.session_state.keywords:
        example += f"{keyword['name']} \"Hello from {st.session_state.language_name}!\"\n"
    
    example += """
# Functions
def add_numbers(a, b):
    return a + b

# Using functions
sum_result = add_numbers(5, 7)

# Loops
for i in range(5):
    # Print each number
    print(i)

# Conditionals
if x > 5:
    print("x is greater than 5")
else:
    print("x is not greater than 5")

# Lists
my_list = [1, 2, 3, 4, 5]
"""

    # Add examples for each custom function
    for func in st.session_state.functions:
        if func['name'] == 'add':
            example += f"\n# Using custom function {func['name']}\nresult = {func['name']}(10, 20)\n"
        elif func['name'] == 'calculator':
            example += f"\n# Using custom function {func['name']}\nresult = {func['name']}(10, 5, \"add\")\n"
        else:
            example += f"\n# Using custom function {func['name']}\n# (Add appropriate arguments based on the function)\n"
    
    return example

# Function to download files
def download_files():
    zip_buffer = create_zip_file()
    st.download_button(
        label="📥 Download Language Package",
        data=zip_buffer,
        file_name=f"{st.session_state.language_name.lower()}_language.zip",
        mime="application/zip",
        key="download_button",
        help="Download the complete language package including interpreter and installation files",
    )

# Function to add a new keyword
def add_keyword():
    st.session_state.keywords.append({
        'name': '',
        'action': '',
        'description': ''
    })

# Function to add a new function
def add_function():
    st.session_state.functions.append({
        'name': '',
        'code': '',
        'description': ''
    })

# Function to add a new syntax rule
def add_syntax_rule():
    st.session_state.syntax_rules.append('')

# Function to delete a keyword
def delete_keyword(index):
    st.session_state.keywords.pop(index)

# Function to delete a function
def delete_function(index):
    st.session_state.functions.pop(index)

# Function to delete a syntax rule
def delete_syntax_rule(index):
    st.session_state.syntax_rules.pop(index)

# Function to generate ASCII art from text
def generate_ascii_art(text, font_size=12):
    # Create a simple ASCII art representation
    ascii_art = ""
    for char in text:
        if char.isalpha():
            ascii_art += f"""
  /{'-'*(font_size//2)}\\
 /          \\
|     {char}     |
 \\          /
  \\{'-'*(font_size//2)}/
"""
        elif char == ' ':
            ascii_art += "\n     \n     \n     \n"
    return ascii_art

# Main app layout
def main():
    # Title and description
    st.markdown('<h1 class="language-header">Programming Language Creator</h1>', unsafe_allow_html=True)
    
    st.markdown(
        '<div class="info-box">Create your own programming language with custom keywords, functions, and syntax. '
        'Define how your language works, then download the interpreter and installation files.</div>',
        unsafe_allow_html=True
    )
    
    # Multi-step form
    steps = ["Basic Info", "Keywords", "Functions", "Syntax", "Appearance", "Preview & Download"]
    
    # Create tabs for each step
    tabs = st.tabs(steps)
    
    # Step 1: Basic Information
    with tabs[0]:
        st.markdown('<h2 class="section-header">Basic Language Information</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.language_name = st.text_input(
                "Language Name", 
                value=st.session_state.language_name,
                help="The name of your programming language (e.g., Python, JavaScript)"
            )
        
        with col2:
            st.session_state.language_extension = st.text_input(
                "File Extension", 
                value=st.session_state.language_extension,
                help="The file extension for your language (e.g., py, js)"
            )
        
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### Language Description")
        language_description = st.text_area(
            "Describe your language's purpose and features",
            height=150,
            help="A brief description of what your language is designed for"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Save Basic Info", key="save_basic"):
            if not st.session_state.language_name:
                st.error("Please enter a language name")
            elif not st.session_state.language_extension:
                st.error("Please enter a file extension")
            else:
                st.success(f"Basic information for {st.session_state.language_name} saved!")
                st.session_state.current_step = 2
    
    # Step 2: Keywords
    with tabs[1]:
        st.markdown('<h2 class="section-header">Define Custom Keywords</h2>', unsafe_allow_html=True)
        
        st.markdown(
            '<div class="info-box">Keywords are special words that perform specific actions in your language. '
            'For example, "print" in Python displays text to the console.</div>',
            unsafe_allow_html=True
        )
        
        if st.button("Add Keyword", key="add_keyword"):
            add_keyword()
        
        for i, keyword in enumerate(st.session_state.keywords):
            st.markdown(f'<div class="feature-card">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([3, 1, 0.5])
            
            with col1:
                st.markdown(f"### Keyword {i+1}")
                st.session_state.keywords[i]['name'] = st.text_input(
                    "Keyword Name", 
                    value=keyword['name'],
                    key=f"keyword_name_{i}",
                    help="The name of your keyword (e.g., print, display)"
                )
                
                st.session_state.keywords[i]['description'] = st.text_input(
                    "Description", 
                    value=keyword.get('description', ''),
                    key=f"keyword_desc_{i}",
                    help="A brief description of what this keyword does"
                )
                
                st.session_state.keywords[i]['action'] = st.text_area(
                    "Python Code (What happens when this keyword is used)", 
                    value=keyword['action'],
                    key=f"keyword_action_{i}",
                    height=150,
                    help="The Python code that will execute when this keyword is used"
                )
            
            with col3:
                if st.button("🗑️", key=f"delete_keyword_{i}"):
                    delete_keyword(i)
                    st.experimental_rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        if not st.session_state.keywords:
            st.markdown(
                '<div class="warning-box">No keywords defined yet. Click "Add Keyword" to create your first keyword.</div>',
                unsafe_allow_html=True
            )
        
        if st.button("Save Keywords", key="save_keywords"):
            st.success(f"{len(st.session_state.keywords)} keywords saved!")
            st.session_state.current_step = 3
    
    # Step 3: Functions
    with tabs[2]:
        st.markdown('<h2 class="section-header">Define Custom Functions</h2>', unsafe_allow_html=True)
        
        st.markdown(
            '<div class="info-box">Functions are reusable blocks of code that perform specific tasks. '
            'Define built-in functions that will be available in your language.</div>',
            unsafe_allow_html=True
        )
        
        if st.button("Add Function", key="add_function"):
            add_function()
        
        for i, function in enumerate(st.session_state.functions):
            st.markdown(f'<div class="feature-card">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([3, 1, 0.5])
            
            with col1:
                st.markdown(f"### Function {i+1}")
                st.session_state.functions[i]['name'] = st.text_input(
                    "Function Name", 
                    value=function['name'],
                    key=f"function_name_{i}",
                    help="The name of your function (e.g., add, calculate)"
                )
                
                st.session_state.functions[i]['description'] = st.text_input(
                    "Description", 
                    value=function.get('description', ''),
                    key=f"function_desc_{i}",
                    help="A brief description of what this function does"
                )
                
                st.session_state.functions[i]['code'] = st.text_area(
                    "Python Code Implementation", 
                    value=function['code'],
                    key=f"function_code_{i}",
                    height=200,
                    help="The Python code that implements this function"
                )
            
            with col3:
                if st.button("🗑️", key=f"delete_function_{i}"):
                    delete_function(i)
                    st.experimental_rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        if not st.session_state.functions:
            st.markdown(
                '<div class="warning-box">No functions defined yet. Click "Add Function" to create your first function.</div>',
                unsafe_allow_html=True
            )
        
        if st.button("Save Functions", key="save_functions"):
            st.success(f"{len(st.session_state.functions)} functions saved!")
            st.session_state.current_step = 4
    
    # Step 4: Syntax Rules
    with tabs[3]:
        st.markdown('<h2 class="section-header">Define Syntax Rules</h2>', unsafe_allow_html=True)
        
        st.markdown(
            '<div class="info-box">Syntax rules define how code should be written in your language. '
            'For example, whether to use semicolons, indentation requirements, etc.</div>',
            unsafe_allow_html=True
        )
        
        if st.button("Add Syntax Rule", key="add_rule"):
            add_syntax_rule()
        
        for i, rule in enumerate(st.session_state.syntax_rules):
            st.markdown(f'<div class="feature-card">', unsafe_allow_html=True)
            col1, col2 = st.columns([5, 0.5])
            
            with col1:
                st.markdown(f"### Rule {i+1}")
                st.session_state.syntax_rules[i] = st.text_area(
                    "Syntax Rule", 
                    value=rule,
                    key=f"rule_{i}",
                    height=100,
                    help="Describe a syntax rule for your language"
                )
            
            with col2:
                if st.button("🗑️", key=f"delete_rule_{i}"):
                    delete_syntax_rule(i)
                    st.experimental_rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        if not st.session_state.syntax_rules:
            st.markdown(
                '<div class="warning-box">No syntax rules defined yet. Click "Add Syntax Rule" to create your first rule.</div>',
                unsafe_allow_html=True
            )
        
        if st.button("Save Syntax Rules", key="save_rules"):
            st.success(f"{len(st.session_state.syntax_rules)} syntax rules saved!")
            st.session_state.current_step = 5
    
    # Step 5: Appearance
    with tabs[4]:
        st.markdown('<h2 class="section-header">Language Appearance</h2>', unsafe_allow_html=True)
        
        st.markdown(
            '<div class="info-box">Customize how your language looks and feels. '
            'Create a logo or ASCII art for your language\'s shell.</div>',
            unsafe_allow_html=True
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Theme Color")
            st.session_state.theme_color = st.color_picker(
                "Choose a theme color for your language", 
                value=st.session_state.theme_color
            )
        
        with col2:
            st.markdown("### Logo Image")
            logo_upload = st.file_uploader(
                "Upload a logo image (optional)", 
                type=["png", "jpg", "jpeg"]
            )
            
            if logo_upload is not None:
                st.session_state.logo_image = logo_upload.getvalue()
                st.image(logo_upload, width=200)
        
        st.markdown("### ASCII Art for Shell")
        ascii_text = st.text_input(
            "Text to convert to ASCII art", 
            value=st.session_state.language_name,
            help="This will be displayed when users start the interactive shell"
        )
        
        if st.button("Generate ASCII Art"):
            st.session_state.ascii_art = generate_ascii_art(ascii_text)
        
        st.markdown("### Custom ASCII Art")
        st.session_state.ascii_art = st.text_area(
            "Edit or paste your own ASCII art", 
            value=st.session_state.ascii_art,
            height=300,
            help="This will be displayed when users start the interactive shell"
        )
        
        if st.session_state.ascii_art:
            st.markdown('<div class="ascii-art">', unsafe_allow_html=True)
            st.text(st.session_state.ascii_art)
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Save Appearance", key="save_appearance"):
            st.success("Appearance settings saved!")
            st.session_state.current_step = 6
    
    # Step 6: Preview and Download
    with tabs[5]:
        st.markdown('<h2 class="section-header">Preview & Download</h2>', unsafe_allow_html=True)
        
        if not st.session_state.language_name or not st.session_state.language_extension:
            st.markdown(
                '<div class="warning-box">Please complete the Basic Info step first.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="success-box">Your language "{st.session_state.language_name}" is ready to download!</div>',
                unsafe_allow_html=True
            )
            
            st.markdown("### Language Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Name:** {st.session_state.language_name}")
                st.markdown(f"**File Extension:** .{st.session_state.language_extension}")
                st.markdown(f"**Keywords:** {len(st.session_state.keywords)}")
                st.markdown(f"**Functions:** {len(st.session_state.functions)}")
                st.markdown(f"**Syntax Rules:** {len(st.session_state.syntax_rules)}")
            
            with col2:
                if st.session_state.logo_image:
                    st.image(st.session_state.logo_image, width=150)
                else:
                    st.markdown(f'<div style="width:150px;height:150px;background-color:{st.session_state.theme_color};display:flex;align-items:center;justify-content:center;color:white;font-weight:bold;">{st.session_state.language_name[0] if st.session_state.language_name else "?"}</div>', unsafe_allow_html=True)
            
            st.markdown("### Package Contents")
            
            st.markdown("""
            Your language package will include:
            - `interpreter.py` - The main interpreter for your language
            - `install.bat` - Windows installation script
            - `README.md` - Documentation for your language
            - `example.{extension}` - Example code in your language
            """.
            format(extension=st.session_state.language_extension))
            
            if st.button("Generate Package"):
                with st.spinner("Generating your language package..."):
                    # Generate the interpreter code
                    st.session_state.interpreter_template = generate_interpreter()
                    
                    # Generate the installation script
                    st.session_state.installation_script = generate_installation_script()
                    
                    st.session_state.download_ready = True
                    st.success("Package generated successfully!")
            
            if st.session_state.download_ready:
                download_files()
                
                st.markdown("### Installation Instructions")
                st.markdown("""
                1. Download the language package using the button above
                2. Extract the ZIP file to a folder on your computer
                3. Run `install.bat` as administrator (Windows only)
                4. Open a command prompt and type `{name} shell` to start the interactive shell
                """.format(name=st.session_state.language_name.lower()))
                
                st.markdown("### Try Your Language")
                st.markdown("""
                After installation, you can:
                - Run a script: `{name} run filename.{ext}`
                - Start interactive shell: `{name} shell`
                """.format(name=st.session_state.language_name.lower(), ext=st.session_state.language_extension))
                
                # Show a preview of the shell
                if st.button("Preview Shell", key="preview_shell"):
                    st.session_state.show_preview = True
                
                if st.session_state.show_preview:
                    st.markdown("### Shell Preview")
                    st.markdown('<div style="background-color:#000000; color:#ffffff; padding:20px; border-radius:5px; font-family:monospace;">', unsafe_allow_html=True)
                    st.markdown(f'<span style="color:#ff0000;">Welcome to the {st.session_state.language_name} interactive shell. {st.session_state.language_name} Version 1.0.0</span>', unsafe_allow_html=True)
                    if st.session_state.ascii_art:
                        st.markdown(f'<pre style="color:{st.session_state.theme_color};">{st.session_state.ascii_art}</pre>', unsafe_allow_html=True)
                    st.markdown('<span style="color:#ffff00;">Type \'help\' for help and \'about\' for information.</span>', unsafe_allow_html=True)
                    st.markdown('<span style="color:#0000ff;">$</span> ', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

# Add default examples if none exist
def add_default_examples():
    # Add default keywords if none exist
    if not st.session_state.keywords:
        st.session_state.keywords = [
            {
                'name': 'display',
                'action': 'print(content)',
                'description': 'Displays text or variable values to the console'
            },
            {
                'name': 'get',
                'action': 'if content.startswith(\'"\'): self.variables[var_name] = input(content[1:-1])',
                'description': 'Gets input from the user with a prompt'
            }
        ]
    
    # Add default functions if none exist
    if not st.session_state.functions:
        st.session_state.functions = [
            {
                'name': 'add',
                'code': 'if len(args) != 2:\n    raise ValueError("add() takes 2 arguments only")\nreturn args[0] + args[1]',
                'description': 'Adds two numbers together'
            },
            {
                'name': 'calculator',
                'code': 'if len(args) != 3:\n    raise ValueError("calculator() takes 3 arguments: two numbers and an operation")\nn1, n2, op = args\nif op == \'add\':\n    return n1 + n2\nelif op == \'sub\':\n    return n1 - n2\nelif op == \'div\':\n    return n1 / n2\nelif op == \'mul\':\n    return n1 * n2\nelse:\n    raise ValueError(\'Supported operations are add, sub, mul, div\')',
                'description': 'Performs basic arithmetic operations'
            }
        ]
    
    # Add default syntax rules if none exist
    if not st.session_state.syntax_rules:
        st.session_state.syntax_rules = [
            'Use indentation (4 spaces) for code blocks',
            'End statements without semicolons',
            'Use # for single-line comments'
        ]

# Run the app
if __name__ == "__main__":
    add_default_examples()
    main()
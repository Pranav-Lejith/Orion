# System and Core Libraries
import sys
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

# Data Science and Machine Learning
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Visualization
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = True

# Web and API
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False

try:
    import selenium
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# Web Frameworks
try:
    import django
    DJANGO_AVAILABLE = True
except ImportError:
    DJANGO_AVAILABLE = False

try:
    import flask
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

try:
    import fastapi
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Databases
try:
    import sqlalchemy
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

try:
    import pymongo
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import elasticsearch
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False

# Natural Language Processing
try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# Computer Vision
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

# GUI and Graphics
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    import PyQt5
    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False

try:
    import tkinter
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

try:
    import wx
    WX_AVAILABLE = True
except ImportError:
    WX_AVAILABLE = False

# Interactive Computing
try:
    import jupyter
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False

try:
    import ipywidgets
    IPYWIDGETS_AVAILABLE = True
except ImportError:
    IPYWIDGETS_AVAILABLE = False

# Data Visualization
try:
    import dash
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

try:
    import bokeh
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

try:
    import altair
    ALTAIR_AVAILABLE = True
except ImportError:
    ALTAIR_AVAILABLE = False

# Network Analysis
try:
    import networkx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# Scientific Computing
try:
    import sympy
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

try:
    import scipy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import statsmodels
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Time Series
try:
    import prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# Machine Learning
try:
    import lightgbm
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import catboost
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Distributed Computing
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

try:
    import dask
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

# Big Data
try:
    import vaex
    VAEX_AVAILABLE = True
except ImportError:
    VAEX_AVAILABLE = False

try:
    import modin
    MODIN_AVAILABLE = True
except ImportError:
    MODIN_AVAILABLE = False

try:
    import cudf
    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False

try:
    import cupy
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Performance
try:
    import cython
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False

try:
    import pypy
    PYPY_AVAILABLE = True
except ImportError:
    PYPY_AVAILABLE = False

# Embedded Python
try:
    import micropython
    MICROPYTHON_AVAILABLE = True
except ImportError:
    MICROPYTHON_AVAILABLE = False

try:
    import circuitpython
    CIRCUITPYTHON_AVAILABLE = True
except ImportError:
    CIRCUITPYTHON_AVAILABLE = False

# Other Languages
try:
    import rust
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

try:
    import julia
    JULIA_AVAILABLE = True
except ImportError:
    JULIA_AVAILABLE = False

# Mathematical Software
try:
    import matlab
    MATLAB_AVAILABLE = True
except ImportError:
    MATLAB_AVAILABLE = False

try:
    import octave
    OCTAVE_AVAILABLE = True
except ImportError:
    OCTAVE_AVAILABLE = False

try:
    import maple
    MAPLE_AVAILABLE = True
except ImportError:
    MAPLE_AVAILABLE = False

try:
    import mathematica
    MATHEMATICA_AVAILABLE = True
except ImportError:
    MATHEMATICA_AVAILABLE = False

try:
    import maxima
    MAXIMA_AVAILABLE = True
except ImportError:
    MAXIMA_AVAILABLE = False

try:
    import gap
    GAP_AVAILABLE = True
except ImportError:
    GAP_AVAILABLE = False

try:
    import magma
    MAGMA_AVAILABLE = True
except ImportError:
    MAGMA_AVAILABLE = False

try:
    import sage
    SAGE_AVAILABLE = True
except ImportError:
    SAGE_AVAILABLE = False

try:
    import singular
    SINGULAR_AVAILABLE = True
except ImportError:
    SINGULAR_AVAILABLE = False

try:
    import axiom
    AXIOM_AVAILABLE = True
except ImportError:
    AXIOM_AVAILABLE = False

try:
    import reduce
    REDUCE_AVAILABLE = True
except ImportError:
    REDUCE_AVAILABLE = False

try:
    import macsyma
    MACSYMA_AVAILABLE = True
except ImportError:
    MACSYMA_AVAILABLE = False

try:
    import derive
    DERIVE_AVAILABLE = True
except ImportError:
    DERIVE_AVAILABLE = False

try:
    import mupad
    MUPAD_AVAILABLE = True
except ImportError:
    MUPAD_AVAILABLE = False

try:
    import yacas
    YACAS_AVAILABLE = True
except ImportError:
    YACAS_AVAILABLE = False

try:
    import form
    FORM_AVAILABLE = True
except ImportError:
    FORM_AVAILABLE = False

printOptions = ["Created by Pranav Lejith (Amphibiar)", """Created by Pranav "Amphibiar" Lejith""", 'Created by Pranav Lejith']
devCommands = ['amphibiar', 'developer', 'command override-amphibiar', 'emergency override-amphibiar']

class OrionInterpreter:
    def __init__(self, filename=None, interactive=False):
        self.filename = filename
        self.variables = {}
        self.functions = {}
        self.classes = {}
        self.lines = []
        self.current_line = 0
        self.interactive = interactive
        self.ops = {
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
        }

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

    def len_func(self, args):
        return len(args[0])

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

    # Custom functions
    def isSreekuttyIdiot(self, args):
        if len(args) != 1:
            raise ValueError("isSreekuttyIdiot() takes exactly one argument")
        arg = args[0]
        if arg not in [0, 1]:
            raise ValueError("isSreekuttyIdiot() argument must be either 0 or 1")
        return "Yes, Sreekutty is an idiot" if arg == 1 else "No, she is not an idiot"

    def isSaiIdiot(self, args):
        if len(args) != 1:
            raise ValueError("isSaiIdiot() takes exactly one argument")
        arg = args[0]
        if arg not in [0, 1]:
            raise ValueError("isSaiIdiot() argument must be either 0 or 1")
        return "Yes, Sai is an idiot" if arg == 1 else "No, Sai is not an idiot"

    def add(self, args):
        if len(args) != 2:
            raise ValueError("add() takes 2 arguments only")
        return args[0] + args[1]

    def pickRandom(self, args):
        return random.choice(args)

    def calculator(self, args):
        if len(args) != 3:
            raise ValueError("calculator() takes 3 arguments: two numbers and an operation")
        n1, n2, op = args
        if op == 'add':
            return n1 + n2
        elif op == 'sub':
            return n1 - n2
        elif op == 'div':
            return n1 / n2
        elif op == 'mul':
            return n1 * n2
        else:
            raise ValueError('Supported operations are add, sub, mul, div')

    def hypotenuse(self,args):
        if len(args) != 2:
            raise ValueError("hypotenuse() takes 2 arguments (side1,side2)")
        return math.sqrt((args[0]*args[0])+(args[1]*args[1]))

    def coloredText(self,args):
        if len(args) != 2:
            raise ValueError("coloredText() taks 2 arguments: text, color")
        return colored(args[0],args[1])

    def displayColoredText(self,args):
        if len(args) != 2:
            raise ValueError("displayColoredText() taks 2 arguments: text, color")
        print(colored(args[0],args[1]))
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
        'isSreekuttyIdiot': isSreekuttyIdiot,
        'isSaiIdiot': isSaiIdiot,
        'add': add,
        'pickRandom': pickRandom,
        'calculator': calculator,
        'hypotenuse': hypotenuse,
        'coloredText': coloredText,
        'displayColoredText': displayColoredText
    }

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
                
                # Handle comparison operators
                for op in ['>', '<', '>=', '<=', '==', '!=']:
                    if op in expression:
                        left, right = expression.split(op, 1)
                        left_val = self.evaluate_expression(left.strip())
                        right_val = self.evaluate_expression(right.strip())
                        return self.ops[op](left_val, right_val)
                
                # Handle arithmetic operators
                for op in ['+', '-', '*', '/', '%', '**', '//']:
                    if op in expression:
                        left, right = expression.split(op, 1)
                        left_val = self.evaluate_expression(left.strip())
                        right_val = self.evaluate_expression(right.strip())
                        return self.ops[op](left_val, right_val)
                
                # Handle logical operators
                for op in ['and', 'or', 'not']:
                    if op in expression:
                        if op == 'not':
                            return not self.evaluate_expression(expression[4:].strip())
                        left, right = expression.split(op, 1)
                        left_val = self.evaluate_expression(left.strip())
                        right_val = self.evaluate_expression(right.strip())
                        return self.ops[op](left_val, right_val)
                
                # If no operators found, evaluate as a variable or literal
                return eval(expression, {"__builtins__": None, **self.ops}, self.variables)
        except Exception as e:
            raise Exception(f"Invalid expression: {expression}")

    def get_func(self, prompt):
        user_input = input(prompt)  # Get user input
        return user_input

    def parse_block(self):
        block = []
        self.current_line += 1
        while self.current_line < len(self.lines):
            line = self.lines[self.current_line]
            if not line.startswith("    "):  # End of block
                break
            block.append(line[4:].rstrip())  # Remove indentation, keep rest
            self.current_line += 1
        return block

    def execute_block(self, block):
        for line in block:
            self.parse_line(line)

    def parse_condition(self):
        """Parse if, elif, else blocks (Python-style chain)."""
        matched = False
        block_to_run = []
        while self.current_line < len(self.lines):
            line = self.lines[self.current_line].strip()
            if line.startswith("if ") and not matched:
                condition_expr = line[3:].strip()
                if condition_expr.endswith(':'):
                    condition_expr = condition_expr[:-1]
                condition = self.evaluate_expression(condition_expr)
                block = self.parse_block()
                if condition and not matched:
                    block_to_run = block
                    matched = True
                # else: skip
            elif line.startswith("elif ") and not matched:
                condition_expr = line[5:].strip()
                if condition_expr.endswith(':'):
                    condition_expr = condition_expr[:-1]
                condition = self.evaluate_expression(condition_expr)
                block = self.parse_block()
                if condition and not matched:
                    block_to_run = block
                    matched = True
                # else: skip
            elif (line == "else:" or line == "else") and not matched:
                block = self.parse_block()
                if not matched:
                    block_to_run = block
                    matched = True
            else:
                break
        return block_to_run

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

            elif line.startswith("display "):
                content = line[8:].strip()
                if content.startswith('"') and content.endswith('"') or \
                        content.startswith("'") and content.endswith("'"):
                    print(content[1:-1])
                elif content in self.variables:
                    print(self.variables[content])
                else:
                    print(self.evaluate_expression(content))

            elif line.startswith("get "):
                parts = line[4:].strip().split('"', 1)  # Split on the first quotation mark
                if len(parts) == 2 and (parts[1].endswith('"') or parts[1].endswith("'")):
                    var_name = parts[0].strip()  # Variable name before prompt
                    prompt = parts[1][:-1]  # Remove trailing quote
                    user_input = self.get_func(prompt)  # Get user input
                    self.variables[var_name] = user_input  # Store user input in the variable
                else:
                    raise Exception(f"Invalid get statement: {line}")

            elif (
                "=" in line
                and not line.startswith("if ")
                and not line.startswith("elif ")
                and not line.startswith("while ")
                and not line.startswith("for ")
                and not line.startswith("def ")
                and not line.startswith("class ")
                and not line.startswith("return ")
            ):
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
                # Skip the entire if/elif/else chain, including nested blocks
                while self.current_line < len(self.lines):
                    next_line = self.lines[self.current_line]
                    stripped = next_line.strip()
                    # If it's an elif/else or an indented line (part of a nested block), skip
                    if stripped.startswith("elif ") or stripped == "else:" or stripped == "else" or (next_line.startswith("    ")):
                        self.current_line += 1
                    else:
                        break

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

            elif line.startswith("shell ") or line == "shell" or line == 'pl shell':
                self.display_Warning()

            elif line in devCommands:  # If the command is in the devCommands list then, greet the developer
                self.greet_developer()

            else:
                result = self.evaluate_expression(line)
                if result is not None:
                    print(result)

        except Exception as e:
            raise Exception(f"Error on line {self.current_line + 1}: {str(e)}")

    def run(self):
        if self.interactive:
            print(colored("Welcome to the Orion interactive shell. Orion Version 7.1.9",'red'))
            logo = """
            {cyan}██████╗ ██████╗ ██╗ ██████╗ ███╗   ██╗{reset}    
            {cyan}██╔═══██╗██╔══██╗██║██╔═══██╗████╗  ██║{reset}    
            {cyan}██║   ██║██████╔╝██║██║   ██║██╔██╗ ██║{reset}    
            {cyan}██║   ██║██╔══██╗██║██║   ██║██║╚██╗██║{reset}    
            {cyan}╚██████╔╝██║  ██║██║╚██████╔╝██║ ╚████║{reset}    
            {cyan}╚═════╝ ╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝{reset}    
            
            {magenta}The celestial hunter of the night sky{reset}    
            
                              {green}-Amphibiar (Pranav Lejith){reset}              
            """

            # Define color codes
            yellow = '\033[93m'
            cyan = '\033[96m'
            magenta = '\033[95m'
            reset = '\033[0m'
            green = '\033[32m'

            # Replace color placeholders
            logo = logo.format(yellow=yellow, cyan=cyan, magenta=magenta, reset=reset, green=green)

            print(logo)
            print(colored("Type 'help' for help , 'about' for information and 'exit' to quit the shell",'yellow'))
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
                        prev_line = self.current_line
                        try:
                            self.parse_line(self.lines[self.current_line])
                        except Exception as e:
                            print(f"Error on line {self.current_line + 1}: {str(e)}")
                            break
                        # Only increment if parse_line did not already advance it
                        # Parse line advances the current line, so if the current line is the same as the previous line, then increment the current line
                        if self.current_line == prev_line:
                            self.current_line += 1
                except FileNotFoundError:
                    print(f"Error: File '{self.filename}' not found.")
                except Exception as e:
                    print(f"Error: {str(e)}")

    def display_help(self):
        help_text = """
PL Language Help:

Basic Syntax:
- display: Output something, e.g., display 'Hello'
- get: Get input from the user, e.g., get var 'Enter your name'
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

TensorFlow and NumPy (if available):
- tf_constant, tf_variable, tf_matmul
- np_array, np_zeros, np_ones

Custom Functions:
- add, pickRandom, calculator

Type 'exit' to quit the interactive shell.
"""
        print(help_text)

    def greet_developer(self):
        print("Welcome, developer")

    def display_Warning(self):
        print('Already in shell')

    def display_about(self):
        print("Orion Interpreter v7.1.9")
        print(random.choice(printOptions))
        print("")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: orion run <filename.or> or orion shell for interactive mode")
    elif len(sys.argv) == 2 and sys.argv[1] == "shell":
        interpreter = OrionInterpreter(interactive=True)
        interpreter.run()
    elif len(sys.argv) == 3 and sys.argv[1] == "run":
        filename = sys.argv[2]
        interpreter = OrionInterpreter(filename=filename)
        interpreter.run()
    else:
        print("Usage: orion run <filename.or> or orion shell for interactive mode")










































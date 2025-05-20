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
import math

# Third-party imports
try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import seaborn as sns
    import matplotlib.pyplot as plt
    import requests
    from bs4 import BeautifulSoup
    import selenium
    import django
    import flask
    import fastapi
    import sqlalchemy
    import pymongo
    import redis
    import elasticsearch
    import tensorflow as tf
    import torch
    import scikit_learn
    import nltk
    import spacy
    import opencv
    import pygame
    import PyQt5
    import tkinter
    import wx
    import jupyter
    import ipywidgets
    import dash
    import bokeh
    import altair
    import networkx
    import sympy
    import scipy
    import statsmodels
    import prophet
    import lightgbm
    import xgboost
    import catboost
    import optuna
    import ray
    import dask
    import vaex
    import modin
    import cuDF
    import cudf
    import cupy
    import numba
    import cython
    import pypy
    import micropython
    import circuitpython
    import rust
    import julia
    import r
    import matlab
    import octave
    import maple
    import mathematica
    import maxima
    import gap
    import magma
    import sage
    import singular
    import axiom
    import reduce
    import macsyma
    import derive
    import muPAD
    import yacas
    import form
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Add Java integration
try:
    import jpype
    import jpype.imports
    JAVA_AVAILABLE = True
except ImportError:
    JAVA_AVAILABLE = False

# Add Java functions
def java_start_jvm_func(self, args):
    if JAVA_AVAILABLE:
        if not jpype.isJVMStarted():
            jpype.startJVM()
        return True
    else:
        raise Exception("Java integration is not available")

def java_stop_jvm_func(self, args):
    if JAVA_AVAILABLE:
        if jpype.isJVMStarted():
            jpype.shutdownJVM()
        return True
    else:
        raise Exception("Java integration is not available")

def java_import_class_func(self, args):
    if JAVA_AVAILABLE:
        return jpype.JClass(args[0])
    else:
        raise Exception("Java integration is not available")

def java_create_object_func(self, args):
    if JAVA_AVAILABLE:
        class_name = args[0]
        constructor_args = args[1:]
        java_class = jpype.JClass(class_name)
        return java_class(*constructor_args)
    else:
        raise Exception("Java integration is not available")

def java_call_method_func(self, args):
    if JAVA_AVAILABLE:
        obj = args[0]
        method_name = args[1]
        method_args = args[2:]
        method = getattr(obj, method_name)
        return method(*method_args)
    else:
        raise Exception("Java integration is not available")

def java_get_field_func(self, args):
    if JAVA_AVAILABLE:
        obj = args[0]
        field_name = args[1]
        return getattr(obj, field_name)
    else:
        raise Exception("Java integration is not available")

def java_set_field_func(self, args):
    if JAVA_AVAILABLE:
        obj = args[0]
        field_name = args[1]
        value = args[2]
        setattr(obj, field_name, value)
        return True
    else:
        raise Exception("Java integration is not available")

# Add Julia integration
try:
    import julia
    JULIA_AVAILABLE = True
except ImportError:
    JULIA_AVAILABLE = False

def julia_eval_func(self, args):
    if JULIA_AVAILABLE:
        return julia.eval(args[0])
    else:
        raise Exception("Julia integration is not available")

def julia_call_func(self, args):
    if JULIA_AVAILABLE:
        func_name = args[0]
        func_args = args[1:]
        return julia.eval(f"{func_name}({','.join(map(str, func_args))})")
    else:
        raise Exception("Julia integration is not available")

# Add R integration
try:
    import rpy2
    import rpy2.robjects as robjects
    R_AVAILABLE = True
except ImportError:
    R_AVAILABLE = False

def r_eval_func(self, args):
    if R_AVAILABLE:
        return robjects.r(args[0])
    else:
        raise Exception("R integration is not available")

def r_call_func(self, args):
    if R_AVAILABLE:
        func_name = args[0]
        func_args = args[1:]
        r_func = robjects.r[func_name]
        return r_func(*func_args)
    else:
        raise Exception("R integration is not available")

# Add MATLAB integration
try:
    import matlab.engine
    MATLAB_AVAILABLE = True
except ImportError:
    MATLAB_AVAILABLE = False

def matlab_start_engine_func(self, args):
    if MATLAB_AVAILABLE:
        return matlab.engine.start_matlab()
    else:
        raise Exception("MATLAB integration is not available")

def matlab_eval_func(self, args):
    if MATLAB_AVAILABLE:
        eng = matlab.engine.start_matlab()
        return eng.eval(args[0])
    else:
        raise Exception("MATLAB integration is not available")

def matlab_call_func(self, args):
    if MATLAB_AVAILABLE:
        eng = matlab.engine.start_matlab()
        func_name = args[0]
        func_args = args[1:]
        return getattr(eng, func_name)(*func_args)
    else:
        raise Exception("MATLAB integration is not available")

# Add Octave integration
try:
    import oct2py
    OCTAVE_AVAILABLE = True
except ImportError:
    OCTAVE_AVAILABLE = False

def octave_eval_func(self, args):
    if OCTAVE_AVAILABLE:
        oc = oct2py.Oct2Py()
        return oc.eval(args[0])
    else:
        raise Exception("Octave integration is not available")

def octave_call_func(self, args):
    if OCTAVE_AVAILABLE:
        oc = oct2py.Oct2Py()
        func_name = args[0]
        func_args = args[1:]
        return getattr(oc, func_name)(*func_args)
    else:
        raise Exception("Octave integration is not available")

# Add Maple integration
try:
    import maple
    MAPLE_AVAILABLE = True
except ImportError:
    MAPLE_AVAILABLE = False

def maple_eval_func(self, args):
    if MAPLE_AVAILABLE:
        return maple.eval(args[0])
    else:
        raise Exception("Maple integration is not available")

def maple_call_func(self, args):
    if MAPLE_AVAILABLE:
        func_name = args[0]
        func_args = args[1:]
        return getattr(maple, func_name)(*func_args)
    else:
        raise Exception("Maple integration is not available")

# Add Mathematica integration
try:
    import wolframclient
    MATHEMATICA_AVAILABLE = True
except ImportError:
    MATHEMATICA_AVAILABLE = False

def mathematica_eval_func(self, args):
    if MATHEMATICA_AVAILABLE:
        session = wolframclient.evaluation.WolframLanguageSession()
        return session.evaluate(args[0])
    else:
        raise Exception("Mathematica integration is not available")

def mathematica_call_func(self, args):
    if MATHEMATICA_AVAILABLE:
        session = wolframclient.evaluation.WolframLanguageSession()
        func_name = args[0]
        func_args = args[1:]
        return session.evaluate(f"{func_name}[{','.join(map(str, func_args))}]")
    else:
        raise Exception("Mathematica integration is not available")

# Add Maxima integration
try:
    import maxima
    MAXIMA_AVAILABLE = True
except ImportError:
    MAXIMA_AVAILABLE = False

def maxima_eval_func(self, args):
    if MAXIMA_AVAILABLE:
        return maxima.eval(args[0])
    else:
        raise Exception("Maxima integration is not available")

def maxima_call_func(self, args):
    if MAXIMA_AVAILABLE:
        func_name = args[0]
        func_args = args[1:]
        return getattr(maxima, func_name)(*func_args)
    else:
        raise Exception("Maxima integration is not available")

# Add GAP integration
try:
    import gap
    GAP_AVAILABLE = True
except ImportError:
    GAP_AVAILABLE = False

def gap_eval_func(self, args):
    if GAP_AVAILABLE:
        return gap.eval(args[0])
    else:
        raise Exception("GAP integration is not available")

def gap_call_func(self, args):
    if GAP_AVAILABLE:
        func_name = args[0]
        func_args = args[1:]
        return getattr(gap, func_name)(*func_args)
    else:
        raise Exception("GAP integration is not available")

# Add Magma integration
try:
    import magma
    MAGMA_AVAILABLE = True
except ImportError:
    MAGMA_AVAILABLE = False

def magma_eval_func(self, args):
    if MAGMA_AVAILABLE:
        return magma.eval(args[0])
    else:
        raise Exception("Magma integration is not available")

def magma_call_func(self, args):
    if MAGMA_AVAILABLE:
        func_name = args[0]
        func_args = args[1:]
        return getattr(magma, func_name)(*func_args)
    else:
        raise Exception("Magma integration is not available")

# Add Sage integration
try:
    import sage
    SAGE_AVAILABLE = True
except ImportError:
    SAGE_AVAILABLE = False

def sage_eval_func(self, args):
    if SAGE_AVAILABLE:
        return sage.eval(args[0])
    else:
        raise Exception("Sage integration is not available")

def sage_call_func(self, args):
    if SAGE_AVAILABLE:
        func_name = args[0]
        func_args = args[1:]
        return getattr(sage, func_name)(*func_args)
    else:
        raise Exception("Sage integration is not available")

# Add Singular integration
try:
    import singular
    SINGULAR_AVAILABLE = True
except ImportError:
    SINGULAR_AVAILABLE = False

def singular_eval_func(self, args):
    if SINGULAR_AVAILABLE:
        return singular.eval(args[0])
    else:
        raise Exception("Singular integration is not available")

def singular_call_func(self, args):
    if SINGULAR_AVAILABLE:
        func_name = args[0]
        func_args = args[1:]
        return getattr(singular, func_name)(*func_args)
    else:
        raise Exception("Singular integration is not available")

# Add Axiom integration
try:
    import axiom
    AXIOM_AVAILABLE = True
except ImportError:
    AXIOM_AVAILABLE = False

def axiom_eval_func(self, args):
    if AXIOM_AVAILABLE:
        return axiom.eval(args[0])
    else:
        raise Exception("Axiom integration is not available")

def axiom_call_func(self, args):
    if AXIOM_AVAILABLE:
        func_name = args[0]
        func_args = args[1:]
        return getattr(axiom, func_name)(*func_args)
    else:
        raise Exception("Axiom integration is not available")

# Add Reduce integration
try:
    import reduce
    REDUCE_AVAILABLE = True
except ImportError:
    REDUCE_AVAILABLE = False

def reduce_eval_func(self, args):
    if REDUCE_AVAILABLE:
        return reduce.eval(args[0])
    else:
        raise Exception("Reduce integration is not available")

def reduce_call_func(self, args):
    if REDUCE_AVAILABLE:
        func_name = args[0]
        func_args = args[1:]
        return getattr(reduce, func_name)(*func_args)
    else:
        raise Exception("Reduce integration is not available")

# Add Macsyma integration
try:
    import macsyma
    MACSYMA_AVAILABLE = True
except ImportError:
    MACSYMA_AVAILABLE = False

def macsyma_eval_func(self, args):
    if MACSYMA_AVAILABLE:
        return macsyma.eval(args[0])
    else:
        raise Exception("Macsyma integration is not available")

def macsyma_call_func(self, args):
    if MACSYMA_AVAILABLE:
        func_name = args[0]
        func_args = args[1:]
        return getattr(macsyma, func_name)(*func_args)
    else:
        raise Exception("Macsyma integration is not available")

# Add Derive integration
try:
    import derive
    DERIVE_AVAILABLE = True
except ImportError:
    DERIVE_AVAILABLE = False

def derive_eval_func(self, args):
    if DERIVE_AVAILABLE:
        return derive.eval(args[0])
    else:
        raise Exception("Derive integration is not available")

def derive_call_func(self, args):
    if DERIVE_AVAILABLE:
        func_name = args[0]
        func_args = args[1:]
        return getattr(derive, func_name)(*func_args)
    else:
        raise Exception("Derive integration is not available")

# Add muPAD integration
try:
    import mupad
    MUPAD_AVAILABLE = True
except ImportError:
    MUPAD_AVAILABLE = False

def mupad_eval_func(self, args):
    if MUPAD_AVAILABLE:
        return mupad.eval(args[0])
    else:
        raise Exception("muPAD integration is not available")

def mupad_call_func(self, args):
    if MUPAD_AVAILABLE:
        func_name = args[0]
        func_args = args[1:]
        return getattr(mupad, func_name)(*func_args)
    else:
        raise Exception("muPAD integration is not available")

# Add Yacas integration
try:
    import yacas
    YACAS_AVAILABLE = True
except ImportError:
    YACAS_AVAILABLE = False

def yacas_eval_func(self, args):
    if YACAS_AVAILABLE:
        return yacas.eval(args[0])
    else:
        raise Exception("Yacas integration is not available")

def yacas_call_func(self, args):
    if YACAS_AVAILABLE:
        func_name = args[0]
        func_args = args[1:]
        return getattr(yacas, func_name)(*func_args)
    else:
        raise Exception("Yacas integration is not available")

# Add Form integration
try:
    import form
    FORM_AVAILABLE = True
except ImportError:
    FORM_AVAILABLE = False

def form_eval_func(self, args):
    if FORM_AVAILABLE:
        return form.eval(args[0])
    else:
        raise Exception("Form integration is not available")

def form_call_func(self, args):
    if FORM_AVAILABLE:
        func_name = args[0]
        func_args = args[1:]
        return getattr(form, func_name)(*func_args)
    else:
        raise Exception("Form integration is not available")

printOptions = ["Created by Pranav Lejith (Amphibiar)", """Created by Pranav "Amphibiar" Lejith"""]
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

    # Custom ML/NLP functions
    def create_nlp_model(self, args):
        if len(args) != 1:
            raise ValueError("create_nlp_model() takes exactly one argument (model_type)")
        model_type = args[0]
        if model_type == "bert":
            return "Created BERT model"
        elif model_type == "gpt":
            return "Created GPT model"
        elif model_type == "lstm":
            return "Created LSTM model"
        else:
            raise ValueError("Unsupported model type. Use 'bert', 'gpt', or 'lstm'")

    def train_model(self, args):
        if len(args) != 3:
            raise ValueError("train_model() takes exactly 3 arguments (model, data, epochs)")
        model, data, epochs = args
        return f"Training {model} on {data} for {epochs} epochs"

    def predict(self, args):
        if len(args) != 2:
            raise ValueError("predict() takes exactly 2 arguments (model, input)")
        model, input_data = args
        return f"Prediction from {model} on {input_data}"

    def evaluate_model(self, args):
        if len(args) != 3:
            raise ValueError("evaluate_model() takes exactly 3 arguments (model, test_data, metrics)")
        model, test_data, metrics = args
        return f"Evaluating {model} on {test_data} using {metrics}"

    # Basic functions
    def display(self, args):
        print(*args)

    def get(self, args):
        return input(*args)

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

    def parse_line(self, line):
        line = line.strip()
        try:
            if line.startswith('display '):
                content = line[8:].strip()
                if content.startswith('"') and content.endswith('"') or \
                        content.startswith("'") and content.endswith("'"):
                    print(content[1:-1])
                elif content in self.variables:
                    print(self.variables[content])
                else:
                    print(self.evaluate_expression(content))

            elif line.startswith('get '):
                parts = line[4:].strip().split('"', 1)
                if len(parts) == 2 and (parts[1].endswith('"') or parts[1].endswith("'")):
                    var_name = parts[0].strip()
                    prompt = parts[1][:-1]
                    user_input = self.get([prompt])
                    self.variables[var_name] = user_input
                else:
                    raise Exception(f"Invalid get statement: {line}")

            elif "=" in line:
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

            elif line == "" or line.startswith("#"):
                pass

        except Exception as e:
            raise Exception(f"Error in line: {line}\n{str(e)}")

    # Register built-in functions
    builtin_functions = {
        'display': display,
        'get': get,
        'create_nlp_model': create_nlp_model,
        'train_model': train_model,
        'predict': predict,
        'evaluate_model': evaluate_model,
        'len': len,
        'max': max,
        'min': min,
        'sum': sum,
        'abs': abs,
        'round': round,
        'int': int,
        'float': float,
        'str': str,
        'bool': bool,
        'list': list,
        'tuple': tuple,
        'set': set,
        'dict': dict,
        'range': range,
    }

    def run(self):
        if self.interactive:
            print("Welcome to Orion Interpreter!")
            print("Type 'exit' to quit or 'help' for help")
            while True:
                try:
                    line = input(">>> ")
                    if line.lower() == 'exit':
                        break
                    elif line.lower() == 'help':
                        self.display_help()
                    else:
                        self.parse_line(line)
                except Exception as e:
                    print(f"Error: {str(e)}")
        else:
            with open(self.filename, 'r') as f:
                self.lines = f.readlines()
            while self.current_line < len(self.lines):
                self.parse_line(self.lines[self.current_line])
                self.current_line += 1

    def display_help(self):
        help_text = """
Orion Interpreter Help:

Basic Syntax:
- display: Output something, e.g., display "Hello"
- get: Get input from the user, e.g., get name "Enter your name"
- Variables: Use = for assignment, e.g., x = 10
- Functions: Define functions using def, e.g., def my_function(x): ...
- Control structures: Use if, elif, else, while, for
- List comprehensions: [x for x in range(10) if x % 2 == 0]
- Lambda functions: lambda x: x * 2

Built-in Functions:
- Basic: len, max, min, sum, abs, round
- Type conversion: int, float, str, bool, list, tuple, set, dict
- Sequences: range

ML/NLP Functions:
- create_nlp_model(model_type): Create a new NLP model
- train_model(model, data, epochs): Train a model
- predict(model, input): Make predictions
- evaluate_model(model, test_data, metrics): Evaluate model performance

Type 'exit' to quit the interactive shell.
"""
        print(help_text)

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
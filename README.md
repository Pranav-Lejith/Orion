<p align="center">
  <a href="https://github.com/Pranav-Lejith/Orion" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-Repo-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Repo" />
  </a>
 <img src="https://img.shields.io/badge/python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12" />
  <img src="https://img.shields.io/badge/windows-11-00A4EF?style=for-the-badge&logo=windows&logoColor=white" alt="Windows 11" />
  <img src="https://img.shields.io/badge/intel-Core_i7-0071C5?style=for-the-badge&logo=intel&logoColor=white" alt="Intel Core i7" />
  <img src="https://img.shields.io/badge/architecture-Intel-00A1F1?style=for-the-badge" alt="Architecture Intel" />
  <img src="https://img.shields.io/badge/ram-32GB-6F42C1?style=for-the-badge" alt="32GB RAM" />
  <img src="https://img.shields.io/badge/VS_Code-Visual_Studio_Code-005FCC?style=for-the-badge&logo=visual-studio-code&logoColor=white" alt="VS Code" />
  <img src="https://img.shields.io/badge/Linux-Tux-FAA61A?style=for-the-badge&logo=linux&logoColor=black" alt="Linux" />
  <img src="https://img.shields.io/badge/Fedora-Blue-294172?style=for-the-badge&logo=fedora&logoColor=white" alt="Fedora" />
  <img src="https://img.shields.io/badge/Debugging-Debug-FF8C00?style=for-the-badge&logo=bug&logoColor=white" alt="Debugging" />  <img src="https://img.shields.io/badge/TensorFlow-Compatible-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow Compatible" />
  <img src="https://img.shields.io/badge/NumPy-Compatible-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy Compatible" />
  <img src="https://img.shields.io/badge/Interactive-Shell-4CAF50?style=for-the-badge&logo=terminal&logoColor=white" alt="Interactive Shell" />
  <img src="https://img.shields.io/badge/Cross--Platform-Windows%20%7C%20Linux%20%7C%20macOS-blue?style=for-the-badge" alt="Cross Platform" />
  <img src="https://img.shields.io/badge/Colored-Output-FF69B4?style=for-the-badge&logo=rainbow&logoColor=white" alt="Colored Output" />
  <img src="https://img.shields.io/badge/File-Execution-orange?style=for-the-badge&logo=file&logoColor=white" alt="File Execution" />
  <img src="https://img.shields.io/badge/Built--in-Functions-purple?style=for-the-badge&logo=function&logoColor=white" alt="Built-in Functions" />
  <img src="https://img.shields.io/badge/Object--Oriented-Programming-red?style=for-the-badge&logo=code&logoColor=white" alt="OOP Support" />
  <img src="https://img.shields.io/badge/Technokreate-FF4B4B?style=for-the-badge&logo=taipy&logoColor=white" alt="Technokreate" />
    <a href="./LICENSE">
  <img src="https://img.shields.io/badge/license-PUOL--v1.0-red?style=for-the-badge" alt="License: PUOL v1.0" />
</a></p>

# Orion Interpreter - The Celestial Hunter of Programming Languages

Orion Interpreter is a powerful, feature-rich custom programming language interpreter built in Python. Named after the celestial hunter constellation, Orion brings together the simplicity of Python-like syntax with extensive built-in functionality, making it perfect for both beginners learning programming concepts and advanced users who need a flexible scripting environment.

This project was created by **Pranav Lejith (Amphibiar)**.

![Orion Logo](https://img.shields.io/badge/ORION-Interpreter-cyan?style=for-the-badge&logo=star&logoColor=white)

## Features

### 🚀 **Core Language Features**
- **Python-like Syntax**: Familiar and intuitive syntax for easy adoption
- **Interactive Shell**: REPL environment for testing and experimentation
- **File Execution**: Run complete programs from `.or` files
- **Variable Management**: Dynamic variable assignment and manipulation
- **Control Structures**: Full support for if/elif/else, while loops, and for loops

### 🔧 **Advanced Programming Constructs**
- **Function Definitions**: Create custom functions with parameters and return values
- **Object-Oriented Programming**: Class definitions with methods and inheritance
- **Lambda Functions**: Anonymous function support for functional programming
- **Decorators**: Function decoration capabilities
- **Context Managers**: `with` statement support for resource management

### 📚 **Extensive Built-in Library**
- **Mathematical Operations**: sin, cos, tan, sqrt, log, exp, floor, ceil, abs, round
- **String Manipulation**: upper, lower, capitalize, split, join, replace, find
- **Data Structures**: Lists, dictionaries, sets, tuples with full method support
- **File Operations**: File reading, writing, and manipulation
- **Random Functions**: Random number generation, choice selection, shuffling
- **Date & Time**: Current time, date creation, time formatting
- **JSON Support**: JSON encoding and decoding
- **Regular Expressions**: Pattern matching and text processing
- **Statistics**: Mean, median, mode, standard deviation calculations

### 🎨 **Enhanced User Experience**
- **Colored Output**: Beautiful terminal output with color support
- **Error Handling**: Comprehensive error messages with line numbers
- **Help System**: Built-in help and documentation
- **Developer Commands**: Special commands for development and debugging

### 🔬 **Scientific Computing Integration**
- **TensorFlow Support**: TensorFlow operations when available
- **NumPy Integration**: Array operations and mathematical computing
- **Data Processing**: CSV, XML, and database connectivity
- **Cryptography**: MD5, SHA256 hashing and Base64 encoding
- **Compression**: Data compression and decompression utilities

### 🎯 **Custom Functions**
- **Calculator**: Multi-operation calculator function
- **Hypotenuse**: Pythagorean theorem calculations
- **Random Picker**: Random selection from multiple options
- **Colored Text Display**: Terminal text coloring utilities

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Pranav-Lejith/Orion.git
   cd orion-interpreter
   ```

2. **Run the installation script**:
   ```bash
   install.bat
   ```
   This will automatically set up Orion and add it to your system PATH.

3. **Verify installation**:
   ```bash
   orion shell
   ```

## Usage

### Interactive Shell Mode
Launch the interactive REPL environment:
```bash
orion shell
```

### File Execution Mode
Run Orion programs from `.or` files:
```bash
orion run filename.or
```

### Example Orion Code

Create a file called `calculator.or`:
```orion
display "Welcome to Orion Calculator"
get op "Enter operation (+, -, *, /): "
get a "Enter first number: "
get b "Enter second number: "
a = int(a)
b = int(b)

if op == "+":
    display a + b
elif op == "-":
    display a - b
elif op == "*":
    display a * b
elif op == "/":
    if b != 0:
        display a / b
    else:
        display "Cannot divide by zero"
else:
    display "Unknown operation"

display "Thankyou!"
displayColoredText("CODE EXECUTED SUCCESSFULLY",'green')
```

Run it with:
```bash
orion run calculator.or
```

## Command Reference

### Basic Commands
- `display <expression>` - Output text or variables
- `get <variable> "<prompt>"` - Get user input
- `help` - Show help information
- `about` - Display version and author information
- `exit` - Quit interactive shell

### Built-in Functions
- **Math**: `abs()`, `round()`, `sqrt()`, `sin()`, `cos()`, `tan()`
- **Strings**: `upper()`, `lower()`, `split()`, `join()`, `replace()`
- **Lists**: `append()`, `extend()`, `remove()`, `pop()`, `sort()`
- **Utility**: `len()`, `type()`, `range()`, `enumerate()`

## System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, Linux, macOS
- **Memory**: Minimum 512MB RAM
- **Storage**: 50MB free space
- **Optional**: TensorFlow and NumPy for enhanced scientific computing

## Architecture

The Orion Interpreter follows a modular architecture:

1. **Lexical Analysis**: Token parsing and syntax recognition
2. **Expression Evaluation**: Mathematical and logical expression processing
3. **Control Flow**: Conditional and loop execution management
4. **Function Management**: User-defined and built-in function handling
5. **Variable Scope**: Context-aware variable management
6. **Error Handling**: Comprehensive error reporting and recovery

## Contributing

We welcome contributions to the Orion Interpreter! Please feel free to:

- Report bugs and issues
- Suggest new features
- Submit pull requests
- Improve documentation
- Add example programs

## Version History

- **v7.1.9** (Current): Enhanced error handling, improved built-in functions
- **v7.1.8**: TensorFlow and NumPy integration
- **v7.0.1**: Object-oriented programming support
- **v6.7.0**: Interactive shell and file execution
- **v5.0.0**: Core language features and built-in functions

## License

This project is licensed under the PUOL(v1.0) License. See [LICENSE](./LICENSE) for details.

## Credits and Acknowledgments

**Orion Interpreter** was created by **Pranav Lejith (Amphibiar)**.

*"The celestial hunter of the night sky"* - A programming language that guides you through the vast universe of code.

---

**Explore the cosmos of programming with Orion Interpreter!** 🌟

For support, questions, or contributions, please visit our [GitHub repository](https://github.com/Pranav-Lejith/Orion).

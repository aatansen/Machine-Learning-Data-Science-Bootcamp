<details>
<summary>Uses in learning section</summary>

**Generate random colors**

```python
# generating 200 colors from given list of 8 colors
import random
randomcolors = []
colours=['blue','brown','green','red','yellow','orange','black','white']
# print(colours[0])
for i in range(0, 200):
    n = random.randint(0, 7)
    randomcolors.append(colours[n])
print(randomcolors)
```

**Generate random numbers**

```python
# generating 200 numbers in a list between 0 and 7
import random
randomnum = []
for i in range(0, 200):
    n = random.randint(0, 7)
    randomnum.append(n)
print(randomnum)
```

</details>

<details>
<summary>Python Basic</summary>

### What is a Programming Language?

- It is simply a way to give instruction to computer
- Computer understand 0,1
- Lower level language —> Close to machine language, Assembly
- Higher level language — Close to human language, Python
- Source code —> Translate —> Machine Language
- Translator: [Compiler (C,C++) & Interpreter (Python)](https://www.programiz.com/article/difference-compiler-interpreter)
    
    ```mermaid
    %%{init: {'theme': 'dark', "flowchart" : { "curve" : "basis" } } }%%
    graph LR
    A[Source Code] -->|Translate|B[Compiler]
    A -->|Translate|C[Interpreter]
    C -->|Interpreted|D[Machine Language]
    B -->|Compiled|D[Machine Language]
    ```
    

### Python Interpreter

**Python Translator** : Cython (written in C) , Jython (written in Java), PyPy(written in Python), IronPython (Written in .net)

**Working procedure:**

Source Code—>Interpreter—>Byte Code—>Cpython VM —> Machine

### Environment Setup

- Terminal
    - [Windows Terminal](https://apps.microsoft.com/store/detail/windows-terminal/9N0DX20HK701)
- Code Editors
    - [Visual Studio Code](https://code.visualstudio.com/)
    - [Sublime Text](https://www.sublimetext.com/)
- IDE(Integrated Development Environment**)**
    - [PyCharm](https://www.jetbrains.com/pycharm/)
    - [Spyder](https://www.spyder-ide.org/)
    - [Replit](https://replit.com/) (Online)
    - [Glot](https://glot.io/) (Online Open Source)
- Notebooks
    - [Jupyter](https://jupyter.org/)

### Python Beginner

Notebook

[Difference between Python 2 & 3](https://www.geeksforgeeks.org/important-differences-between-python-2-x-and-python-3-x-with-examples/)

**Data types**

| Fundamental Data Types | Custom Types |
| --- | --- |
| int | Classes |
| float |  |
| bool |  |
| str |  |
| list |  |
| tuple |  |
| set | Specialized Types |
| dict | Module  |
| None | Packages |

[Math Functions](https://docs.python.org/3.10/library/numbers.html#numbers.Integral)

**Developer Fundamentals**

- Don’t read the dictionary
- Commenting your code
- Understanding Data Structures (`list`-.index but `dict`-key,value)
- What is clean / good code? —> Readability

**Operator Precedence**

| Operators | Meaning |
| --- | --- |
| () | Parentheses |
| ** | Exponent |
| +x, -x, ~x | Unary plus, Unary minus, Bitwise NOT |
| *, /, //, % | Multiplication, Division, Floor division, Modulus |
| +, - | Addition, Subtraction |
| <<, >> | Bitwise shift operators |
| & | Bitwise AND |
| ^ | Bitwise XOR |
| | | Bitwise OR |
| ==, !=, >, >=, <, <=, is, is not, in, not in | Comparisons, Identity, Membership operators |
| not | Logical NOT |
| and | Logical AND |
| or | Logical OR |

**[Python Variables](https://www.programiz.com/python-programming/variables-constants-literals)**

- Create a name that makes sense
- snake_case
- Start with lowercase or underscore
- letters, numbers, underscores
- Case sensitive
- Don’t overwrite keywords

**Notes**

Immutability

- A particular part of a string can’t be reassign

```python
# this will give error
num = '35345345'
num[0] = '2'
```

**Expressions vs Statements**

```python
iq=100 #this whole line is statement
user_age = iq / 5 #this whole line is statement
```

in user_age , `iq / 5` is expression

**Augmented Assignment Variable**

```python
some_value = 5
some_value = some_value+5
#now using augmented assignment
some_value = 5
some_value +=5
```

[**String Methods**](https://www.w3schools.com/python/python_ref_string.asp)

### Datatype List

- it is an order sequence of object
    - imagine as string —> `“hello”`
    - its a form of array but it has differences

```python
# all are list
li= [1,2,3,4,5]
li= ['a','b','c']
li= [1,2,'a']
```

- to copy a list to new one —> new_cart = old_cart[:]

</details>
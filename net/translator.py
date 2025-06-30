import re
import ast
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import keyword
from collections import defaultdict

class IntentType(Enum):
    VARIABLE_DECLARATION = "variable_declaration"
    FUNCTION_DEFINITION = "function_definition"
    CLASS_DEFINITION = "class_definition"
    LOOP = "loop"
    CONDITIONAL = "conditional"
    IMPORT = "import"
    OPERATION = "operation"
    COMMENT = "comment"
    DATA_STRUCTURE = "data_structure"
    ERROR_HANDLING = "error_handling"
    FILE_OPERATION = "file_operation"
    API_CALL = "api_call"
    MATHEMATICAL = "mathematical"
    STRING_OPERATION = "string_operation"
    RETURN_STATEMENT = "return"
    PRINT_STATEMENT = "print"
    METHOD_CALL = "method_call"
    PROPERTY_ACCESS = "property_access"
    ASSIGNMENT = "assignment"
    CONTINUATION = "continuation"

@dataclass
class Context:
    """Maintains conversation and code context"""
    variables: Dict[str, Any] = field(default_factory=dict)
    functions: Dict[str, Dict] = field(default_factory=dict)
    classes: Dict[str, Dict] = field(default_factory=dict)
    imports: Set[str] = field(default_factory=set)
    current_function: Optional[str] = None
    current_class: Optional[str] = None
    current_loop: Optional[str] = None
    last_subject: Optional[str] = None
    last_object: Optional[str] = None
    last_action: Optional[str] = None
    pronouns: Dict[str, str] = field(default_factory=dict)
    code_blocks: List[str] = field(default_factory=list)
    indent_level: int = 0
    in_condition: bool = False
    conversation_history: List[str] = field(default_factory=list)
    implicit_types: Dict[str, str] = field(default_factory=dict)

class UltimateNaturalLanguageToPython:
    def __init__(self):
        self.context = Context()
        self.initialize_language_patterns()
        self.initialize_python_knowledge()
        
    def initialize_language_patterns(self):
        """Initialize comprehensive English language patterns"""
        
        # Conversational patterns that indicate intent
        self.conversational_patterns = {
            'need': ['I need', 'we need', "I'll need", 'gonna need', 'gotta have'],
            'want': ['I want', 'we want', "I'd like", 'let me', 'wanna'],
            'should': ['should', 'ought to', 'better', 'supposed to'],
            'must': ['must', 'have to', 'need to', 'got to'],
            'create': ['make', 'create', 'build', 'construct', 'generate', 'init', 'start with'],
            'check': ['check', 'see if', 'verify', 'ensure', 'make sure', 'test'],
            'get': ['get', 'fetch', 'retrieve', 'grab', 'pull', 'obtain', 'acquire'],
            'process': ['process', 'handle', 'deal with', 'work with', 'manipulate'],
        }
        
        # Implicit type indicators
        self.type_indicators = {
            'number': ['number', 'count', 'amount', 'quantity', 'total', 'sum', 'age', 'score', 'price', 'cost', 'value'],
            'text': ['name', 'message', 'text', 'string', 'word', 'sentence', 'description', 'title', 'label'],
            'list': ['list', 'array', 'collection', 'group', 'set of', 'bunch of', 'series', 'sequence'],
            'boolean': ['flag', 'is', 'has', 'can', 'should', 'enabled', 'active', 'valid', 'done', 'finished'],
            'dict': ['dictionary', 'map', 'mapping', 'lookup', 'record', 'data', 'info', 'details'],
            'file': ['file', 'document', 'csv', 'json', 'txt', 'log'],
            'date': ['date', 'time', 'datetime', 'timestamp', 'when', 'moment'],
        }
        
        # Pronoun resolution patterns
        self.pronoun_patterns = {
            'it': ['it', "it's", 'its'],
            'they': ['they', 'them', 'their', "they're"],
            'this': ['this', 'that', 'these', 'those'],
        }
        
        # Action verb mappings to Python operations
        self.action_mappings = {
            # Variable operations
            'store': 'assign', 'save': 'assign', 'keep': 'assign', 'remember': 'assign',
            'set': 'assign', 'make': 'assign', 'let': 'assign', 'have': 'assign',
            
            # List operations
            'add': 'append', 'push': 'append', 'insert': 'insert', 'put': 'append',
            'remove': 'remove', 'delete': 'remove', 'pop': 'pop', 'take out': 'remove',
            'clear': 'clear', 'empty': 'clear', 'reset': 'clear',
            
            # Control flow
            'repeat': 'loop', 'iterate': 'loop', 'go through': 'loop', 'loop': 'loop',
            'check': 'condition', 'test': 'condition', 'verify': 'condition',
            'stop': 'break', 'exit': 'break', 'quit': 'break', 'end': 'break',
            'skip': 'continue', 'next': 'continue', 'pass': 'continue',
            
            # I/O operations
            'show': 'print', 'display': 'print', 'output': 'print', 'write': 'print',
            'say': 'print', 'tell': 'print', 'report': 'print',
            'read': 'input', 'ask': 'input', 'get input': 'input', 'prompt': 'input',
            
            # File operations
            'open': 'open', 'load': 'open', 'access': 'open',
            'save': 'write', 'write': 'write', 'dump': 'write',
            'close': 'close', 'finish': 'close',
        }
        
        # Connecting words and their implications
        self.connectors = {
            'and': 'sequential',
            'then': 'sequential', 
            'after': 'sequential',
            'next': 'sequential',
            'but': 'contrast',
            'however': 'contrast',
            'or': 'alternative',
            'either': 'alternative',
            'so': 'consequence',
            'therefore': 'consequence',
            'because': 'reason',
            'since': 'reason',
            'while': 'simultaneous',
            'during': 'simultaneous',
            'until': 'terminal_condition',
            'unless': 'negative_condition',
        }
        
        # Common programming phrases in natural language
        self.programming_phrases = {
            'for each': 'for_loop',
            'for every': 'for_loop',
            'for all': 'for_loop',
            'as long as': 'while_loop',
            'keep going': 'while_loop',
            'one by one': 'iteration',
            'step by step': 'iteration',
            'if possible': 'try_except',
            'in case': 'conditional',
            'by the way': 'comment',
            'note that': 'comment',
            'remember': 'comment',
            'TODO': 'comment',
            'FIXME': 'comment',
        }
        
    def initialize_python_knowledge(self):
        """Initialize comprehensive Python language knowledge"""
        
        # Python built-in functions and their natural language aliases
        self.builtin_aliases = {
            'length': 'len', 'size': 'len', 'count': 'len', 'how many': 'len',
            'maximum': 'max', 'biggest': 'max', 'largest': 'max', 'highest': 'max',
            'minimum': 'min', 'smallest': 'min', 'lowest': 'min', 'least': 'min',
            'total': 'sum', 'add up': 'sum', 'sum up': 'sum', 'altogether': 'sum',
            'sort': 'sorted', 'order': 'sorted', 'arrange': 'sorted',
            'reverse': 'reversed', 'backwards': 'reversed', 'flip': 'reversed',
            'type': 'type', 'kind': 'type', 'what is': 'type',
            'convert': 'cast', 'change': 'cast', 'make into': 'cast',
        }
        
        # Common module aliases
        self.module_aliases = {
            'random numbers': 'random',
            'math operations': 'math',
            'dates': 'datetime',
            'times': 'datetime',
            'files': 'os',
            'system': 'os',
            'web requests': 'requests',
            'http': 'requests',
            'json': 'json',
            'regular expressions': 're',
            'patterns': 're',
            'data analysis': 'pandas',
            'dataframes': 'pandas',
            'numbers': 'numpy',
            'arrays': 'numpy',
            'plotting': 'matplotlib',
            'graphs': 'matplotlib',
        }
        
        # Python operators in natural language
        self.operators = {
            'plus': '+', 'add': '+', 'and': '+', '+': '+',
            'minus': '-', 'subtract': '-', 'less': '-', '-': '-',
            'times': '*', 'multiply': '*', 'multiplied by': '*', '*': '*',
            'divided by': '/', 'divide': '/', 'over': '/', '/': '/',
            'modulo': '%', 'remainder': '%', 'mod': '%', '%': '%',
            'power': '**', 'to the power': '**', 'squared': '**2', 'cubed': '**3',
            'equals': '==', 'is equal to': '==', 'same as': '==',
            'not equal': '!=', 'different': '!=', 'not the same': '!=',
            'greater than': '>', 'more than': '>', 'bigger than': '>',
            'less than': '<', 'smaller than': '<', 'fewer than': '<',
            'at least': '>=', 'greater or equal': '>=', 'no less than': '>=',
            'at most': '<=', 'less or equal': '<=', 'no more than': '<=',
            'and': 'and', 'both': 'and', 'as well as': 'and',
            'or': 'or', 'either': 'or',
            'not': 'not', "isn't": 'not', "doesn't": 'not',
        }

    def understand_intent(self, text: str) -> List[Tuple[IntentType, Dict[str, Any]]]:
        """Understand the intent(s) from natural language"""
        text = text.strip()
        intents = []
        
        # Add to conversation history
        self.context.conversation_history.append(text)
        
        # Resolve pronouns based on context
        text = self.resolve_pronouns(text)
        
        # Split into logical segments
        segments = self.split_into_segments(text)
        
        for segment in segments:
            # Check each type of intent
            intent = self.identify_primary_intent(segment)
            if intent:
                intents.append(intent)
        
        # If no specific intent found, try to infer from context
        if not intents and text:
            inferred = self.infer_intent_from_context(text)
            if inferred:
                intents.append(inferred)
        
        return intents

    def resolve_pronouns(self, text: str) -> str:
        """Resolve pronouns based on context"""
        words = text.split()
        resolved = []
        
        for word in words:
            lower_word = word.lower()
            if lower_word in ['it', 'its', "it's"]:
                # Replace with last mentioned subject or object
                if self.context.last_object:
                    resolved.append(self.context.last_object)
                elif self.context.last_subject:
                    resolved.append(self.context.last_subject)
                else:
                    resolved.append(word)
            elif lower_word in ['they', 'them', 'their']:
                # Check if referring to a list or collection
                for var, var_type in self.context.implicit_types.items():
                    if 'list' in var_type or 'array' in var_type:
                        resolved.append(var)
                        break
                else:
                    resolved.append(word)
            else:
                resolved.append(word)
        
        return ' '.join(resolved)

    def split_into_segments(self, text: str) -> List[str]:
        """Split text into logical segments based on connectors and punctuation"""
        # First split by common sentence boundaries
        segments = re.split(r'[.!?;]\s*', text)
        
        # Further split by connecting words that indicate new actions
        final_segments = []
        for segment in segments:
            if not segment.strip():
                continue
                
            # Split by sequential connectors
            sub_segments = re.split(r'\b(?:then|after that|next|also|and then)\b', segment, flags=re.IGNORECASE)
            final_segments.extend([s.strip() for s in sub_segments if s.strip()])
        
        return final_segments

    def identify_primary_intent(self, text: str) -> Optional[Tuple[IntentType, Dict[str, Any]]]:
        """Identify the primary intent of a text segment"""
        text_lower = text.lower()
        
        # Check for explicit programming constructs first
        if self.is_function_definition(text):
            return self.parse_function_definition(text)
        
        if self.is_class_definition(text):
            return self.parse_class_definition(text)
        
        if self.is_loop(text):
            return self.parse_loop(text)
        
        if self.is_conditional(text):
            return self.parse_conditional(text)
        
        if self.is_import(text):
            return self.parse_import(text)
        
        # Check for data structure creation
        if self.is_data_structure_creation(text):
            return self.parse_data_structure(text)
        
        # Check for operations on existing variables
        if self.is_operation(text):
            return self.parse_operation(text)
        
        # Check for variable assignment
        if self.is_assignment(text):
            return self.parse_assignment(text)
        
        # Check for print/output
        if self.is_output(text):
            return self.parse_output(text)
        
        # Check for return statement
        if self.is_return(text):
            return self.parse_return(text)
        
        # Check for file operations
        if self.is_file_operation(text):
            return self.parse_file_operation(text)
        
        # Default to comment if unclear
        return (IntentType.COMMENT, {'text': text})

    def is_function_definition(self, text: str) -> bool:
        """Check if text describes a function definition"""
        function_indicators = [
            'function', 'method', 'def', 'define', 'create a function',
            'make a function', 'procedure', 'subroutine', 'that takes',
            'that accepts', 'with parameters', 'takes arguments'
        ]
        return any(indicator in text.lower() for indicator in function_indicators)

    def parse_function_definition(self, text: str) -> Tuple[IntentType, Dict[str, Any]]:
        """Parse function definition from natural language"""
        # Extract function name
        name_patterns = [
            r'function\s+(?:called\s+)?(\w+)',
            r'(?:create|make|define)\s+(?:a\s+)?(?:function|method)\s+(?:called\s+)?(\w+)',
            r'(\w+)\s+function',
            r'def\s+(\w+)',
        ]
        
        func_name = None
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                func_name = match.group(1)
                break
        
        if not func_name:
            # Try to infer from context
            func_name = self.extract_identifier(text, 'function')
        
        # Extract parameters
        params = self.extract_function_parameters(text)
        
        # Extract return info
        returns = self.extract_return_info(text)
        
        # Extract description
        description = self.extract_description(text)
        
        return (IntentType.FUNCTION_DEFINITION, {
            'name': func_name,
            'parameters': params,
            'returns': returns,
            'description': description
        })

    def extract_function_parameters(self, text: str) -> List[str]:
        """Extract function parameters from natural language"""
        param_patterns = [
            r'(?:takes?|accepts?|with parameters?|with arguments?)\s+([^.]+?)(?:\s+and\s+)?(?:returns?|$)',
            r'(?:parameters?|arguments?)\s*:?\s*([^.]+?)(?:\s+and\s+)?(?:returns?|$)',
            r'\(([^)]+)\)',
        ]
        
        for pattern in param_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                param_text = match.group(1)
                # Parse individual parameters
                params = []
                
                # Split by commas and 'and'
                param_parts = re.split(r',|\s+and\s+', param_text)
                for part in param_parts:
                    part = part.strip()
                    # Remove articles and clean up
                    part = re.sub(r'\b(a|an|the)\b', '', part).strip()
                    if part and part.lower() not in ['nothing', 'none', 'no parameters']:
                        params.append(self.clean_identifier(part))
                
                return params
        
        return []

    def extract_identifier(self, text: str, context_type: str) -> str:
        """Extract an identifier (variable/function/class name) from text"""
        # Remove common prefixes
        prefixes = ['create', 'make', 'define', 'declare', 'call it', 'called', 'named']
        text_clean = text.lower()
        for prefix in prefixes:
            text_clean = text_clean.replace(prefix, '')
        
        # Look for quoted names
        quoted = re.search(r'["\'](\w+)["\']', text_clean)
        if quoted:
            return quoted.group(1)
        
        # Look for clear identifiers
        words = text_clean.split()
        
        # Filter out common words
        stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                     'that', 'which', 'this', 'these', 'those', 'to', 'for',
                     'with', 'function', 'variable', 'class', 'method'}
        
        candidates = [w for w in words if w not in stop_words and w.isalnum()]
        
        if candidates:
            # Prefer longer, more specific names
            return max(candidates, key=len)
        
        # Default based on context
        defaults = {
            'function': 'func',
            'variable': 'var',
            'class': 'MyClass',
            'method': 'method'
        }
        
        return defaults.get(context_type, 'item')

    def clean_identifier(self, name: str) -> str:
        """Clean an identifier to be Python-compliant"""
        # Remove special characters
        name = re.sub(r'[^\w]', '_', name)
        
        # Ensure doesn't start with number
        if name and name[0].isdigit():
            name = '_' + name
        
        # Remove multiple underscores
        name = re.sub(r'_+', '_', name)
        
        # Strip underscores from ends
        name = name.strip('_')
        
        # Check for Python keywords
        if keyword.iskeyword(name):
            name = name + '_var'
        
        return name or 'var'

    def is_assignment(self, text: str) -> bool:
        """Check if text describes a variable assignment"""
        assignment_patterns = [
            r'\b(?:is|equals?|=)\b',
            r'\b(?:set|make|let|have|store|save|keep)\b.*\b(?:to|as|be)\b',
            r':=',  # Some people use this
            'gets the value',
            'has the value',
            'should be',
            'will be',
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in assignment_patterns)

    def parse_assignment(self, text: str) -> Tuple[IntentType, Dict[str, Any]]:
        """Parse variable assignment from natural language"""
        # Patterns for different assignment styles
        patterns = [
            # Direct assignment: "x is 5", "x = 5", "x equals 5"
            r'(\w+)\s*(?:is|equals?|=)\s*(.+)',
            
            # Verbose assignment: "set x to 5", "make x equal to 5"
            r'(?:set|make|let)\s+(\w+)\s+(?:to|be|equal to|equals?)\s*(.+)',
            
            # Storage style: "store 5 in x", "save 5 as x"
            r'(?:store|save|put)\s+(.+?)\s+(?:in|as|to)\s+(\w+)',
            
            # Declaration style: "create variable x with value 5"
            r'(?:create|declare)\s+(?:a\s+)?(?:variable\s+)?(\w+)\s+(?:with value|as|=)\s+(.+)',
            
            # Have style: "have x be 5", "x should be 5"
            r'(?:have\s+)?(\w+)\s+(?:should\s+)?be\s+(.+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if 'store' in pattern or 'save' in pattern:
                    # Reversed order
                    value_text = match.group(1)
                    var_name = match.group(2)
                else:
                    var_name = match.group(1)
                    value_text = match.group(2)
                
                var_name = self.clean_identifier(var_name)
                value = self.parse_value(value_text)
                
                # Track variable and infer type
                self.context.variables[var_name] = value
                self.context.last_subject = var_name
                self.infer_type(var_name, value_text)
                
                return (IntentType.VARIABLE_DECLARATION, {
                    'name': var_name,
                    'value': value
                })
        
        return (IntentType.COMMENT, {'text': text})

    def parse_value(self, text: str) -> str:
        """Parse a value from natural language to Python representation"""
        text = text.strip()
        
        # Handle None/null
        if text.lower() in ['none', 'null', 'nothing', 'empty', 'nil']:
            return 'None'
        
        # Handle booleans
        if text.lower() in ['true', 'yes', 'on', 'enabled', 'active']:
            return 'True'
        if text.lower() in ['false', 'no', 'off', 'disabled', 'inactive']:
            return 'False'
        
        # Handle numbers
        # Remove commas from numbers
        text_no_commas = text.replace(',', '')
        
        # Check for numeric words
        number_words = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'ten': '10', 'hundred': '100', 'thousand': '1000', 'million': '1000000'
        }
        
        if text.lower() in number_words:
            return number_words[text.lower()]
        
        # Try to parse as number
        try:
            if '.' in text_no_commas:
                float(text_no_commas)
                return text_no_commas
            else:
                int(text_no_commas)
                return text_no_commas
        except ValueError:
            pass
        
        # Handle lists
        if any(indicator in text.lower() for indicator in ['list of', 'array of', '[', ']']):
            return self.parse_list_value(text)
        
        # Handle dictionaries
        if any(indicator in text.lower() for indicator in ['dictionary', 'map', '{', '}']):
            return self.parse_dict_value(text)
        
        # Handle mathematical expressions
        if self.is_mathematical_expression(text):
            return self.parse_mathematical_expression(text)
        
        # Handle existing variable references
        if text in self.context.variables:
            return text
        
        # Handle function calls
        if '(' in text and ')' in text:
            return self.parse_function_call(text)
        
        # Handle property access
        if '.' in text and self.is_property_access(text):
            return text
        
        # Check if it's likely a variable reference
        if self.is_likely_variable_reference(text):
            return text
        
        # Default to string
        # Remove quotes if already quoted
        if (text.startswith('"') and text.endswith('"')) or \
           (text.startswith("'") and text.endswith("'")):
            return text
        
        return f'"{text}"'

    def parse_list_value(self, text: str) -> str:
        """Parse a list from natural language"""
        # Remove list indicators
        text = re.sub(r'\b(list|array)\s+of\s+', '', text, flags=re.IGNORECASE)
        
        # Check if already in Python list format
        if text.strip().startswith('[') and text.strip().endswith(']'):
            return text.strip()
        
        # Parse items
        items = []
        
        # Split by common separators
        if ' and ' in text or ',' in text:
            parts = re.split(r',|\s+and\s+', text)
            for part in parts:
                part = part.strip()
                if part:
                    items.append(self.parse_value(part))
        else:
            # Single item or range
            if 'through' in text or 'to' in text or '-' in text:
                # Parse range
                range_match = re.search(r'(\d+)\s*(?:through|to|-)\s*(\d+)', text)
                if range_match:
                    start = int(range_match.group(1))
                    end = int(range_match.group(2))
                    return f'list(range({start}, {end + 1}))'
            
            # Single item
            items.append(self.parse_value(text))
        
        return f'[{", ".join(items)}]'

    def is_mathematical_expression(self, text: str) -> bool:
        """Check if text contains mathematical operations"""
        math_indicators = ['plus', 'minus', 'times', 'divided', 'multiplied',
                          '+', '-', '*', '/', 'squared', 'cubed', 'power',
                          'sum of', 'product of', 'difference']
        
        return any(indicator in text.lower() for indicator in math_indicators)

    def parse_mathematical_expression(self, text: str) -> str:
        """Parse mathematical expressions from natural language"""
        # Replace word operators with symbols
        replacements = {
            ' plus ': ' + ',
            ' add ': ' + ',
            ' minus ': ' - ',
            ' subtract ': ' - ',
            ' times ': ' * ',
            ' multiplied by ': ' * ',
            ' divided by ': ' / ',
            ' over ': ' / ',
            ' modulo ': ' % ',
            ' mod ': ' % ',
            ' to the power of ': ' ** ',
            ' squared': ' ** 2',
            ' cubed': ' ** 3',
        }
        
        result = text.lower()
        for word, symbol in replacements.items():
            result = result.replace(word, symbol)
        
        # Handle "sum of x and y" style
        sum_match = re.search(r'sum of (.+?) and (.+)', result)
        if sum_match:
            return f'{sum_match.group(1)} + {sum_match.group(2)}'
        
        # Clean up and parse components
        components = re.split(r'([+\-*/%()**])', result)
        parsed = []
        
        for comp in components:
            comp = comp.strip()
            if comp in ['+', '-', '*', '/', '%', '**', '(', ')']:
                parsed.append(comp)
            elif comp:
                # Parse as value
                parsed.append(self.parse_value(comp))
        
        return ' '.join(parsed)

    def is_loop(self, text: str) -> bool:
        """Check if text describes a loop"""
        loop_indicators = [
            'for each', 'for every', 'for all', 'iterate', 'loop',
            'repeat', 'while', 'until', 'go through', 'process each',
            'do this for', 'keep doing', 'continue until'
        ]
        
        return any(indicator in text.lower() for indicator in loop_indicators)

    def parse_loop(self, text: str) -> Tuple[IntentType, Dict[str, Any]]:
        """Parse loop from natural language"""
        text_lower = text.lower()
        
        # For each/every style loops
        for_each_match = re.search(r'(?:for\s+)?(?:each|every|all)\s+(\w+)\s+(?:in|of|from)\s+(\w+)', text, re.IGNORECASE)
        if for_each_match:
            var = self.clean_identifier(for_each_match.group(1))
            iterable = for_each_match.group(2)
            
            self.context.variables[var] = None
            self.context.current_loop = var
            
            return (IntentType.LOOP, {
                'type': 'for',
                'variable': var,
                'iterable': iterable
            })
        
        # Repeat N times
        repeat_match = re.search(r'repeat\s+(.+?)\s+times?', text, re.IGNORECASE)
        if repeat_match:
            times = self.parse_value(repeat_match.group(1))
            return (IntentType.LOOP, {
                'type': 'for',
                'variable': '_',
                'iterable': f'range({times})'
            })
        
        # While loops
        while_match = re.search(r'(?:while|as long as|keep doing while|until)\s+(.+)', text, re.IGNORECASE)
        if while_match:
            condition = self.parse_condition(while_match.group(1))
            
            # If "until", negate the condition
            if 'until' in text_lower:
                condition = f'not ({condition})'
            
            return (IntentType.LOOP, {
                'type': 'while',
                'condition': condition
            })
        
        # Range-based for loops
        range_match = re.search(r'(?:for\s+)?(\w+)\s+from\s+(\d+)\s+to\s+(\d+)(?:\s+step\s+(\d+))?', text, re.IGNORECASE)
        if range_match:
            var = self.clean_identifier(range_match.group(1))
            start = range_match.group(2)
            end = range_match.group(3)
            step = range_match.group(4) if range_match.group(4) else '1'
            
            return (IntentType.LOOP, {
                'type': 'for',
                'variable': var,
                'iterable': f'range({start}, {int(end) + 1}, {step})'
            })
        
        return (IntentType.COMMENT, {'text': text})

    def parse_condition(self, text: str) -> str:
        """Parse a condition from natural language"""
        # Handle compound conditions
        if ' and ' in text.lower():
            parts = text.split(' and ')
            conditions = [self.parse_single_condition(p.strip()) for p in parts]
            return ' and '.join(conditions)
        
        if ' or ' in text.lower():
            parts = text.split(' or ')
            conditions = [self.parse_single_condition(p.strip()) for p in parts]
            return ' or '.join(conditions)
        
        return self.parse_single_condition(text)

    def parse_single_condition(self, text: str) -> str:
        """Parse a single condition"""
        # Normalize text
        text = text.strip()
        
        # Patterns for conditions
        patterns = [
            # Comparison patterns
            (r'(\w+)\s+is\s+(?:equal\s+to|the\s+same\s+as)\s+(.+)', '{0} == {1}'),
            (r'(\w+)\s+(?:equals?|==)\s+(.+)', '{0} == {1}'),
            (r'(\w+)\s+is\s+not\s+(?:equal\s+to)?\s*(.+)', '{0} != {1}'),
            (r'(\w+)\s+(?:!=|<>)\s+(.+)', '{0} != {1}'),
            (r'(\w+)\s+is\s+(?:greater|more|higher|bigger)\s+than\s+(.+)', '{0} > {1}'),
            (r'(\w+)\s+>\s+(.+)', '{0} > {1}'),
            (r'(\w+)\s+is\s+(?:less|fewer|lower|smaller)\s+than\s+(.+)', '{0} < {1}'),
            (r'(\w+)\s+<\s+(.+)', '{0} < {1}'),
            (r'(\w+)\s+is\s+(?:at\s+least|greater\s+than\s+or\s+equal\s+to)\s+(.+)', '{0} >= {1}'),
            (r'(\w+)\s+>=\s+(.+)', '{0} >= {1}'),
            (r'(\w+)\s+is\s+(?:at\s+most|less\s+than\s+or\s+equal\s+to)\s+(.+)', '{0} <= {1}'),
            (r'(\w+)\s+<=\s+(.+)', '{0} <= {1}'),
            
            # Membership patterns
            (r'(.+?)\s+is\s+in\s+(\w+)', '{0} in {1}'),
            (r'(\w+)\s+contains?\s+(.+)', '{1} in {0}'),
            (r'(\w+)\s+has\s+(.+)', '{1} in {0}'),
            
            # Boolean patterns
            (r'(\w+)\s+is\s+(true|false)', '{0} == {1}'),
            (r'(\w+)\s+is\s+(on|off)', '{0} == {1}'),
            (r'(\w+)\s+is\s+(enabled|disabled)', '{0} == {1}'),
            (r'(\w+)\s+exists?', '{0} is not None'),
            (r'(\w+)\s+is\s+(?:none|null|nothing)', '{0} is None'),
            (r'not\s+(\w+)', 'not {0}'),
            (r'(\w+)\s+is\s+not\s+(true|false)', '{0} != {1}'),
            
            # Type checking
            (r'(\w+)\s+is\s+a\s+(\w+)', 'isinstance({0}, {1})'),
            (r'type\s+of\s+(\w+)\s+is\s+(\w+)', 'type({0}) == {1}'),
        ]
        
        # Try each pattern
        for pattern, template in patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                groups = list(match.groups())
                
                # Process each group
                for i, group in enumerate(groups):
                    # Special handling for boolean values
                    if group.lower() in ['true', 'on', 'enabled']:
                        groups[i] = 'True'
                    elif group.lower() in ['false', 'off', 'disabled']:
                        groups[i] = 'False'
                    # Check if it's a variable reference
                    elif group in self.context.variables or self.is_likely_variable_reference(group):
                        groups[i] = group
                    else:
                        # Parse as value
                        groups[i] = self.parse_value(group)
                
                return template.format(*groups)
        
        # If no pattern matches, assume it's a boolean check
        if text in self.context.variables:
            return text
        
        # Try to clean up and make a best guess
        return self.clean_identifier(text)

    def is_likely_variable_reference(self, text: str) -> bool:
        """Check if text is likely a variable reference"""
        # Single word that looks like an identifier
        if re.match(r'^[a-zA-Z_]\w*$', text):
            # Not a common literal word
            literal_words = {'true', 'false', 'none', 'null'}
            if text.lower() not in literal_words:
                return True
        
        # Array or object access
        if re.match(r'^[a-zA-Z_]\w*\[.+\]$', text):
            return True
        
        # Property access
        if re.match(r'^[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+$', text):
            return True
        
        # Function call
        if re.match(r'^[a-zA-Z_]\w*\(.+\)$', text):
            return True
        
        return False

    def is_output(self, text: str) -> bool:
        """Check if text describes output/print"""
        output_indicators = ['print', 'show', 'display', 'output', 'write',
                           'say', 'tell', 'report', 'echo', 'log']
        
        return any(indicator in text.lower() for indicator in output_indicators)

    def parse_output(self, text: str) -> Tuple[IntentType, Dict[str, Any]]:
        """Parse output/print statement"""
        # Remove output indicators
        output_words = ['print', 'show', 'display', 'output', 'write',
                       'say', 'tell', 'report', 'echo', 'log']
        
        text_clean = text
        for word in output_words:
            text_clean = re.sub(f'\\b{word}\\b', '', text_clean, flags=re.IGNORECASE)
        
        text_clean = text_clean.strip()
        
        # Parse what to output
        output_value = self.parse_value(text_clean) if text_clean else '""'
        
        return (IntentType.PRINT_STATEMENT, {
            'value': output_value
        })

    def infer_type(self, var_name: str, value_text: str):
        """Infer variable type from name and value"""
        value_lower = value_text.lower()
        
        # Check explicit type indicators
        for type_name, indicators in self.type_indicators.items():
            if any(ind in var_name.lower() for ind in indicators):
                self.context.implicit_types[var_name] = type_name
                return
            if any(ind in value_lower for ind in indicators):
                self.context.implicit_types[var_name] = type_name
                return

    def generate_code(self, intents: List[Tuple[IntentType, Dict[str, Any]]]) -> str:
        """Generate Python code from parsed intents"""
        code_lines = []
        
        for intent_type, data in intents:
            if intent_type == IntentType.VARIABLE_DECLARATION:
                line = f"{data['name']} = {data['value']}"
                code_lines.append(line)
                
            elif intent_type == IntentType.FUNCTION_DEFINITION:
                params = ', '.join(data['parameters']) if data['parameters'] else ''
                code_lines.append(f"def {data['name']}({params}):")
                self.context.current_function = data['name']
                self.context.indent_level += 1
                
                # Add docstring if description available
                if data.get('description'):
                    code_lines.append('    """' + data['description'] + '"""')
                
                # Add pass if no body yet
                code_lines.append('    pass')
                
            elif intent_type == IntentType.LOOP:
                if data['type'] == 'for':
                    code_lines.append(f"for {data['variable']} in {data['iterable']}:")
                else:  # while
                    code_lines.append(f"while {data['condition']}:")
                self.context.indent_level += 1
                code_lines.append('    pass')
                
            elif intent_type == IntentType.CONDITIONAL:
                if data['type'] == 'if':
                    code_lines.append(f"if {data['condition']}:")
                elif data['type'] == 'elif':
                    code_lines.append(f"elif {data['condition']}:")
                else:  # else
                    code_lines.append("else:")
                self.context.indent_level += 1
                code_lines.append('    pass')
                
            elif intent_type == IntentType.PRINT_STATEMENT:
                indent = '    ' * self.context.indent_level
                code_lines.append(f"{indent}print({data['value']})")
                
            elif intent_type == IntentType.RETURN_STATEMENT:
                indent = '    ' * self.context.indent_level
                code_lines.append(f"{indent}return {data['value']}")
                
            elif intent_type == IntentType.IMPORT:
                # Add imports at the top
                import_line = data['import_statement']
                if import_line not in self.context.imports:
                    self.context.imports.add(import_line)
                    
            elif intent_type == IntentType.COMMENT:
                indent = '    ' * self.context.indent_level
                code_lines.append(f"{indent}# {data['text']}")
        
        # Prepend imports
        if self.context.imports:
            import_lines = sorted(list(self.context.imports))
            code_lines = import_lines + [''] + code_lines
        
        return '\n'.join(code_lines)

    def process_natural_language(self, text: str) -> str:
        """Main method to process natural language and generate Python code"""
        # Understand intents
        intents = self.understand_intent(text)
        
        # Generate code
        if intents:
            return self.generate_code(intents)
        
        return "# Unable to parse input"

    # Additional parsing methods for other intent types...
    
    def is_conditional(self, text: str) -> bool:
        """Check if text describes a conditional"""
        conditional_indicators = ['if', 'when', 'whenever', 'in case', 'provided',
                                'unless', 'else', 'otherwise', 'elif']
        
        return any(indicator in text.lower().split() for indicator in conditional_indicators)

    def parse_conditional(self, text: str) -> Tuple[IntentType, Dict[str, Any]]:
        """Parse conditional from natural language"""
        text_lower = text.lower()
        
        # Determine type
        if text_lower.strip() in ['else', 'otherwise']:
            return (IntentType.CONDITIONAL, {'type': 'else'})
        
        cond_type = 'if'
        if any(word in text_lower for word in ['else if', 'elif', 'otherwise if']):
            cond_type = 'elif'
        
        # Extract condition
        condition_match = re.search(r'(?:if|when|elif|provided|unless)\s+(.+?)(?:\s+then|\s*:|$)', 
                                  text, re.IGNORECASE)
        
        if condition_match:
            condition_text = condition_match.group(1)
            condition = self.parse_condition(condition_text)
            
            # Handle "unless" by negating
            if 'unless' in text_lower:
                condition = f'not ({condition})'
            
            return (IntentType.CONDITIONAL, {
                'type': cond_type,
                'condition': condition
            })
        
        return (IntentType.COMMENT, {'text': text})

    def is_import(self, text: str) -> bool:
        """Check if text describes an import"""
        import_indicators = ['import', 'use', 'include', 'require', 'need',
                           'load', 'bring in']
        
        # Also check for known module names
        if any(module in text.lower() for module in self.module_aliases.values()):
            return True
        
        return any(indicator in text.lower() for indicator in import_indicators)

    def parse_import(self, text: str) -> Tuple[IntentType, Dict[str, Any]]:
        """Parse import statement"""
        # Check for module aliases
        for alias, module in self.module_aliases.items():
            if alias in text.lower():
                return (IntentType.IMPORT, {
                    'import_statement': f'import {module}'
                })
        
        # Standard import patterns
        import_match = re.search(r'(?:import|use|include|require)\s+(\w+)(?:\s+as\s+(\w+))?', 
                               text, re.IGNORECASE)
        
        if import_match:
            module = import_match.group(1)
            alias = import_match.group(2)
            
            if alias:
                import_stmt = f'import {module} as {alias}'
            else:
                import_stmt = f'import {module}'
            
            return (IntentType.IMPORT, {
                'import_statement': import_stmt
            })
        
        # From import
        from_match = re.search(r'from\s+(\w+)\s+import\s+(.+)', text, re.IGNORECASE)
        if from_match:
            return (IntentType.IMPORT, {
                'import_statement': f'from {from_match.group(1)} import {from_match.group(2)}'
            })
        
        return (IntentType.COMMENT, {'text': text})

    def is_return(self, text: str) -> bool:
        """Check if text describes a return statement"""
        return_indicators = ['return', 'give back', 'output', 'result',
                           'send back', 'yield']
        
        return any(indicator in text.lower() for indicator in return_indicators)

    def parse_return(self, text: str) -> Tuple[IntentType, Dict[str, Any]]:
        """Parse return statement"""
        # Remove return indicators
        return_words = ['return', 'give back', 'output', 'result',
                       'send back', 'yield']
        
        text_clean = text
        for word in return_words:
            text_clean = re.sub(f'\\b{word}\\b', '', text_clean, flags=re.IGNORECASE)
        
        text_clean = text_clean.strip()
        
        # Parse what to return
        return_value = self.parse_value(text_clean) if text_clean else 'None'
        
        return (IntentType.RETURN_STATEMENT, {
            'value': return_value
        })

    def is_data_structure_creation(self, text: str) -> bool:
        """Check if text describes creating a data structure"""
        ds_indicators = ['list', 'array', 'dictionary', 'dict', 'set',
                        'tuple', 'collection', 'mapping']
        
        return any(indicator in text.lower() for indicator in ds_indicators)

    def parse_data_structure(self, text: str) -> Tuple[IntentType, Dict[str, Any]]:
        """Parse data structure creation"""
        # List creation
        list_match = re.search(r'(?:create|make)\s+(?:a\s+)?(?:list|array)\s+(?:called\s+)?(\w+)(?:\s+with\s+(.+))?',
                             text, re.IGNORECASE)
        if list_match:
            name = self.clean_identifier(list_match.group(1))
            values = list_match.group(2)
            
            if values:
                value = self.parse_list_value(values)
            else:
                value = '[]'
            
            self.context.variables[name] = value
            self.context.implicit_types[name] = 'list'
            
            return (IntentType.VARIABLE_DECLARATION, {
                'name': name,
                'value': value
            })
        
        # Dictionary creation
        dict_match = re.search(r'(?:create|make)\s+(?:a\s+)?(?:dictionary|dict|mapping)\s+(?:called\s+)?(\w+)',
                             text, re.IGNORECASE)
        if dict_match:
            name = self.clean_identifier(dict_match.group(1))
            self.context.variables[name] = '{}'
            self.context.implicit_types[name] = 'dict'
            
            return (IntentType.VARIABLE_DECLARATION, {
                'name': name,
                'value': '{}'
            })
        
        return (IntentType.COMMENT, {'text': text})

    def is_operation(self, text: str) -> bool:
        """Check if text describes an operation on existing data"""
        operation_indicators = ['add', 'append', 'remove', 'delete', 'insert',
                              'update', 'modify', 'change', 'sort', 'reverse',
                              'clear', 'pop', 'push']
        
        return any(indicator in text.lower() for indicator in operation_indicators)

    def parse_operation(self, text: str) -> Tuple[IntentType, Dict[str, Any]]:
        """Parse operations on existing data structures"""
        # Append/Add to list
        append_match = re.search(r'(?:add|append|push)\s+(.+?)\s+to\s+(?:the\s+)?(?:list\s+)?(\w+)',
                               text, re.IGNORECASE)
        if append_match:
            value = self.parse_value(append_match.group(1))
            list_name = append_match.group(2)
            
            return (IntentType.METHOD_CALL, {
                'object': list_name,
                'method': 'append',
                'arguments': [value]
            })
        
        # Remove from list
        remove_match = re.search(r'(?:remove|delete)\s+(.+?)\s+from\s+(?:the\s+)?(?:list\s+)?(\w+)',
                               text, re.IGNORECASE)
        if remove_match:
            value = self.parse_value(remove_match.group(1))
            list_name = remove_match.group(2)
            
            return (IntentType.METHOD_CALL, {
                'object': list_name,
                'method': 'remove',
                'arguments': [value]
            })
        
        return (IntentType.COMMENT, {'text': text})

    def is_file_operation(self, text: str) -> bool:
        """Check if text describes file operations"""
        file_indicators = ['open', 'read', 'write', 'save', 'load', 'close',
                         'file', 'document', 'csv', 'json', 'text file']
        
        return any(indicator in text.lower() for indicator in file_indicators)

    def parse_file_operation(self, text: str) -> Tuple[IntentType, Dict[str, Any]]:
        """Parse file operations"""
        # Open file
        open_match = re.search(r'(?:open|read|load)\s+(?:the\s+)?(?:file\s+)?(.+?)(?:\s+as\s+(\w+))?',
                             text, re.IGNORECASE)
        if open_match:
            filename = self.parse_value(open_match.group(1))
            var_name = open_match.group(2) if open_match.group(2) else 'file'
            
            mode = 'r'  # default read
            if 'write' in text.lower():
                mode = 'w'
            elif 'append' in text.lower():
                mode = 'a'
            
            return (IntentType.OPERATION, {
                'type': 'file_open',
                'filename': filename,
                'mode': mode,
                'variable': var_name
            })
        
        return (IntentType.COMMENT, {'text': text})

    def is_class_definition(self, text: str) -> bool:
        """Check if text describes a class definition"""
        class_indicators = ['class', 'object', 'type', 'create a class',
                          'define a class', 'inheritance', 'extends']
        
        return any(indicator in text.lower() for indicator in class_indicators)

    def parse_class_definition(self, text: str) -> Tuple[IntentType, Dict[str, Any]]:
        """Parse class definition"""
        class_match = re.search(r'(?:create|define)\s+(?:a\s+)?class\s+(?:called\s+)?(\w+)(?:\s+that\s+extends\s+(\w+))?',
                              text, re.IGNORECASE)
        
        if class_match:
            class_name = self.clean_identifier(class_match.group(1))
            parent = class_match.group(2) if class_match.group(2) else None
            
            self.context.current_class = class_name
            self.context.classes[class_name] = {
                'parent': parent,
                'methods': {},
                'attributes': []
            }
            
            return (IntentType.CLASS_DEFINITION, {
                'name': class_name,
                'parent': parent
            })
        
        return (IntentType.COMMENT, {'text': text})

    def extract_return_info(self, text: str) -> Optional[str]:
        """Extract return information from function description"""
        return_patterns = [
            r'returns?\s+(.+)',
            r'gives?\s+back\s+(.+)',
            r'outputs?\s+(.+)',
            r'results?\s+in\s+(.+)',
        ]
        
        for pattern in return_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None

    def extract_description(self, text: str) -> str:
        """Extract description from text"""
        # Remove common programming keywords to get the essence
        desc = text
        remove_patterns = [
            r'(?:create|define|make)\s+(?:a\s+)?(?:function|method)\s+(?:called\s+)?\w+',
            r'(?:that\s+)?(?:takes?|accepts?|with\s+parameters?).*?(?:returns?|$)',
            r'(?:returns?|gives?\s+back|outputs?).*$',
        ]
        
        for pattern in remove_patterns:
            desc = re.sub(pattern, '', desc, flags=re.IGNORECASE)
        
        desc = desc.strip()
        return desc if desc else None

    def parse_dict_value(self, text: str) -> str:
        """Parse dictionary from natural language"""
        # Check if already in Python dict format
        if text.strip().startswith('{') and text.strip().endswith('}'):
            return text.strip()
        
        # Try to parse key-value pairs
        # Look for patterns like "with key1 as value1 and key2 as value2"
        pairs_match = re.findall(r'(\w+)\s+(?:as|:|=)\s+(.+?)(?:\s+and\s+|$)', text, re.IGNORECASE)
        
        if pairs_match:
            items = []
            for key, value in pairs_match:
                parsed_value = self.parse_value(value.strip())
                items.append(f'"{key}": {parsed_value}')
            
            return '{' + ', '.join(items) + '}'
        
        return '{}'

    def parse_function_call(self, text: str) -> str:
        """Parse function call from text"""
        # Already looks like a function call
        if re.match(r'^\w+\([^)]*\)$', text):
            return text
        
        # Parse "call function with args" style
        call_match = re.search(r'(?:call|run|execute)\s+(\w+)\s+(?:with\s+)?(.+)', text, re.IGNORECASE)
        if call_match:
            func_name = call_match.group(1)
            args = self.parse_function_arguments(call_match.group(2))
            return f'{func_name}({args})'
        
        return text

    def parse_function_arguments(self, text: str) -> str:
        """Parse function arguments"""
        if not text or text.lower() in ['nothing', 'no arguments', 'none']:
            return ''
        
        # Split by commas and 'and'
        args = []
        parts = re.split(r',|\s+and\s+', text)
        
        for part in parts:
            part = part.strip()
            if part:
                # Check for named arguments
                if '=' in part:
                    args.append(part)
                else:
                    args.append(self.parse_value(part))
        
        return ', '.join(args)

    def is_property_access(self, text: str) -> bool:
        """Check if text is property access"""
        return bool(re.match(r'^[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+$', text))

    def infer_intent_from_context(self, text: str) -> Optional[Tuple[IntentType, Dict[str, Any]]]:
        """Infer intent from context when direct parsing fails"""
        # If we're in a function and see a plain statement, it might be an action
        if self.context.current_function:
            # Check if it's describing what the function does
            if any(word in text.lower() for word in ['calculates', 'computes', 'finds', 'gets']):
                return (IntentType.COMMENT, {'text': f'Function that {text}'})
        
        # If we have a subject and see an action word, might be a method call
        if self.context.last_subject:
            for action, operation in self.action_mappings.items():
                if action in text.lower():
                    return self.parse_operation(f'{action} {text} {self.context.last_subject}')
        
        # Default to comment
        return (IntentType.COMMENT, {'text': text})


class UltimateTranslator:
    """Main interface for the ultimate natural language to Python translator"""
    
    def __init__(self):
        self.translator = UltimateNaturalLanguageToPython()
        self.code_history = []
        self.conversation_mode = True
        
    def translate(self, text: str) -> str:
        """Translate natural language to Python code"""
        # Process the text
        code = self.translator.process_natural_language(text)
        
        # Store in history
        if code and code != "# Unable to parse input":
            self.code_history.append(code)
        
        return code
    
    def get_full_code(self) -> str:
        """Get all generated code"""
        return '\n\n'.join(self.code_history)
    
    def clear_context(self):
        """Clear the context and start fresh"""
        self.translator.context = Context()
        self.code_history = []
    
    def interactive_mode(self):
        """Run in interactive mode"""
        print("=== Ultimate Natural Language to Python Translator ===")
        print("Just speak naturally! I'll understand what you want to code.")
        print("\nCommands:")
        print("  'show all code' - Display all generated code")
        print("  'clear' - Start fresh")
        print("  'save to [filename]' - Save code to file")
        print("  'help' - Show examples")
        print("  'quit' - Exit")
        print("=" * 60)
        print("\nStart talking about what you want to code...\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break
                    
                elif user_input.lower() == 'show all code':
                    if self.code_history:
                        print("\n=== Generated Code ===")
                        print(self.get_full_code())
                        print("=" * 40)
                    else:
                        print("No code generated yet.")
                        
                elif user_input.lower() == 'clear':
                    self.clear_context()
                    print("Context cleared. Starting fresh!")
                    
                elif user_input.lower().startswith('save to'):
                    filename = user_input[8:].strip()
                    if not filename:
                        filename = 'generated_code.py'
                    self.save_code(filename)
                    
                elif user_input.lower() == 'help':
                    self.show_examples()
                    
                else:
                    # Translate the input
                    code = self.translate(user_input)
                    
                    if code and code != "# Unable to parse input":
                        print("\nGenerated:")
                        print("-" * 40)
                        print(code)
                        print("-" * 40)
                    else:
                        print("I'm not quite sure how to translate that. Could you rephrase?")
                        
            except KeyboardInterrupt:
                print("\nUse 'quit' to exit")
            except Exception as e:
                print(f"Error: {e}")
    
    def save_code(self, filename: str):
        """Save generated code to file"""
        if not self.code_history:
            print("No code to save.")
            return
        
        try:
            with open(filename, 'w') as f:
                f.write(self.get_full_code())
                f.write('\n\n# Generated by Ultimate Natural Language to Python Translator')
            print(f"Code saved to {filename}")
        except Exception as e:
            print(f"Error saving file: {e}")
    
    def show_examples(self):
        """Show natural language examples"""
        examples = """
=== Natural Language Examples ===

You can speak completely naturally! Here are some examples:

1. "I need a variable called age and set it to 25"
   "Let's have a score that starts at zero"
   "x should be 10"

2. "Create a function that calculates the area of a rectangle"
   "I want a method that takes two numbers and returns their sum"
   "Make a function called greet that says hello to someone"

3. "Go through each item in my shopping list and print it"
   "Keep asking for input until they say stop"
   "Repeat this 5 times"

4. "If the temperature is above 30, print that it's hot"
   "When x equals zero, return none"
   "Check if the user is logged in"

5. "I need to read data from a file called data.csv"
   "Save the results to output.txt"
   "Let's import pandas for data analysis"

6. "Create a class for a bank account with deposit and withdraw methods"
   "Make a person object with name and age"

7. "Add 'apples' to my shopping list"
   "Remove the first item from the list"
   "Sort the numbers in ascending order"

Just describe what you want in plain English!
"""
        print(examples)


# Main execution
if __name__ == "__main__":
    translator = UltimateTranslator()
    translator.interactive_mode()

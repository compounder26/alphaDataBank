"""
Alpha Expression Parser

This module parses alpha expressions to extract operators and datafields,
supporting both unique and nominal counting for analysis purposes.
"""
import re
import pandas as pd
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
import logging
import os
import sys

# Setup project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.bootstrap import setup_project_path
setup_project_path()
from database.schema import get_connection

logger = logging.getLogger(__name__)

class AlphaExpressionParser:
    """Parser for alpha expressions to extract operators and datafields."""
    
    # Symbol to operator name mapping for mathematical expressions
    OPERATOR_SYMBOLS = {
        '+': 'add',
        '-': 'subtract', 
        '*': 'multiply',
        '/': 'divide',
        '^': 'power',
        '<': 'less',
        '>': 'greater',
        '<=': 'less_equal',
        '>=': 'greater_equal',
        '==': 'equal',
        '!=': 'not_equal',
        '&&': 'and',
        '||': 'or'
    }
    
    def __init__(self, operators_file: str, operators_list: Optional[List[str]] = None,
                 available_datafields_list: Optional[List[str]] = None,
                 region_datafields_map: Optional[Dict[Tuple[str, str], bool]] = None):
        """
        Initialize parser with operators and datafields.

        Args:
            operators_file: Path to operators file (.txt or .json for dynamic data)
            operators_list: Optional list of operators available to user's tier
            available_datafields_list: Optional list of datafields available to user's tier
            region_datafields_map: Optional map of (region, datafield_id) -> available
        """
        if operators_list:
            self.available_operators = set(operators_list)
            logger.info(f"Using provided operators list: {len(self.available_operators)} operators")
        else:
            self.available_operators = self._load_operators(operators_file)

        # Load ALL operators for pattern matching to detect unavailable ones
        self.all_operators = self._load_all_operators(operators_file)
        logger.info(f"Loaded {len(self.all_operators)} total operators for pattern matching")
        logger.info(f"User has access to {len(self.available_operators)} operators")

        self.datafields = self._load_datafields_from_database()

        # Set available datafields for tier-based filtering
        if available_datafields_list:
            self.available_datafields = set(available_datafields_list)
            logger.info(f"Using provided datafields list: {len(self.available_datafields)} datafields")
        else:
            # If no explicit list provided, assume all loaded datafields are available
            self.available_datafields = set(self.datafields.keys())
            logger.info(f"No datafields filter provided - using all {len(self.available_datafields)} datafields")

        # Store region-datafield mapping for region-aware filtering
        self.region_datafields_map = region_datafields_map or {}
        if self.region_datafields_map:
            logger.info(f"Using region-datafield mapping with {len(self.region_datafields_map)} entries")
            # Get unique regions from the map
            regions = set(region for region, _ in self.region_datafields_map.keys())
            logger.info(f"Available regions: {sorted(regions)}")

        self.operator_pattern = self._create_operator_pattern()
        self.datafield_pattern = self._create_datafield_pattern()

        # Create symbol patterns
        self.symbol_patterns = self._create_symbol_patterns()
    
    def _filter_operators(self, operators_data: List[Dict]) -> List[Dict]:
        """
        Filter operators based on scope and category.
        
        Exclude operators that:
        1. Have category == "Special"
        2. Have scope containing only "COMBO" 
        3. Have scope containing only "SELECTION"
        
        Args:
            operators_data: List of operator dictionaries with name, scope, category
            
        Returns:
            Filtered list of operator dictionaries
        """
        filtered_operators = []
        excluded_count = 0
        excluded_reasons = {"Special": 0, "OnlyCOMBO": 0, "OnlySELECTION": 0}
        
        for op in operators_data:
            scope = op.get('scope', [])
            category = op.get('category', '')
            name = op.get('name', '')
            
            # Exclusion logic
            should_exclude = False
            reason = None
            
            # Check for Special category
            if category == "Special":
                should_exclude = True
                reason = "Special"
            # Check for operators with only COMBO scope
            elif scope == ["COMBO"]:
                should_exclude = True
                reason = "OnlyCOMBO"
            # Check for operators with only SELECTION scope
            elif scope == ["SELECTION"]:
                should_exclude = True
                reason = "OnlySELECTION"
            
            if should_exclude:
                excluded_count += 1
                excluded_reasons[reason] += 1
                logger.debug(f"Excluding operator '{name}' - {reason}")
            else:
                filtered_operators.append(op)
        
        logger.info(f"Operator filtering results:")
        logger.info(f"  Original operators: {len(operators_data)}")
        logger.info(f"  Excluded operators: {excluded_count}")
        logger.info(f"    Special category: {excluded_reasons['Special']}")
        logger.info(f"    Only COMBO scope: {excluded_reasons['OnlyCOMBO']}")
        logger.info(f"    Only SELECTION scope: {excluded_reasons['OnlySELECTION']}")
        logger.info(f"  Final operators: {len(filtered_operators)}")
        
        return filtered_operators

    def _load_all_operators(self, operators_file: str) -> Set[str]:
        """Load ALL operators from file without filtering (needed for exclusion detection)."""
        operators = set()
        try:
            # Make JSON detection more robust (case-insensitive, strip whitespace)
            operators_file_clean = operators_file.strip().lower()
            if operators_file_clean.endswith('.json'):
                # Handle dynamic JSON format
                import json
                with open(operators_file, 'r') as f:
                    data = json.load(f)

                if isinstance(data, dict) and 'operators' in data:
                    # Extract ALL operators from API response format WITHOUT filtering
                    original_operators = data['operators']
                    operator_list = [op['name'] for op in original_operators]
                elif isinstance(data, list):
                    # Direct list of operator names
                    operator_list = data
                else:
                    raise ValueError(f"Unsupported JSON format in {operators_file}")

                operators.update(operator_list)
                logger.info(f"Loaded {len(operators)} total operators from JSON cache (NO filtering)")

            else:
                # Handle traditional txt format
                with open(operators_file, 'r') as f:
                    content = f.read().strip()
                    # Split by comma and clean up whitespace
                    operator_list = [op.strip() for op in content.split(',')]
                    operators.update(operator_list)
                logger.info(f"Loaded {len(operators)} total operators from TXT file")

        except FileNotFoundError:
            logger.warning(f"Operators file not found: {operators_file}")
        except Exception as e:
            logger.error(f"Error loading all operators from {operators_file}: {e}")

        return operators

    def _load_operators(self, operators_file: str) -> Set[str]:
        """Load operators from operators.txt or operators.json file."""
        operators = set()
        try:
            # Make JSON detection more robust (case-insensitive, strip whitespace)
            operators_file_clean = operators_file.strip().lower()
            if operators_file_clean.endswith('.json'):
                # Handle dynamic JSON format
                import json
                with open(operators_file, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, dict) and 'operators' in data:
                    # Extract and filter operators from API response format
                    original_operators = data['operators']
                    filtered_operators = self._filter_operators(original_operators)
                    operator_list = [op['name'] for op in filtered_operators]
                elif isinstance(data, list):
                    # Direct list of operator names - assume no filtering needed for simple lists
                    operator_list = data
                else:
                    raise ValueError(f"Unsupported JSON format in {operators_file}")
                
                operators.update(operator_list)
                logger.info(f"Loaded {len(operators)} operators from JSON cache (after filtering)")
                
                # Validate reasonable operator count (should be ~100-300, not thousands)
                if len(operators) > 1000:
                    logger.warning(f"Suspicious operator count: {len(operators)} - possible JSON parsing as text!")
                    raise ValueError(f"Too many operators loaded ({len(operators)}), possible parsing error")
            else:
                # Handle traditional txt format
                with open(operators_file, 'r') as f:
                    content = f.read().strip()
                    # Split by comma and clean up whitespace
                    operator_list = [op.strip() for op in content.split(',')]
                    operators.update(operator_list)
                logger.info(f"Loaded {len(operators)} operators from TXT file")
                
                # Validate operator count for text files too
                if len(operators) > 1000:
                    logger.warning(f"Suspicious operator count from TXT file: {len(operators)}")
                    # Show sample operators to help debug
                    sample_ops = list(operators)[:10]
                    logger.warning(f"Sample operators: {sample_ops}")
                    raise ValueError(f"Too many operators loaded from TXT ({len(operators)}), check file format")
                
        except Exception as e:
            logger.error(f"Error loading operators from {operators_file}: {e}")
            raise
        return operators
    
    def _create_symbol_patterns(self) -> Dict[str, re.Pattern]:
        """Create regex patterns for mathematical symbols."""
        patterns = {}
        
        # Sort symbols by length to match longer operators first (e.g., <= before <)
        sorted_symbols = sorted(self.OPERATOR_SYMBOLS.keys(), key=len, reverse=True)
        
        for symbol in sorted_symbols:
            # Escape special regex characters
            escaped_symbol = re.escape(symbol)
            # Create pattern that matches the symbol with word boundaries or spaces
            # to avoid matching parts of other tokens
            patterns[symbol] = re.compile(r'\s*' + escaped_symbol + r'\s*')
        
        # Special pattern for ternary operator (condition ? true_value : false_value)
        patterns['ternary'] = re.compile(r'([^?]+)\?\s*([^:]+):\s*(.+)')
        
        return patterns
    
    def _load_datafields_from_database(self) -> Dict[str, Dict]:
        """Load datafields from database only."""
        datafields = {}
        
        try:
            from sqlalchemy import text
            
            db_engine = get_connection()
            with db_engine.connect() as connection:
                # Check if datafields table exists and has data
                result = connection.execute(text("SELECT COUNT(*) FROM datafields"))
                count = result.scalar()
                
                if count == 0:
                    logger.error("No datafields found in database. Please run with --renew to populate datafields.")
                    raise Exception("Datafields table is empty. Please populate with fresh data using --renew.")
                
                # Load from database - slim schema
                # Get unique datafield IDs (deduplicated across regions for parsing)
                query = text("""
                    SELECT DISTINCT ON (datafield_id) 
                           datafield_id, data_description, dataset_id, data_category, 
                           data_type, delay, region
                    FROM datafields
                    ORDER BY datafield_id, region
                """)
                
                result = connection.execute(query)
                
                for row in result:
                    datafield_id = row.datafield_id or ''
                    dataset_id = row.dataset_id or ''
                    data_category = row.data_category or 'unknown'
                    
                    # Use provided data_category or fallback to parsing dataset_id
                    if not data_category or data_category == 'unknown':
                        data_category = self._parse_dataset_category(dataset_id)
                    
                    datafields[datafield_id] = {
                        'dataset_id': dataset_id,
                        'data_category': data_category,
                        'data_type': row.data_type or '',
                        'description': row.data_description or ''
                    }
                
                # Also get region-specific count for statistics
                region_count_query = text("""
                    SELECT COUNT(*) as total_region_specific
                    FROM datafields
                """)
                region_result = connection.execute(region_count_query)
                self.total_region_specific_datafields = region_result.scalar()
                
                logger.info(f"Loaded {len(datafields)} datafields from database")
                return datafields
                
        except Exception as e:
            logger.error(f"Failed to load datafields from database: {e}")
            raise Exception("Cannot proceed without datafields data. Please ensure database is populated with --renew.")
    
    def _parse_dataset_category(self, dataset_id: str) -> str:
        """Parse dataset_id to extract meaningful category."""
        if not dataset_id:
            return 'Other'
        
        dataset_id_lower = dataset_id.lower()
        
        # Map common dataset patterns to categories
        if dataset_id_lower.startswith('fundamental'):
            return 'Fundamental'
        elif dataset_id_lower.startswith('analyst'):
            return 'Analyst'
        elif dataset_id_lower.startswith('pv'):
            return 'Price Volume'
        elif dataset_id_lower.startswith('macro'):
            return 'Macroeconomic'
        elif dataset_id_lower.startswith('altern'):
            return 'Alternative'
        elif dataset_id_lower.startswith('tech'):
            return 'Technical'
        elif 'sentiment' in dataset_id_lower:
            return 'Sentiment'
        elif 'news' in dataset_id_lower:
            return 'News'
        elif 'esg' in dataset_id_lower:
            return 'ESG'
        elif 'crypto' in dataset_id_lower:
            return 'Cryptocurrency'
        else:
            return 'Other'
    
    def _create_operator_pattern(self) -> re.Pattern:
        """Create regex pattern for matching operators."""
        # Use ALL operators (both available and unavailable) for pattern matching
        # This ensures we can detect unavailable operators for exclusion
        # Sort by length (longest first) to avoid partial matches
        sorted_operators = sorted(self.all_operators, key=len, reverse=True)
        # Escape special regex characters and create pattern
        escaped_operators = [re.escape(op) for op in sorted_operators]
        pattern = r'\b(' + '|'.join(escaped_operators) + r')\b'
        return re.compile(pattern, re.IGNORECASE)
    
    def _create_datafield_pattern(self) -> re.Pattern:
        """Create regex pattern for matching datafields."""
        # Sort by length (longest first) to avoid partial matches
        sorted_datafields = sorted(self.datafields.keys(), key=len, reverse=True)
        # Escape special regex characters and create pattern
        escaped_datafields = [re.escape(df) for df in sorted_datafields]
        pattern = r'\b(' + '|'.join(escaped_datafields) + r')\b'
        return re.compile(pattern, re.IGNORECASE)
    
    def _convert_symbols_to_operators(self, expression: str) -> Tuple[str, Set[str]]:
        """
        Convert mathematical symbols to operator names and track found symbols.
        
        Args:
            expression: The alpha expression string
            
        Returns:
            Tuple of (converted_expression, set_of_found_symbol_operators)
        """
        found_symbols = set()
        converted_expression = expression
        
        # Handle ternary operators first (they contain other symbols)
        ternary_matches = self.symbol_patterns['ternary'].findall(expression)
        for condition, true_val, false_val in ternary_matches:
            # Replace ternary with if_else function call
            ternary_pattern = re.escape(f"{condition.strip()}?{true_val.strip()}:{false_val.strip()}")
            replacement = f"if_else({condition.strip()}, {true_val.strip()}, {false_val.strip()})"
            converted_expression = re.sub(ternary_pattern, replacement, converted_expression)
            found_symbols.add('if_else')
        
        # Convert mathematical symbols to operator names
        for symbol, operator_name in self.OPERATOR_SYMBOLS.items():
            if self.symbol_patterns[symbol].search(converted_expression):
                # Only add if the operator is available to the user
                if operator_name in self.available_operators:
                    found_symbols.add(operator_name)
                    # Replace symbol with function notation for consistent parsing
                    # Note: We don't actually replace in the expression, just track the operator
        
        return converted_expression, found_symbols
    
    def _filter_operators_by_availability(self, operators: Set[str]) -> Set[str]:
        """
        Filter operators to only include those available to the user's tier.

        Args:
            operators: Set of operator names found in expression

        Returns:
            Set of operators that are actually available to the user
        """
        return operators & self.available_operators

    def _extract_variables(self, expression: str) -> Set[str]:
        """
        Extract all variable names defined in the expression.
        Variables are identified by the pattern: variable_name = expression
        Excludes named parameters inside function calls (k=1, ignore="NAN")
        """
        variables = set()
        # Pattern: variable_name = expression at statement level (not inside parentheses)
        # Split by semicolon to get individual statements
        statements = expression.split(';')
        for statement in statements:
            # Look for assignments at the statement level (not inside function calls)
            # Pattern: start of line/statement, variable name, =, but not ==
            var_pattern = r'^\s*([a-zA-Z_]\w*)\s*=(?!=)'
            match = re.match(var_pattern, statement.strip())
            if match:
                variables.add(match.group(1).lower())
        return variables

    def _extract_operators_strict(self, expression: str) -> Tuple[Set[str], List[str]]:
        """
        Extract all operators (functions with parentheses).
        Returns both recognized and unrecognized operators.
        """
        found_operators = set()
        unrecognized = []
        # Pattern: word followed by opening parenthesis
        op_pattern = r'\b([a-zA-Z_]\w*)\s*\('
        for match in re.finditer(op_pattern, expression.lower()):
            op_name = match.group(1)
            if op_name in self.available_operators:
                found_operators.add(op_name)
            else:
                # It looks like an operator but not in our available list
                unrecognized.append(op_name)
        return found_operators, unrecognized

    def _extract_datafields_strict(self, expression: str, variables: Set[str], region: Optional[str] = None) -> Tuple[Set[str], List[str]]:
        """
        Extract datafields (identifiers that aren't variables or operators).
        Returns both recognized and unrecognized datafields.

        Args:
            expression: The expression to parse
            variables: Set of user-defined variables to exclude
            region: Optional region context for region-aware datafield checking
        """
        found_datafields = set()
        unrecognized = []

        # Remove string literals first to avoid parsing their contents
        string_pattern = r'"[^"]*"|\'[^\']*\''
        expression_no_strings = re.sub(string_pattern, '""', expression)

        # Pattern: standalone identifiers NOT followed by (
        id_pattern = r'\b([a-zA-Z_]\w*)\b(?!\s*\()'

        # Keywords to skip
        KEYWORDS = {'if', 'else', 'then', 'and', 'or', 'not', 'true', 'false', 'null', 'nan', 'inf'}

        for match in re.finditer(id_pattern, expression_no_strings.lower()):
            identifier = match.group(1)

            # Skip if it's a variable, keyword, or number
            if identifier in variables or identifier in KEYWORDS or identifier.isdigit():
                continue

            # Skip if it's being assigned (check if followed by =)
            pos = match.end()
            if pos < len(expression_no_strings) and expression_no_strings[pos:].lstrip().startswith('='):
                continue

            # Check if it's an operator (some operators can be used without parentheses)
            if identifier in self.available_operators:
                continue

            # Check if it's a known datafield
            # Use region-aware checking if region is provided and map exists
            if region and self.region_datafields_map:
                # Check if (region, datafield_id) is available
                if (region, identifier) in self.region_datafields_map:
                    found_datafields.add(identifier)
                else:
                    # Unknown or unavailable in this region
                    unrecognized.append(identifier)
            else:
                # Fallback to simple checking
                if identifier in self.available_datafields:
                    found_datafields.add(identifier)
                else:
                    # Unknown identifier - likely an unavailable datafield or operator
                    unrecognized.append(identifier)

        return found_datafields, unrecognized

    def parse_expression_strict(self, expression: str, region: Optional[str] = None) -> Dict[str, Any]:
        """
        Strict inclusion-based parsing that only includes alphas where ALL components are available.

        This method:
        1. Extracts variables defined in the expression
        2. Identifies operators and checks availability
        3. Identifies datafields (excluding variables) and checks availability
        4. Excludes alpha if ANY unrecognized component is found

        Args:
            expression: The alpha expression to parse
            region: Optional region context for region-aware datafield checking

        Returns:
            Dict with operators, datafields, inclusion status, and unrecognized components
        """
        if not expression or pd.isna(expression):
            return {
                'operators': {'unique': set(), 'nominal': Counter()},
                'datafields': {'unique': set(), 'nominal': Counter()},
                'should_exclude': False,
                'exclusion_reasons': [],
                'unrecognized_components': []
            }

        # Clean expression
        clean_expr = ' '.join(expression.split())

        # Step 1: Extract variables defined in this expression
        variables = self._extract_variables(clean_expr)

        # Step 2: Convert mathematical symbols to operators and track them
        converted_expr, symbol_operators = self._convert_symbols_to_operators(clean_expr)

        # Check if any symbol operators are unavailable
        unavailable_symbols = []
        for op in symbol_operators:
            if op not in self.available_operators:
                unavailable_symbols.append(op)

        # Step 3: Extract function-style operators
        func_operators, unavailable_func_ops = self._extract_operators_strict(converted_expr)

        # Step 4: Extract datafields (excluding variables)
        datafields, unavailable_datafields = self._extract_datafields_strict(converted_expr, variables, region)

        # Step 5: Compile list of unrecognized components
        unrecognized = []
        if unavailable_symbols:
            unrecognized.extend([f"operator:{op}" for op in unavailable_symbols])
        if unavailable_func_ops:
            unrecognized.extend([f"operator:{op}" for op in unavailable_func_ops])
        if unavailable_datafields:
            unrecognized.extend([f"datafield:{df}" for df in unavailable_datafields])

        # Determine if we should exclude this alpha
        should_exclude = len(unrecognized) > 0
        exclusion_reasons = []
        if should_exclude:
            exclusion_reasons.append(f"Unrecognized components: {', '.join(unrecognized)}")

        # Step 6: Calculate occurrence counts for available components
        all_operators = func_operators | (symbol_operators & self.available_operators)
        operator_counter = Counter()

        # Count operator occurrences
        for op in all_operators:
            # Count function-style occurrences
            count = len(re.findall(rf'\b{re.escape(op)}\s*\(', clean_expr.lower()))
            if count > 0:
                operator_counter[op] = count
            # For symbol operators, count symbol occurrences
            elif op in symbol_operators:
                symbol = next((s for s, name in self.OPERATOR_SYMBOLS.items() if name == op), None)
                if symbol:
                    # Escape special regex characters in symbol
                    escaped_symbol = re.escape(symbol)
                    count = len(re.findall(escaped_symbol, clean_expr))
                    if count > 0:
                        operator_counter[op] = count

        # Count datafield occurrences
        datafield_counter = Counter()
        for df in datafields:
            count = len(re.findall(rf'\b{re.escape(df)}\b', clean_expr.lower()))
            if count > 0:
                datafield_counter[df] = count

        return {
            'operators': {
                'unique': all_operators,
                'nominal': operator_counter
            },
            'datafields': {
                'unique': datafields,
                'nominal': datafield_counter
            },
            'should_exclude': should_exclude,
            'exclusion_reasons': exclusion_reasons,
            'unrecognized_components': unrecognized,
            'variables_found': list(variables)
        }

    def parse_expression(self, expression: str, region: Optional[str] = None) -> Dict[str, Dict]:
        """
        Parse alpha expression using strict inclusion-based approach.

        This now uses the strict parser that only includes alphas where ALL
        components can be verified as available to the user's tier.

        Args:
            expression: Alpha expression string
            region: Optional region context for region-aware datafield checking

        Returns:
            Dictionary containing operators and datafields with unique/nominal counts,
            plus exclusion flag
        """
        # Use the new strict parsing method
        strict_result = self.parse_expression_strict(expression, region)

        # Map to the old format for backward compatibility
        return {
            'operators': strict_result['operators'],
            'datafields': strict_result['datafields'],
            'should_exclude': strict_result['should_exclude'],
            'exclusion_reasons': strict_result['exclusion_reasons'] if strict_result['exclusion_reasons'] else []
        }

    def parse_expression_old(self, expression: str) -> Dict[str, Dict]:
        """
        DEPRECATED: Old parsing method kept for reference.
        Use parse_expression() which now calls parse_expression_strict().
        """
        if not expression or pd.isna(expression):
            return {
                'operators': {'unique': set(), 'nominal': Counter()},
                'datafields': {'unique': set(), 'nominal': Counter()},
                'should_exclude': False
            }

        # Clean expression (remove extra whitespace)
        clean_expression = ' '.join(expression.split())

        # Convert mathematical symbols to operators and get symbol-based operators
        converted_expression, symbol_operators = self._convert_symbols_to_operators(clean_expression)

        # Find function-style operators (like ts_mean, abs, etc.)
        operator_matches = self.operator_pattern.findall(converted_expression)
        function_operators = set(match.lower() for match in operator_matches)

        # Combine function operators with symbol operators
        all_operators_found = function_operators | symbol_operators

        # Check for tier-based exclusion - exclude if alpha uses unavailable operators/datafields
        should_exclude = False
        exclusion_reasons = []

        # Check operators availability
        unavailable_operators = all_operators_found - self.available_operators
        if unavailable_operators:
            should_exclude = True
            exclusion_reasons.append(f"Unavailable operators: {', '.join(sorted(unavailable_operators))}")

        # Filter to only include available operators in the final result
        available_operators_found = all_operators_found & self.available_operators

        # Count occurrences (for nominal counting)
        operator_counter = Counter()
        for op in available_operators_found:
            # Count function-style occurrences
            function_count = sum(1 for match in operator_matches if match.lower() == op)
            # For symbol operators, count symbol occurrences
            if op in symbol_operators:
                symbol = next((s for s, name in self.OPERATOR_SYMBOLS.items() if name == op), None)
                if symbol and self.symbol_patterns[symbol].search(clean_expression):
                    symbol_count = len(self.symbol_patterns[symbol].findall(clean_expression))
                    operator_counter[op] += symbol_count
            operator_counter[op] += function_count

        unique_operators = available_operators_found

        # Find datafields
        datafield_matches = self.datafield_pattern.findall(clean_expression)
        all_datafields_found = set(match.lower() for match in datafield_matches)

        # Check datafields availability
        unavailable_datafields = all_datafields_found - self.available_datafields
        if unavailable_datafields:
            should_exclude = True
            exclusion_reasons.append(f"Unavailable datafields: {', '.join(sorted(unavailable_datafields))}")

        # Filter to only include available datafields in the final result
        available_datafields_found = all_datafields_found & self.available_datafields

        # Count occurrences for available datafields only
        datafield_counter = Counter()
        for df in available_datafields_found:
            # Count how many times this datafield appears
            datafield_counter[df] = clean_expression.lower().count(df.lower())

        unique_datafields = available_datafields_found

        return {
            'operators': {
                'unique': unique_operators,
                'nominal': operator_counter
            },
            'datafields': {
                'unique': unique_datafields,
                'nominal': datafield_counter
            },
            'should_exclude': should_exclude,
            'exclusion_reasons': exclusion_reasons
        }
    
    def analyze_alpha_batch(self, alphas_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Analyze a batch of alphas for operator and datafield usage.
        
        Args:
            alphas_df: DataFrame containing alpha data with 'alpha_id' and 'code' columns
            
        Returns:
            Dictionary containing aggregated analysis results
        """
        results = {
            'operators': {
                'unique_usage': defaultdict(set),  # operator -> set of alpha_ids using it
                'nominal_usage': defaultdict(int),  # operator -> total count across all alphas
                'alpha_breakdown': {}  # alpha_id -> {operator -> count}
            },
            'datafields': {
                'unique_usage': defaultdict(set),  # datafield -> set of alpha_ids using it
                'nominal_usage': defaultdict(int),  # datafield -> total count across all alphas
                'alpha_breakdown': {},  # alpha_id -> {datafield -> count}
                'by_category': defaultdict(lambda: defaultdict(set))  # category -> datafield -> set of alpha_ids
            },
            'alpha_metadata': {},  # alpha_id -> parsed data for storage
            'excluded_alphas': set(),  # set of alpha_ids that were excluded
            'exclusion_stats': {
                'total_processed': 0,
                'total_excluded': 0,
                'total_included': 0
            }
        }
        
        for _, row in alphas_df.iterrows():
            alpha_id = row['alpha_id']
            expression = row.get('code', '')
            region = row.get('region_name', None)  # Get region if available

            # Parse expression with region context for region-aware datafield checking
            parsed = self.parse_expression(expression, region)

            # Update statistics
            results['exclusion_stats']['total_processed'] += 1

            # Check if alpha should be excluded based on tier restrictions
            should_exclude = parsed.get('should_exclude', False)
            exclusion_reasons = parsed.get('exclusion_reasons', [])

            if should_exclude:
                results['exclusion_stats']['total_excluded'] += 1
                results['excluded_alphas'].add(alpha_id)

                # Store metadata for excluded alpha
                results['alpha_metadata'][alpha_id] = {
                    'operators_unique': list(parsed['operators']['unique']),
                    'operators_nominal': dict(parsed['operators']['nominal']),
                    'datafields_unique': list(parsed['datafields']['unique']),
                    'datafields_nominal': dict(parsed['datafields']['nominal']),
                    'excluded': True,
                    'exclusion_reason': '; '.join(exclusion_reasons)
                }

                # Skip processing this alpha for usage statistics
                continue
            else:
                results['exclusion_stats']['total_included'] += 1

                # Store metadata for included alpha
                results['alpha_metadata'][alpha_id] = {
                    'operators_unique': list(parsed['operators']['unique']),
                    'operators_nominal': dict(parsed['operators']['nominal']),
                    'datafields_unique': list(parsed['datafields']['unique']),
                    'datafields_nominal': dict(parsed['datafields']['nominal']),
                    'excluded': False
                }
            
            # Process operators
            results['operators']['alpha_breakdown'][alpha_id] = dict(parsed['operators']['nominal'])
            
            for operator in parsed['operators']['unique']:
                results['operators']['unique_usage'][operator].add(alpha_id)
            
            for operator, count in parsed['operators']['nominal'].items():
                results['operators']['nominal_usage'][operator] += count
            
            # Process datafields
            results['datafields']['alpha_breakdown'][alpha_id] = dict(parsed['datafields']['nominal'])
            
            for datafield in parsed['datafields']['unique']:
                results['datafields']['unique_usage'][datafield].add(alpha_id)
                # Get category information
                if datafield in self.datafields:
                    category = self.datafields[datafield].get('data_category', 'unknown')
                    results['datafields']['by_category'][category][datafield].add(alpha_id)
            
            for datafield, count in parsed['datafields']['nominal'].items():
                results['datafields']['nominal_usage'][datafield] += count
        
        # Convert defaultdicts and sets to regular dicts/lists for JSON serialization
        return self._serialize_results(results)
    
    def _serialize_results(self, results: Dict) -> Dict:
        """Convert results to JSON-serializable format."""
        serialized = {
            'operators': {
                'unique_usage': {op: list(alphas) for op, alphas in results['operators']['unique_usage'].items()},
                'nominal_usage': dict(results['operators']['nominal_usage']),
                'alpha_breakdown': results['operators']['alpha_breakdown']
            },
            'datafields': {
                'unique_usage': {df: list(alphas) for df, alphas in results['datafields']['unique_usage'].items()},
                'nominal_usage': dict(results['datafields']['nominal_usage']),
                'alpha_breakdown': results['datafields']['alpha_breakdown'],
                'by_category': {
                    cat: {df: list(alphas) for df, alphas in datafields.items()}
                    for cat, datafields in results['datafields']['by_category'].items()
                }
            },
            'alpha_metadata': results['alpha_metadata'],
            'excluded_alphas': list(results['excluded_alphas']),
            'exclusion_stats': results['exclusion_stats']
        }
        return serialized

    def get_available_operators(self) -> Set[str]:
        """Get set of operators available to user's tier."""
        return self.available_operators.copy()

    def get_available_datafields(self) -> Set[str]:
        """Get set of datafields available to user's tier."""
        return self.available_datafields.copy()

    def update_available_operators(self, operators_list: List[str]):
        """Update the available operators list (useful for dynamic tier changes)."""
        self.available_operators = set(operators_list)
        logger.info(f"Updated available operators: {len(self.available_operators)} operators")

    def update_available_datafields(self, datafields_list: List[str]):
        """Update the available datafields list (useful for dynamic tier changes)."""
        self.available_datafields = set(datafields_list)
        logger.info(f"Updated available datafields: {len(self.available_datafields)} datafields")

    def check_alpha_accessibility(self, expression: str) -> Dict[str, Any]:
        """
        Check if an alpha expression is accessible given user's tier.

        Args:
            expression: Alpha expression string

        Returns:
            Dictionary with accessibility info:
            {
                'is_accessible': bool,
                'unavailable_operators': List[str],
                'unavailable_datafields': List[str],
                'exclusion_reasons': List[str]
            }
        """
        parsed = self.parse_expression(expression)

        return {
            'is_accessible': not parsed.get('should_exclude', False),
            'unavailable_operators': [],  # Would need to track these separately
            'unavailable_datafields': [],  # Would need to track these separately
            'exclusion_reasons': parsed.get('exclusion_reasons', [])
        }
    
    def get_top_operators(self, analysis_results: Dict, top_n: int = 20, 
                         count_type: str = 'unique') -> List[Tuple[str, int]]:
        """
        Get top operators by usage.
        
        Args:
            analysis_results: Results from analyze_alpha_batch
            top_n: Number of top operators to return
            count_type: 'unique' or 'nominal'
            
        Returns:
            List of (operator, count) tuples
        """
        if count_type == 'unique':
            usage_data = {op: len(alphas) for op, alphas in analysis_results['operators']['unique_usage'].items()}
        else:  # nominal
            usage_data = analysis_results['operators']['nominal_usage']
        
        return sorted(usage_data.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    def get_top_datafields(self, analysis_results: Dict, top_n: int = 20, 
                          count_type: str = 'unique') -> List[Tuple[str, int]]:
        """
        Get top datafields by usage.
        
        Args:
            analysis_results: Results from analyze_alpha_batch
            top_n: Number of top datafields to return
            count_type: 'unique' or 'nominal'
            
        Returns:
            List of (datafield, count) tuples
        """
        if count_type == 'unique':
            usage_data = {df: len(alphas) for df, alphas in analysis_results['datafields']['unique_usage'].items()}
        else:  # nominal
            usage_data = analysis_results['datafields']['nominal_usage']
        
        return sorted(usage_data.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    def get_category_breakdown(self, analysis_results: Dict) -> Dict[str, int]:
        """
        Get breakdown of datafield usage by category.
        
        Args:
            analysis_results: Results from analyze_alpha_batch
            
        Returns:
            Dictionary of category -> unique alpha count
        """
        category_breakdown = {}
        for category, datafields in analysis_results['datafields']['by_category'].items():
            all_alphas = set()
            for alphas in datafields.values():
                all_alphas.update(alphas)
            category_breakdown[category] = len(all_alphas)
        
        return category_breakdown
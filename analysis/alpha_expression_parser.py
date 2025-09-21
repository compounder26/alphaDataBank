"""
Alpha Expression Parser

This module parses alpha expressions to extract operators and datafields,
supporting both unique and nominal counting for analysis purposes.
"""
import re
import pandas as pd
from typing import Dict, List, Set, Tuple, Optional
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
    
    def __init__(self, operators_file: str, operators_list: Optional[List[str]] = None):
        """
        Initialize parser with operators and datafields.
        
        Args:
            operators_file: Path to operators file (.txt or .json for dynamic data)
            operators_list: Optional list of operators to use directly (overrides file)
        """
        if operators_list:
            self.available_operators = set(operators_list)
            logger.info(f"Using provided operators list: {len(self.available_operators)} operators")
        else:
            self.available_operators = self._load_operators(operators_file)
        
        # Use ONLY the available operators for pattern matching
        # No fallback to static files - only match operators accessible to user's tier
        self.all_operators = self.available_operators.copy()
        logger.info(f"Using {len(self.all_operators)} user-accessible operators for pattern matching")
        
        self.datafields = self._load_datafields_from_database()
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
    
    def parse_expression(self, expression: str) -> Dict[str, Dict]:
        """
        Parse alpha expression to extract operators and datafields.
        
        Args:
            expression: Alpha expression string
            
        Returns:
            Dictionary containing operators and datafields with unique/nominal counts, 
            plus exclusion flag
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
        all_operators = function_operators | symbol_operators
        
        # Since we only match user-accessible operators, no exclusion needed
        # All found operators are guaranteed to be available
        should_exclude = False
        
        # All found operators are already available (no filtering needed)
        available_operators_found = all_operators
        
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
        datafield_counter = Counter(match.lower() for match in datafield_matches)
        unique_datafields = set(datafield_counter.keys())
        
        return {
            'operators': {
                'unique': unique_operators,
                'nominal': operator_counter
            },
            'datafields': {
                'unique': unique_datafields,
                'nominal': datafield_counter
            },
            'should_exclude': should_exclude
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
            
            # Parse expression
            parsed = self.parse_expression(expression)
            
            # Update statistics
            results['exclusion_stats']['total_processed'] += 1
            
            # All alphas are included since we only match accessible operators
            results['exclusion_stats']['total_included'] += 1
            
            # Store metadata for this alpha
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
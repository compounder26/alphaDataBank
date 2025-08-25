"""
Alpha Expression Parser

This module parses alpha expressions to extract operators and datafields,
supporting both unique and nominal counting for analysis purposes.
"""
import re
import pandas as pd
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)

class AlphaExpressionParser:
    """Parser for alpha expressions to extract operators and datafields."""
    
    def __init__(self, operators_file: str, datafields_file: str):
        """
        Initialize parser with operators and datafields.
        
        Args:
            operators_file: Path to operators.txt file
            datafields_file: Path to all_datafields_comprehensive.csv file
        """
        self.operators = self._load_operators(operators_file)
        self.datafields = self._load_datafields(datafields_file)
        self.operator_pattern = self._create_operator_pattern()
        self.datafield_pattern = self._create_datafield_pattern()
    
    def _load_operators(self, operators_file: str) -> Set[str]:
        """Load operators from operators.txt file."""
        operators = set()
        try:
            with open(operators_file, 'r') as f:
                content = f.read().strip()
                # Split by comma and clean up whitespace
                operator_list = [op.strip() for op in content.split(',')]
                operators.update(operator_list)
            logger.info(f"Loaded {len(operators)} operators")
        except Exception as e:
            logger.error(f"Error loading operators from {operators_file}: {e}")
            raise
        return operators
    
    def _load_datafields(self, datafields_file: str) -> Dict[str, Dict]:
        """Load datafields from CSV file."""
        datafields = {}
        try:
            df = pd.read_csv(datafields_file)
            for _, row in df.iterrows():
                datafield_id = row['datafield_id']
                dataset_id = row.get('dataset_id', '')
                
                # Parse dataset_id to create meaningful category
                category = self._parse_dataset_category(dataset_id)
                
                datafields[datafield_id] = {
                    'dataset_id': dataset_id,
                    'data_category': category,
                    'data_type': row.get('data_type', '')
                }
            logger.info(f"Loaded {len(datafields)} datafields")
        except Exception as e:
            logger.error(f"Error loading datafields from {datafields_file}: {e}")
            raise
        return datafields
    
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
        # Sort by length (longest first) to avoid partial matches
        sorted_operators = sorted(self.operators, key=len, reverse=True)
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
    
    def parse_expression(self, expression: str) -> Dict[str, Dict]:
        """
        Parse alpha expression to extract operators and datafields.
        
        Args:
            expression: Alpha expression string
            
        Returns:
            Dictionary containing operators and datafields with unique/nominal counts
        """
        if not expression or pd.isna(expression):
            return {
                'operators': {'unique': set(), 'nominal': Counter()},
                'datafields': {'unique': set(), 'nominal': Counter()}
            }
        
        # Clean expression (remove extra whitespace)
        clean_expression = ' '.join(expression.split())
        
        # Find operators
        operator_matches = self.operator_pattern.findall(clean_expression)
        operator_counter = Counter(match.lower() for match in operator_matches)
        unique_operators = set(operator_counter.keys())
        
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
            }
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
            'alpha_metadata': {}  # alpha_id -> parsed data for storage
        }
        
        for _, row in alphas_df.iterrows():
            alpha_id = row['alpha_id']
            expression = row.get('code', '')
            
            # Parse expression
            parsed = self.parse_expression(expression)
            
            # Store metadata for this alpha
            results['alpha_metadata'][alpha_id] = {
                'operators_unique': list(parsed['operators']['unique']),
                'operators_nominal': dict(parsed['operators']['nominal']),
                'datafields_unique': list(parsed['datafields']['unique']),
                'datafields_nominal': dict(parsed['datafields']['nominal'])
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
            'alpha_metadata': results['alpha_metadata']
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
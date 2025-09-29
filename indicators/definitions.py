import pandas as pd
import numpy as np
import csv
import os
from typing import Dict, List, Any

class IndicatorDefinitions:
    """
    Loads and manages technical indicator definitions from CSV file
    """
    
    def __init__(self, csv_file_path: str = None):
        if csv_file_path is None:
            csv_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'technical_indicators_only.csv')
        self.csv_file_path = csv_file_path
        self.indicators = {}
        self.load_indicators()
    
    def load_indicators(self):
        """Load indicator definitions from CSV file"""
        try:
            with open(self.csv_file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    indicator_name = row['Indicator'].strip()
                    self.indicators[indicator_name] = {
                        'name': indicator_name,
                        'category': row['Category'].strip(),
                        'required_inputs': [inp.strip() for inp in row['Required Inputs'].split(',')],
                        'formula': row['Formula / Calculation'].strip(),
                        'must_keep': row['Must Keep (Not in RFE)'].strip().lower() == 'yes',
                        'rfe_eligible': row['RFE Eligible'].strip().lower() == 'yes',
                        'prerequisite_for': row['Prerequisite For'].strip(),
                        'parameters': self._parse_parameters(row['Parameters'].strip()),
                        'outputs': row['Outputs'].strip()
                    }
            print(f"Loaded {len(self.indicators)} indicator definitions")
        except Exception as e:
            print(f"Error loading indicators: {e}")
            raise
    
    def _parse_parameters(self, param_string: str) -> Dict[str, Any]:
        """Parse parameter string into dictionary"""
        if not param_string:
            return {}
        
        params = {}
        for param in param_string.split(';'):
            if '=' in param:
                key, value = param.split('=', 1)
                # Try to convert to appropriate type
                try:
                    if '.' in value:
                        params[key.strip()] = float(value.strip())
                    else:
                        params[key.strip()] = int(value.strip())
                except ValueError:
                    params[key.strip()] = value.strip()
        return params
    
    def get_indicator(self, name: str) -> Dict[str, Any]:
        """Get indicator definition by name"""
        return self.indicators.get(name, {})
    
    def get_indicators_by_category(self, category: str) -> Dict[str, Dict[str, Any]]:
        """Get all indicators in a specific category"""
        return {name: info for name, info in self.indicators.items() 
                if info['category'].lower() == category.lower()}
    
    def get_rfe_eligible_indicators(self) -> Dict[str, Dict[str, Any]]:
        """Get indicators eligible for RFE selection"""
        return {name: info for name, info in self.indicators.items() 
                if info['rfe_eligible']}
    
    def get_required_indicators(self) -> Dict[str, Dict[str, Any]]:
        """Get indicators that must always be included (not eligible for RFE removal)"""
        return {name: info for name, info in self.indicators.items() 
                if info['must_keep']}
    
    def get_all_indicators(self) -> Dict[str, Dict[str, Any]]:
        """Get all loaded indicators"""
        return self.indicators.copy()
    
    def get_prerequisites(self) -> List[str]:
        """Get list of prerequisite indicators"""
        prereqs = []
        for name, info in self.indicators.items():
            if info['category'].lower() == 'prereq':
                prereqs.append(name)
        return prereqs
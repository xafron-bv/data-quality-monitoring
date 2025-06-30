import pandas as pd
import re
import time
from typing import List, Tuple, Optional, Dict

class MaterialValidator:
    """
    A high-speed, deterministic validator for material composition strings.

    This validator operates on the fundamental rule that the compositions of any
    given "part" of a material must sum to 100. It makes no prior assumptions
    about the format, keywords, or delimiters used.
    """
    def __init__(self):
        # A compiled regex for high-speed tokenization.
        # It finds:
        # 1. Floating-point or integer numbers.
        # 2. Sequences of letters (words).
        # 3. Any single character that isn't a letter, number, or whitespace (delimiters/symbols).
        self.tokenizer_regex = re.compile(r'(\d+(?:\.\d+)?|[a-zA-Z]+|[^a-zA-Z0-9\s])')
        self.validation_cache: Dict[str, bool] = {}

    def _tokenize(self, s: str) -> List[str]:
        """Tokenizes a raw string into its fundamental components."""
        return self.tokenizer_regex.findall(s)

    def _validate_part(self, tokens: List[str]) -> bool:
        """
        Checks if a given sequence of tokens represents a valid part
        by summing its numerical components.
        """
        # Quick check: if there are no numbers, it can't be a valid part.
        if not any(re.fullmatch(r'\d+(?:\.\d+)?', t) for t in tokens):
            return False
            
        total = sum(float(t) for t in tokens if re.fullmatch(r'\d+(?:\.\d+)?', t))
        
        # Use a small tolerance for floating-point comparisons.
        return abs(total - 100.0) < 1e-6

    def _find_valid_partition(self, tokens: List[str]) -> Optional[List[List[str]]]:
        """
        Hypothesizes and tests different ways to partition the token list
        to satisfy the 100-sum rule for all sub-parts.
        """
        # Hypothesis 1 (Most Common): The entire string is a single part.
        if self._validate_part(tokens):
            return [tokens]

        # Hypothesis 2: The string is multi-part, split by common delimiters.
        # This is assumption-free as it only succeeds if the resulting parts are valid.
        potential_delimiters = ['//', '|']
        for delim in potential_delimiters:
            if delim in tokens:
                try:
                    parts = []
                    current_part = []
                    for token in tokens:
                        if token == delim:
                            parts.append(current_part)
                            current_part = []
                        else:
                            current_part.append(token)
                    parts.append(current_part)

                    if all(self._validate_part(p) for p in parts):
                        return parts
                except Exception:
                    continue # Parsing failed, this hypothesis is wrong.

        # Hypothesis 3: The string is multi-part, split by "KEYWORD:" patterns.
        # This is a very common structure in product data.
        try:
            parts = []
            current_part = []
            i = 0
            while i < len(tokens):
                # Check for the WORD followed by ':' pattern
                if (i + 1 < len(tokens) and 
                    re.fullmatch(r'[a-zA-Z]+', tokens[i]) and 
                    tokens[i+1] == ':'):
                    
                    if current_part: # If we have a part accumulated
                        parts.append(current_part)
                    current_part = [tokens[i], tokens[i+1]] # Start new part with keyword
                    i += 2
                else:
                    current_part.append(tokens[i])
                    i += 1
            if current_part:
                parts.append(current_part)

            if len(parts) > 1 and all(self._validate_part(p) for p in parts):
                return parts
        except Exception:
            pass

        # If no valid partition could be found
        return None

    def validate(self, material_string: str) -> bool:
        """
        Validates a single material string. Uses a cache for speed.
        """
        if material_string in self.validation_cache:
            return self.validation_cache[material_string]

        tokens = self._tokenize(material_string)
        if not tokens:
            self.validation_cache[material_string] = False
            return False
            
        result = self._find_valid_partition(tokens) is not None
        self.validation_cache[material_string] = result
        return result

    def bulk_validate(self, material_list: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validates a list of material strings and separates them into valid and invalid lists.
        """
        valid_materials = []
        invalid_materials = []
        for material in material_list:
            if self.validate(material):
                valid_materials.append(material)
            else:
                invalid_materials.append(material)
        return valid_materials, invalid_materials


if __name__ == '__main__':
    # --- 1. Load Data ---
    try:
        df = pd.read_csv('esqualo_2022_fall.csv')
        materials_list = df['material'].dropna().unique().tolist()
        print(f"Loaded {len(materials_list)} unique material strings for analysis from 'esqualo_2022_fall.csv'.\n")
    except FileNotFoundError:
        print("Error: 'esqualo_2022_fall.csv' not found.")
        materials_list = []

    if materials_list:
        # --- 2. Initialize and Run Validator ---
        validator = MaterialValidator()
        
        start_time = time.time()
        valid, invalid = validator.bulk_validate(materials_list)
        end_time = time.time()
        
        # --- 3. Report the Results ---
        print(f"Validation completed in {end_time - start_time:.4f} seconds.\n")
        
        print("="*50)
        print("          VALID Material Strings")
        print("="*50)
        if not valid:
            print("No valid materials found.")
        else:
            print(f"Found {len(valid)} valid material formats:\n")
            for item in sorted(valid):
                print(f"  - '{item}'")

        print("\n\n" + "="*50)
        print("         INVALID Material Strings")
        print("="*50)
        if not invalid:
            print("No invalid materials found.")
        else:
            print(f"Found {len(invalid)} materials that failed validation:\n")
            for item in sorted(invalid):
                print(f"  - '{item}'")
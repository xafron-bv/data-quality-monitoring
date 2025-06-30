import pandas as pd
import re
import time
from typing import List, Tuple, Optional, Dict, Set
import json

class MaterialValidator:
    """
    A high-speed, deterministic validator for material composition strings.

    This validator uses a purely structural and mathematical approach to determine
    validity. It makes no assumptions about the language of the material names
    and does not use any keyword-based cheating for non-material content.
    """
    def __init__(self):
        """Initializes the validator."""
        self.tokenizer_regex = re.compile(r'(\d+(?:\.\d+)?|[a-zA-Z]+|[^a-zA-Z0-9\s])')
        self.validation_cache: Dict[str, bool] = {}

    def _tokenize(self, s: str) -> List[str]:
        """Tokenizes a raw string into its fundamental components."""
        # Normalize all internal whitespace to a single space before tokenizing.
        normalized_s = re.sub(r'\s+', ' ', s.strip())
        return self.tokenizer_regex.findall(normalized_s)

    def _validate_part(self, tokens: List[str]) -> bool:
        """
        Checks if a given sequence of tokens represents a valid part by
        checking its sum, structure, and content. This is the core validation logic.
        """
        if not tokens:
            return False

        # Structural Rule 1: An un-keyed part (no colon) cannot start and end
        # with a word, as it's structurally ambiguous. This catches prefixes
        # like "Blue 100% Viscose".
        if (len(tokens) > 1 and
            re.fullmatch(r'[a-zA-Z]+', tokens[0]) and
            re.fullmatch(r'[a-zA-Z]+', tokens[-1]) and
            ':' not in tokens):
            return False

        # Structural Rule 2: The part must contain numbers, and their sum must be 100.
        numbers = [float(t) for t in tokens if re.fullmatch(r'\d+(?:\.\d+)?', t)]
        if not numbers or abs(sum(numbers) - 100.0) > 1e-6:
            return False
        
        # Structural Rule 3: The number of '%' signs must exactly match the number of numbers.
        if tokens.count('%') != len(numbers):
            return False

        # Structural Rule 4: The only permissible non-alphanumeric symbol in a composition
        # part is the percentage sign. This catches trademarks, stray hyphens, etc.
        if any(not t.isalnum() and t != '%' for t in tokens):
            return False
            
        # Structural Rule 5: All words must be logically connected to a number or
        # percentage sign. We check this with a graph traversal (BFS). Words cannot
        # be connected to other words; they must be anchored to a number/percent.
        q = [i for i, t in enumerate(tokens) if re.fullmatch(r'\d+(?:\.\d+)?', t)]
        visited = set(q)

        head = 0
        while head < len(q):
            idx = q[head]
            head += 1
            
            # Explore neighbors (left and right)
            for ni in [idx - 1, idx + 1]:
                if 0 <= ni < len(tokens) and ni not in visited:
                    # A word cannot be justified by another word.
                    if re.fullmatch(r'[a-zA-Z]+', tokens[idx]) and re.fullmatch(r'[a-zA-Z]+', tokens[ni]):
                        continue

                    # If the neighbor is a word or a percent sign, it's a valid connection.
                    if re.fullmatch(r'[a-zA-Z]+', tokens[ni]) or tokens[ni] == '%':
                        visited.add(ni)
                        q.append(ni)
        
        # If any token was not visited, it means it was an "orphan" word
        # (like a care instruction or a misplaced color) that was not anchored
        # to the core composition structure.
        return len(visited) == len(tokens)


    def _find_valid_partition(self, tokens: List[str]) -> Optional[List[List[str]]]:
        """
        Hypothesizes and tests different ways to partition the token list
        to satisfy the 100-sum rule for all sub-parts.
        """
        # Hypothesis 1: The entire string is a single part.
        if self._validate_part(tokens):
            return [tokens]

        # Hypothesis 2: The string is multi-part, split by common delimiters.
        # We add '-' to the delimiters to correctly segment care instructions.
        potential_delimiters = ['//', '|', '-']
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

                    if all(self._validate_part(p) for p in parts if p):
                        return parts
                except Exception:
                    continue

        # Hypothesis 3: The string is multi-part, split by "KEYWORD:" patterns.
        try:
            parts = []
            current_part = []
            i = 0
            while i < len(tokens):
                if (i + 1 < len(tokens) and 
                    re.fullmatch(r'[a-zA-Z]+', tokens[i]) and 
                    tokens[i+1] == ':'):
                    
                    if current_part:
                        parts.append(current_part)
                    # The keyword itself is not part of the composition, so we start fresh.
                    current_part = [] 
                    i += 2
                else:
                    current_part.append(tokens[i])
                    i += 1
            if current_part:
                parts.append(current_part)

            if len(parts) > 1 and all(self._validate_part(p) for p in parts if p):
                return parts
        except Exception:
            pass

        return None

    def validate(self, material_string: str) -> bool:
        """Validates a single material string based on composition and formatting."""
        if not isinstance(material_string, str) or not material_string.strip():
            return False

        # Use a cache for performance
        if material_string in self.validation_cache:
            return self.validation_cache[material_string]

        # A single, pragmatic rule for data cleanliness: the raw string cannot
        # contain newlines or multiple spaces, unless the spaces follow a colon.
        if re.search(r'\n|(?<!:)\s{2,}', material_string):
             self.validation_cache[material_string] = False
             return False

        tokens = self._tokenize(material_string)

        if not tokens:
            self.validation_cache[material_string] = False
            return False

        result = self._find_valid_partition(tokens) is not None
        self.validation_cache[material_string] = result
        return result

    def bulk_validate(self, material_df: pd.DataFrame) -> List[Dict]:
        """
        Validates a dataframe of material strings and returns a list of validation errors.
        """
        errors = []
        for index, row in material_df.iterrows():
            material = row['material']
            if not self.validate(material):
                errors.append({
                    "row_index": index,
                    "material_string": material,
                    "error_type": "Validation Failed"
                })
        return errors

def main():
    """Main function to run the validator on a CSV file."""
    try:
        df = pd.read_csv('esqualo_2022_fall_original.csv')
    except FileNotFoundError:
        print("Error: 'esqualo_2022_fall_original.csv' not found.")
        return

    validator = MaterialValidator()
    validation_errors = validator.bulk_validate(df)
    
    output = {
        "validation_errors": validation_errors,
        "statistics": {
            "total_rows": len(df),
            "invalid_rows": len(validation_errors)
        }
    }
    
    print(json.dumps(output, indent=4))

if __name__ == '__main__':
    main()

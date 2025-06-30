import pandas as pd
import re
import time
from typing import List, Tuple, Optional, Dict
import json

class MaterialValidator:
    """
    A high-speed, deterministic validator for material composition strings.

    This validator operates on the fundamental rule that the compositions of any
    given "part" of a material must sum to 100. It also enforces several
    formatting rules to reject common data entry errors.
    """
    def __init__(self):
        # A compiled regex for high-speed tokenization.
        self.tokenizer_regex = re.compile(r'(\d+(?:\.\d+)?|[a-zA-Z]+|[^a-zA-Z0-9\s])')
        self.validation_cache: Dict[str, bool] = {}

    def _preprocess(self, s: str) -> str:
        """Cleans and normalizes the material string before tokenization."""
        if not isinstance(s, str):
            return ""
        # Strip leading/trailing whitespace
        s = s.strip()
        # Normalize all whitespace (newlines, tabs, multiple spaces) to a single space
        s = re.sub(r'\s+', ' ', s)
        # Remove symbols that don't affect composition validity
        s = re.sub(r'[™®]', '', s)
        return s

    def _tokenize(self, s: str) -> List[str]:
        """Tokenizes a raw string into its fundamental components."""
        return self.tokenizer_regex.findall(s)

    def _validate_part(self, tokens: List[str]) -> bool:
        """
        Checks if a given sequence of tokens represents a valid part
        by summing its numerical components.
        """
        numbers = [float(t) for t in tokens if re.fullmatch(r'\d+(?:\.\d+)?', t)]
        # A valid part must have at least one number.
        if not numbers:
            return False
        total = sum(numbers)
        return abs(total - 100.0) < 1e-6

    def _find_valid_partition(self, tokens: List[str]) -> Optional[List[List[str]]]:
        """
        Hypothesizes and tests different ways to partition the token list
        to satisfy the 100-sum rule for all sub-parts.
        """
        # Hypothesis 1: The entire string is a single part.
        if self._validate_part(tokens):
            text_content = ' '.join(t.lower() for t in tokens if re.fullmatch(r'[a-zA-Z]+', t))
            part_keywords = {'shell', 'lining', 'main', 'body', 'front', 'back'}
            found_keywords = {kw for kw in part_keywords if kw in text_content}
            if len(found_keywords) < 2:
                 return [tokens]

        # Hypothesis 2: Split by "KEYWORD:" patterns.
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
        """Validates a single material string. Uses a cache for speed."""
        if not isinstance(material_string, str) or not material_string.strip():
            return False

        if material_string in self.validation_cache:
            return self.validation_cache[material_string]

        s = material_string

        # --- Stricter Formatting Rules ---
        # Rule 1: Reject if it contains common care instructions.
        care_keywords = ['wash', 'tumble', 'bleach', 'iron', 'dry clean']
        if any(keyword in s.lower() for keyword in care_keywords):
            self.validation_cache[material_string] = False
            return False

        # Rule 2: Reject if numbers are present but no '%' sign is.
        if re.search(r'\d', s) and '%' not in s:
            self.validation_cache[material_string] = False
            return False

        # Rule 3: Reject hyphenated number-word combos like "80%-Acrylic"
        if re.search(r'%\s*-\s*[a-zA-Z]', s):
            self.validation_cache[material_string] = False
            return False
            
        # Rule 4: Reject if there are multiple spaces or line breaks (indicates poor formatting)
        if '\n' in s or '  ' in s:
             # Allow multiple spaces if they follow a colon, like "Lining:  100%..."
             if not re.search(r':\s\s+',s):
                self.validation_cache[material_string] = False
                return False

        # Preprocess the string to handle minor issues before core validation
        processed_string = self._preprocess(s)
        tokens = self._tokenize(processed_string)

        if not tokens:
            self.validation_cache[material_string] = False
            return False

        # Use the original partitioning logic on the cleaned string
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

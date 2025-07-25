# Validator Generation Prompt

Based on the summary report, improve validate modules to produce less false negatives. Do not change the code structure of the modules, just refactor the functionality. Reproduce the entire module after fixing it. The material validator must not know anything outside of the material format which is one or more parts with percentage for each composition. Do not add any logic that doesn't belong to material format (e.g. care instruction, etc.)

## Refined with AI

> Using the provided summary report, carefully analyze and refactor the uploaded validator modules to reduce false negatives. Maintain the existing code structure—focus only on improving the validation logic. After making improvements, output the complete, revised module code. Ensure the material validator strictly validates the material format: one or more parts, each with a percentage composition. Do not introduce logic unrelated to material format (e.g., care instructions or other metadata).

When improving the validator modules, do not introduce any assumptions or hardcoded lists about part or component names (such as 'shell', 'lining', 'front', etc.) in the material validator. The material validator should only validate the format: one or more parts, each with a percentage composition. Do not add or infer any logic or keywords related to specific part names, care instructions, or metadata outside of the material format.
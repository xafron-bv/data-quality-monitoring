def get_rule_to_column_map():
    """
    Get the mapping from rule files to column names.
    """
    return {
        "category": "article_structure_name_2",
        "color_name": "colour_name",
        "ean": "EAN",
        "article_number": "article_number",
        "colour_code": "colour_code",
        "customs_tariff_number": "customs_tariff_number",
        "description_short_1": "description_short_1",
        "long_description_nl": "long_description_NL",
        "material": "material",
        "product_name_en": "product_name_EN",
        "size": "size_name",  # Fixed: use 'size' rule file for 'size_name' column
        # Excluded: season (only 1 unique value), care_instructions (only 2 unique values)
    }

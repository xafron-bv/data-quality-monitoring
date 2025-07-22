def get_field_to_column_map():
    """
    Get the mapping from field names to column names.
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
        "care_instructions": "Care Instructions",  # Added: maps to the actual column name with capitals and space
        "season": "season",  # Re-added: might be useful for analysis even with limited values
        "manufactured_in": "Manufactured in",  # Added: maps to column with space
        "supplier": "supplier",  # Added: supplier information
        "brand": "brand",  # Added: brand information
        "collection": "collection",  # Added: collection information
        # Note: Some fields were previously excluded due to limited unique values but can be useful for specific analyses
    }

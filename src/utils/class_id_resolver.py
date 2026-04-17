"""
Shared utility for resolving class IDs from acad_term_id and class_boss_id.
Used by ClassAvailabilityProcessor and BidResultProcessor.
"""
import os
import pandas as pd


def find_all_class_ids(acad_term_id, class_boss_id, new_classes, existing_classes_cache, output_base=None):
    """
    Finds all class_ids for a given acad_term_id and class_boss_id.
    Returns ALL class records for multi-professor classes.

    Args:
        acad_term_id: Academic term ID
        class_boss_id: Class boss ID
        new_classes: List of newly created class dictionaries
        existing_classes_cache: List of existing class dictionaries from cache
        output_base: Optional path to check for new_classes.csv file

    Returns:
        List of class IDs
    """
    if pd.isna(acad_term_id) or pd.isna(class_boss_id):
        return []

    found_class_ids = []

    # Source 1: Check newly created classes in this run
    if new_classes:
        for class_obj in new_classes:
            if (class_obj.get('acad_term_id') == acad_term_id and
                str(class_obj.get('boss_id')) == str(class_boss_id)):
                found_class_ids.append(class_obj['id'])

    # Source 2: Check classes that existed before this run (from cache)
    if existing_classes_cache:
        for class_obj in existing_classes_cache:
            if (class_obj.get('acad_term_id') == acad_term_id and
                str(class_obj.get('boss_id')) == str(class_boss_id)):
                found_class_ids.append(class_obj['id'])

    # Source 3: Check new_classes.csv file if cache is incomplete
    if output_base:
        try:
            new_classes_path = os.path.join(output_base, 'new_classes.csv')
            if os.path.exists(new_classes_path):
                df = pd.read_csv(new_classes_path)
                matching_classes = df[
                    (df['acad_term_id'] == acad_term_id) &
                    (df['boss_id'].astype(str) == str(class_boss_id))
                ]
                for _, row in matching_classes.iterrows():
                    if row['id'] not in found_class_ids:
                        found_class_ids.append(row['id'])
        except Exception:
            pass

    # Remove duplicates while preserving order
    unique_class_ids = []
    seen = set()
    for class_id in found_class_ids:
        if class_id not in seen:
            unique_class_ids.append(class_id)
            seen.add(class_id)

    return unique_class_ids

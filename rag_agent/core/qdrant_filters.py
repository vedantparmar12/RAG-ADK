"""
Simple SQL-like filter helper for Qdrant queries.
Converts basic WHERE-like expressions to Qdrant Filter conditions.
"""

import re
from typing import Dict, Any, List, Union, Optional
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, Range
import logging

logger = logging.getLogger(__name__)


class QdrantFilterBuilder:
    """Helper to build Qdrant filters from SQL-like expressions."""
    
    @staticmethod
    def parse_where_clause(where_clause: str) -> Optional[Filter]:
        """
        Parse a simple WHERE clause into Qdrant Filter.
        
        Supported syntax:
        - field = 'value' or field = "value"
        - field != 'value'
        - field IN ('val1', 'val2', ...)
        - field NOT IN (...)
        - field > 10, field >= 10, field < 10, field <= 10
        - field1 = 'val1' AND field2 = 'val2'
        - field1 = 'val1' OR field2 = 'val2'
        
        Examples:
        - "document_id = 'doc123'"
        - "chunk_index >= 5 AND metadata.category = 'tech'"
        - "source IN ('doc1', 'doc2') OR priority > 8"
        """
        if not where_clause or not where_clause.strip():
            return None
        
        try:
            return QdrantFilterBuilder._parse_expression(where_clause.strip())
        except Exception as e:
            logger.error(f"Failed to parse WHERE clause '{where_clause}': {e}")
            return None
    
    @staticmethod
    def _parse_expression(expr: str) -> Filter:
        """Parse a full expression with AND/OR."""
        # Handle OR first (lower precedence)
        if " OR " in expr:
            parts = expr.split(" OR ", 1)
            left = QdrantFilterBuilder._parse_expression(parts[0].strip())
            right = QdrantFilterBuilder._parse_expression(parts[1].strip())
            return Filter(should=[left, right])
        
        # Handle AND
        if " AND " in expr:
            parts = expr.split(" AND ", 1)
            left = QdrantFilterBuilder._parse_expression(parts[0].strip())
            right = QdrantFilterBuilder._parse_expression(parts[1].strip())
            return Filter(must=[left, right])
        
        # Parse single condition
        return QdrantFilterBuilder._parse_condition(expr)
    
    @staticmethod
    def _parse_condition(condition: str) -> Filter:
        """Parse a single condition like 'field = value'."""
        condition = condition.strip()
        
        # Handle IN/NOT IN
        in_match = re.match(r"([\\w\\.]+)\\s+(NOT\\s+IN|IN)\\s*\\((.+)\\)", condition, re.IGNORECASE)
        if in_match:
            field = in_match.group(1)
            operator = in_match.group(2).upper()
            values_str = in_match.group(3)
            
            # Parse values list
            values = []
            for val in values_str.split(','):
                val = val.strip()
                if val.startswith(("'", '"')) and val.endswith(("'", '"')):\n                    values.append(val[1:-1])  # Remove quotes\n                else:\n                    # Try to parse as number\n                    try:\n                        if '.' in val:\n                            values.append(float(val))\n                        else:\n                            values.append(int(val))\n                    except ValueError:\n                        values.append(val)  # Keep as string\n            \n            if operator == "IN":\n                return Filter(must=[FieldCondition(key=field, match=MatchAny(any=values))])\n            else:  # NOT IN\n                return Filter(must_not=[FieldCondition(key=field, match=MatchAny(any=values))])\n        \n        # Handle comparison operators\n        for op in ['>=', '<=', '!=', '=', '>', '<']:\n            if f" {op} " in condition:\n                parts = condition.split(f" {op} ", 1)\n                field = parts[0].strip()\n                value_str = parts[1].strip()\n                \n                # Parse value\n                value = QdrantFilterBuilder._parse_value(value_str)\n                \n                if op == '=':\n                    return Filter(must=[FieldCondition(key=field, match=MatchValue(value=value))])\n                elif op == '!=':\n                    return Filter(must_not=[FieldCondition(key=field, match=MatchValue(value=value))])\n                elif op in ['>', '>=', '<', '<=']:\n                    # Create range condition\n                    range_kwargs = {}\n                    if op == '>':\n                        range_kwargs['gt'] = value\n                    elif op == '>=':\n                        range_kwargs['gte'] = value\n                    elif op == '<':\n                        range_kwargs['lt'] = value\n                    elif op == '<=':\n                        range_kwargs['lte'] = value\n                    \n                    return Filter(must=[FieldCondition(key=field, range=Range(**range_kwargs))])\n        \n        raise ValueError(f"Could not parse condition: {condition}")\n    \n    @staticmethod\n    def _parse_value(value_str: str) -> Union[str, int, float, bool]:\n        """Parse a value string to appropriate type."""\n        value_str = value_str.strip()\n        \n        # String with quotes\n        if value_str.startswith(("'", '"')) and value_str.endswith(("'", '"')):\n            return value_str[1:-1]\n        \n        # Boolean\n        if value_str.lower() in ['true', 'false']:\n            return value_str.lower() == 'true'\n        \n        # Number\n        try:\n            if '.' in value_str:\n                return float(value_str)\n            else:\n                return int(value_str)\n        except ValueError:\n            pass\n        \n        # Default to string\n        return value_str


def create_metadata_filter(metadata_filters: Dict[str, Any]) -> Optional[Filter]:
    """
    Create a Qdrant filter from metadata key-value pairs.
    
    Args:
        metadata_filters: Dict of field -> value mappings
        
    Returns:
        Qdrant Filter object or None
    """
    if not metadata_filters:
        return None
    
    conditions = []
    for field, value in metadata_filters.items():
        if isinstance(value, list):\n            # Multiple values - use MatchAny\n            conditions.append(FieldCondition(key=f"metadata.{field}", match=MatchAny(any=value)))\n        else:\n            # Single value - use MatchValue\n            conditions.append(FieldCondition(key=f"metadata.{field}", match=MatchValue(value=value)))\n    \n    return Filter(must=conditions) if conditions else None


def create_document_filter(document_ids: List[str]) -> Optional[Filter]:
    """Create a filter to match specific document IDs."""\n    if not document_ids:\n        return None\n    \n    if len(document_ids) == 1:\n        return Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_ids[0]))])\n    else:\n        return Filter(must=[FieldCondition(key="document_id", match=MatchAny(any=document_ids))])\n\n\ndef create_text_filter(text_contains: str) -> Optional[Filter]:\n    """Create a simple text contains filter (if Qdrant supports it)."""\n    if not text_contains:\n        return None\n    \n    # Note: Qdrant doesn't have native text contains; this would need to be\n    # handled by preprocessing or using the FTS layer\n    logger.warning("Text contains filters not directly supported by Qdrant - use FTS layer")\n    return None

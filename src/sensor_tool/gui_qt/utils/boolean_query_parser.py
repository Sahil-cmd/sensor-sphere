"""
Boolean Query Parser for Advanced Search

Supports boolean operators (AND, OR, NOT) with parenthetical grouping.
Examples:
- "(cameras OR lidars) AND outdoor"
- "Intel AND NOT (D435i OR D455)"
- "precision AND (weight < 500g OR compact)"
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BooleanOp(Enum):
    AND = "AND"
    OR = "OR"
    NOT = "NOT"


@dataclass
class QueryNode:
    """Node in the boolean expression tree."""

    operator: Optional[BooleanOp] = None
    value: Optional[str] = None
    left: Optional["QueryNode"] = None
    right: Optional["QueryNode"] = None

    def is_leaf(self) -> bool:
        return self.value is not None

    def __str__(self) -> str:
        if self.is_leaf():
            return f'"{self.value}"'
        elif self.operator == BooleanOp.NOT:
            return f"NOT {self.left}"
        else:
            return f"({self.left} {self.operator.value} {self.right})"


class BooleanQueryParser:
    """
    Parser for boolean query expressions with operator precedence.

    Grammar:
    expression   := or_expr
    or_expr      := and_expr ('OR' and_expr)*
    and_expr     := not_expr ('AND' not_expr)*
    not_expr     := 'NOT'? primary
    primary      := '(' expression ')' | term
    term         := [quoted_string | word | filter_expression]
    """

    def __init__(self):
        # Tokenization patterns
        self.token_patterns = [
            (r"\(", "LPAREN"),
            (r"\)", "RPAREN"),
            (r"\bAND\b", "AND"),
            (r"\bOR\b", "OR"),
            (r"\bNOT\b", "NOT"),
            (r'"[^"]*"', "QUOTED"),
            (r"[a-zA-Z_][a-zA-Z0-9_]*\s*[<>=!]+\s*[0-9.]+[a-zA-Z]*", "FILTER"),
            (r"\w+", "WORD"),
            (r"\s+", "WHITESPACE"),
        ]

        # Compile patterns
        self.compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), token_type)
            for pattern, token_type in self.token_patterns
        ]

    def tokenize(self, query: str) -> List[Tuple[str, str]]:
        """Tokenize query string into (token, type) pairs."""
        tokens = []
        pos = 0

        while pos < len(query):
            matched = False

            for pattern, token_type in self.compiled_patterns:
                match = pattern.match(query, pos)
                if match:
                    token = match.group(0)
                    if token_type != "WHITESPACE":  # Skip whitespace
                        tokens.append((token, token_type))
                    pos = match.end()
                    matched = True
                    break

            if not matched:
                # Skip unknown character
                pos += 1

        return tokens

    def parse(self, query: str) -> QueryNode:
        """Parse query string into boolean expression tree."""
        if not query.strip():
            return QueryNode(value="")

        try:
            tokens = self.tokenize(query)
            if not tokens:
                return QueryNode(value=query)

            self.tokens = tokens
            self.pos = 0

            result = self._parse_or_expression()

            # If we didn't consume all tokens, it's likely a simple query
            if self.pos < len(self.tokens):
                # Return original query as single term
                return QueryNode(value=query)

            return result

        except Exception as e:
            logger.debug(f"Boolean parsing failed: {e}, treating as simple query")
            return QueryNode(value=query)

    def _current_token(self) -> Optional[Tuple[str, str]]:
        """Get current token without advancing."""
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def _consume_token(self, expected_type: Optional[str] = None) -> Tuple[str, str]:
        """Consume and return current token."""
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of query")

        token = self.tokens[self.pos]
        self.pos += 1

        if expected_type and token[1] != expected_type:
            raise ValueError(f"Expected {expected_type}, got {token[1]}")

        return token

    def _parse_or_expression(self) -> QueryNode:
        """Parse OR expression (lowest precedence)."""
        left = self._parse_and_expression()

        while self._current_token() and self._current_token()[1] == "OR":
            self._consume_token("OR")
            right = self._parse_and_expression()
            left = QueryNode(operator=BooleanOp.OR, left=left, right=right)

        return left

    def _parse_and_expression(self) -> QueryNode:
        """Parse AND expression (middle precedence)."""
        left = self._parse_not_expression()

        while self._current_token() and self._current_token()[1] in [
            "AND",
            "WORD",
            "QUOTED",
            "FILTER",
            "LPAREN",
        ]:

            # Handle explicit AND
            if self._current_token()[1] == "AND":
                self._consume_token("AND")
                right = self._parse_not_expression()
                left = QueryNode(operator=BooleanOp.AND, left=left, right=right)

            # Handle implicit AND (adjacent terms)
            elif self._current_token()[1] in ["WORD", "QUOTED", "FILTER", "LPAREN"]:
                right = self._parse_not_expression()
                left = QueryNode(operator=BooleanOp.AND, left=left, right=right)
            else:
                break

        return left

    def _parse_not_expression(self) -> QueryNode:
        """Parse NOT expression (highest precedence)."""
        if self._current_token() and self._current_token()[1] == "NOT":
            self._consume_token("NOT")
            operand = self._parse_primary()
            return QueryNode(operator=BooleanOp.NOT, left=operand)

        return self._parse_primary()

    def _parse_primary(self) -> QueryNode:
        """Parse primary expression (parentheses or term)."""
        token = self._current_token()
        if not token:
            raise ValueError("Unexpected end of expression")

        # Handle parentheses
        if token[1] == "LPAREN":
            self._consume_token("LPAREN")
            expr = self._parse_or_expression()
            self._consume_token("RPAREN")
            return expr

        # Handle terms
        if token[1] in ["WORD", "QUOTED", "FILTER"]:
            term_token = self._consume_token()
            term_value = term_token[0]

            # Remove quotes from quoted strings
            if term_token[1] == "QUOTED":
                term_value = term_value.strip('"')

            return QueryNode(value=term_value)

        raise ValueError(f"Unexpected token: {token}")

    def evaluate(
        self,
        node: QueryNode,
        search_func,
        sensor_data: List[Dict[str, Any]],
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Evaluate boolean expression tree against sensor data.

        Args:
            node: Root node of expression tree
            search_func: Function to perform search (e.g., FTS search)
            sensor_data: List of all sensors for filtering
            filters: Additional structured filters

        Returns:
            List of sensor IDs matching the boolean expression
        """
        if node.is_leaf():
            # Leaf node - perform search
            if not node.value:
                return []

            # Check if it's a filter expression (e.g., "weight < 500g")
            if self._is_filter_expression(node.value):
                return self._evaluate_filter(node.value, sensor_data)
            else:
                # Perform text search
                results = search_func(node.value, filters=filters)
                return [result[0] for result in results]  # Extract sensor IDs

        # Internal node - apply boolean operator
        if node.operator == BooleanOp.AND:
            left_results = set(
                self.evaluate(node.left, search_func, sensor_data, filters)
            )
            right_results = set(
                self.evaluate(node.right, search_func, sensor_data, filters)
            )
            return list(left_results.intersection(right_results))

        elif node.operator == BooleanOp.OR:
            left_results = set(
                self.evaluate(node.left, search_func, sensor_data, filters)
            )
            right_results = set(
                self.evaluate(node.right, search_func, sensor_data, filters)
            )
            return list(left_results.union(right_results))

        elif node.operator == BooleanOp.NOT:
            all_sensor_ids = set(sensor["sensor_id"] for sensor in sensor_data)
            excluded_results = set(
                self.evaluate(node.left, search_func, sensor_data, filters)
            )
            return list(all_sensor_ids - excluded_results)

        return []

    def _is_filter_expression(self, term: str) -> bool:
        """Check if term is a filter expression like 'weight < 500g'."""
        filter_pattern = r"\w+\s*[<>=!]+\s*[0-9.]+\w*"
        return bool(re.match(filter_pattern, term))

    def _evaluate_filter(
        self, filter_expr: str, sensor_data: List[Dict[str, Any]]
    ) -> List[str]:
        """Evaluate filter expression against sensor data."""
        # Parse filter expression
        match = re.match(r"(\w+)\s*([<>=!]+)\s*([0-9.]+)(\w*)", filter_expr)
        if not match:
            return []

        field, operator, value_str, unit = match.groups()
        try:
            value = float(value_str)
        except ValueError:
            return []

        matching_sensors = []

        for sensor in sensor_data:
            sensor_value = sensor.get(field)
            if sensor_value is None:
                continue

            # Extract numeric value if it's a string
            if isinstance(sensor_value, str):
                num_match = re.search(r"(\d+(?:\.\d+)?)", sensor_value)
                if num_match:
                    sensor_value = float(num_match.group(1))
                else:
                    continue

            # Apply operator
            if operator == "<" and sensor_value < value:
                matching_sensors.append(sensor["sensor_id"])
            elif operator == ">" and sensor_value > value:
                matching_sensors.append(sensor["sensor_id"])
            elif operator in ["=", "=="] and sensor_value == value:
                matching_sensors.append(sensor["sensor_id"])
            elif operator == "<=" and sensor_value <= value:
                matching_sensors.append(sensor["sensor_id"])
            elif operator == ">=" and sensor_value >= value:
                matching_sensors.append(sensor["sensor_id"])

        return matching_sensors

    def to_string(self, node: QueryNode) -> str:
        """Convert expression tree back to string representation."""
        if node.is_leaf():
            return node.value or ""

        if node.operator == BooleanOp.NOT:
            return f"NOT {self.to_string(node.left)}"
        else:
            left_str = self.to_string(node.left)
            right_str = self.to_string(node.right)
            return f"({left_str} {node.operator.value} {right_str})"


def test_boolean_parser():
    """Test boolean query parser with example queries."""
    parser = BooleanQueryParser()

    test_queries = [
        # Simple queries
        "cameras",
        "Intel cameras",
        # Boolean operations
        "cameras AND outdoor",
        "cameras OR lidars",
        "NOT Intel",
        # Parentheses
        "(cameras OR lidars) AND outdoor",
        "Intel AND (cameras OR lidars)",
        # Complex expressions
        "Intel AND NOT (D435i OR D455)",
        "(precision OR high-quality) AND weight < 500g",
        # Filter expressions
        "weight < 500g",
        "latency < 100ms AND cameras",
    ]

    # Test functionality if run as main module
    for query in test_queries:
        try:
            tree = parser.parse(query)
            logger.info(f"Query '{query}' parsed successfully")
        except Exception as e:
            logger.error(f"Query '{query}' failed: {e}")


if __name__ == "__main__":
    test_boolean_parser()

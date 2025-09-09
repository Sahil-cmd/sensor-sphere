"""
Fuzzy Matching Module for Search Queries

Handles typos, variations, and alternative spellings in search queries.
Uses Levenshtein distance, phonetic matching, and domain-specific rules.
"""

import difflib
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class FuzzyMatch:
    """Represents a fuzzy match result"""

    original: str
    matched: str
    score: float
    match_type: str  # 'exact', 'fuzzy', 'phonetic', 'abbreviation'


class FuzzyMatcher:
    """
    Handles fuzzy matching for search terms with comprehensive typo tolerance.
    Uses multiple strategies: Levenshtein distance, phonetic matching,
    abbreviations, unit normalization, and domain-specific rules.

    Features:
    - Common typo correction
    - Abbreviation expansion
    - Unit normalization with spaces, dashes, symbols
    - Plural/singular variants
    - Phonetic matching for soundalikes
    - Context-aware suggestions
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._setup_mappings()

        # Compile regex patterns for better performance
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for better performance"""
        self.unit_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*([a-zA-Z°]+)")
        self.currency_pattern = re.compile(r"([\$£€])\s*(\d+(?:\.\d+)?)")
        self.resolution_pattern = re.compile(
            r"(\d+)\s*[x×]\s*(\d+)(?:\s*(pixels?|px|p))?"
        )
        self.range_pattern = re.compile(
            r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*([a-zA-Z°]+)"
        )
        self.space_normalize_pattern = re.compile(r"\s+")
        self.number_unit_pattern = re.compile(r"(\d)([a-zA-Z°])")
        self.unit_number_pattern = re.compile(r"(\d)\s+([a-zA-Z°])")

    def _setup_mappings(self):
        """Initialize fuzzy matching rules and mappings"""

        # Common typos and variations
        self.common_typos = {
            "camara": "camera",
            "camrea": "camera",
            "cemera": "camera",
            "lidr": "lidar",
            "lider": "lidar",
            "ladar": "lidar",
            "lazer": "laser",
            "sesnor": "sensor",
            "sensro": "sensor",
            "realsence": "realsense",
            "real sense": "realsense",
            "realsnese": "realsense",
            "stereolap": "stereolabs",
            "stereolab": "stereolabs",
            "kinect": "kinect",
            "kinnect": "kinect",
            "orbecc": "orbbec",
            "orbec": "orbbec",
            "mechmind": "mech-mind",
            "mech mind": "mech-mind",
            "mecheye": "mech-eye",
            "mech eye": "mech-eye",
        }

        # Abbreviation mappings
        self.abbreviations = {
            "fps": "frame rate",
            "ms": "milliseconds",
            "msec": "milliseconds",
            "fov": "field of view",
            "res": "resolution",
            "mp": "megapixels",
            "ir": "infrared",
            "imu": "inertial measurement unit",
            "tof": "time of flight",
            "rgb": "color",
            "rgbd": "depth",
            "ros": "robot operating system",
            "ip": "ingress protection",
            "usb": "universal serial bus",
            "hw": "hardware",
            "sw": "software",
            "doc": "documentation",
            "docs": "documentation",
            "env": "environmental",
            "temp": "temperature",
            "hz": "hertz",
            "ghz": "gigahertz",
            "mhz": "megahertz",
        }

        # Phonetic replacements for soundalike matching
        self.phonetic_patterns = [
            (r"ph", "f"),  # phone -> fone
            (r"ght", "t"),  # light -> lite
            (r"ck", "k"),  # quick -> quik
            (r"qu", "kw"),  # quick -> kwick
            (r"x", "ks"),  # box -> boks
            (r"tion", "shun"),  # motion -> moshun
            (r"sion", "shun"),  # vision -> vishun
            (r"ce", "se"),  # price -> prise
            (r"ci", "si"),  # city -> sity
            (r"cy", "sy"),  # fancy -> fansy
            (r"dge", "j"),  # bridge -> brij
            (r"tch", "ch"),  # match -> mach
            (r"wr", "r"),  # write -> rite
            (r"kn", "n"),  # know -> no
            (r"wh", "w"),  # what -> wat
        ]

        # Unit variations
        self.unit_variations = {
            # Distance/Range
            "m": ["meter", "meters", "metre", "metres", "m"],
            "cm": ["centimeter", "centimeters", "centimetre", "centimetres", "cm"],
            "mm": ["millimeter", "millimeters", "millimetre", "millimetres", "mm"],
            "km": ["kilometer", "kilometers", "kilometre", "kilometres", "km"],
            "ft": ["foot", "feet", "ft"],
            "in": ["inch", "inches", "in"],
            # Time
            "ms": ["millisecond", "milliseconds", "msec", "ms"],
            "s": ["second", "seconds", "sec", "s"],
            "us": ["microsecond", "microseconds", "usec", "us"],
            "ns": ["nanosecond", "nanoseconds", "nsec", "ns"],
            # Weight
            "g": ["gram", "grams", "gm", "g"],
            "kg": ["kilogram", "kilograms", "kg"],
            "lb": ["pound", "pounds", "lb", "lbs"],
            "oz": ["ounce", "ounces", "oz"],
            # Power
            "w": ["watt", "watts", "w"],
            "mw": ["milliwatt", "milliwatts", "mw"],
            "kw": ["kilowatt", "kilowatts", "kw"],
            # Angle
            "deg": ["degree", "degrees", "deg", "°"],
            "rad": ["radian", "radians", "rad"],
            # Frequency
            "hz": ["hertz", "hz"],
            "khz": ["kilohertz", "khz"],
            "mhz": ["megahertz", "mhz"],
            "ghz": ["gigahertz", "ghz"],
        }

    def find_best_match(
        self, query: str, candidates: List[str], threshold: float = 0.6
    ) -> Optional[FuzzyMatch]:
        """
        Find the best fuzzy match for a query from a list of candidates.

        Args:
            query: The search term to match
            candidates: List of possible matches
            threshold: Minimum similarity score (0-1) to consider a match

        Returns:
            FuzzyMatch object with the best match, or None if no good match
        """
        query_lower = query.lower().strip()

        # Check for exact match first
        for candidate in candidates:
            if query_lower == candidate.lower():
                return FuzzyMatch(query, candidate, 1.0, "exact")

        # Check common typos
        if query_lower in self.common_typos:
            corrected = self.common_typos[query_lower]
            for candidate in candidates:
                if corrected == candidate.lower():
                    return FuzzyMatch(query, candidate, 0.9, "typo")

        # Try fuzzy matching
        best_match = None
        best_score = 0.0

        for candidate in candidates:
            # Calculate similarity score
            score = self._calculate_similarity(query_lower, candidate.lower())

            if score > best_score and score >= threshold:
                best_score = score
                best_match = candidate

        if best_match:
            return FuzzyMatch(query, best_match, best_score, "fuzzy")

        # Try phonetic matching as last resort
        phonetic_query = self._to_phonetic(query_lower)
        for candidate in candidates:
            phonetic_candidate = self._to_phonetic(candidate.lower())

            if phonetic_query == phonetic_candidate:
                return FuzzyMatch(query, candidate, 0.7, "phonetic")

        return None

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate similarity score between two strings.
        Uses combination of different metrics for robustness.
        """
        # Levenshtein ratio (sequence matcher)
        ratio = difflib.SequenceMatcher(None, str1, str2).ratio()

        # Boost score if strings start the same
        if str1 and str2 and str1[0] == str2[0]:
            ratio += 0.1

        # Boost score if one is substring of other
        if str1 in str2 or str2 in str1:
            ratio += 0.2

        return min(ratio, 1.0)

    def _to_phonetic(self, text: str) -> str:
        """Convert text to phonetic representation for soundalike matching"""
        result = text.lower()

        # Apply phonetic patterns
        for pattern, replacement in self.phonetic_patterns:
            result = re.sub(pattern, replacement, result)

        # Remove silent letters
        result = re.sub(r"[aeiou]", "", result)  # Remove vowels for consonant skeleton

        return result

    def correct_query(self, query: str) -> str:
        """
        Correct common typos and variations in a query.

        Args:
            query: The raw search query

        Returns:
            Corrected query string
        """
        words = query.lower().split()
        corrected_words = []

        for word in words:
            # Check common typos
            if word in self.common_typos:
                corrected_words.append(self.common_typos[word])
            else:
                corrected_words.append(word)

        return " ".join(corrected_words)

    def expand_abbreviations(self, query: str) -> str:
        """
        Expand abbreviations in query to full terms.

        Args:
            query: The search query

        Returns:
            Query with expanded abbreviations
        """
        words = query.lower().split()
        expanded_words = []

        for word in words:
            # Check if it's an abbreviation
            if word in self.abbreviations:
                expanded_words.append(self.abbreviations[word])
            else:
                expanded_words.append(word)

        return " ".join(expanded_words)

    def normalize_units(self, query: str) -> str:
        """
        Normalize unit variations to standard forms.

        Args:
            query: The search query

        Returns:
            Query with normalized units
        """
        result = query.lower()

        # Pattern to match numbers with units
        pattern = r"(\d+(?:\.\d+)?)\s*([a-zA-Z°]+)"

        def replace_unit(match):
            number = match.group(1)
            unit = match.group(2).lower()

            # Find standard unit
            for standard, variations in self.unit_variations.items():
                if unit in variations:
                    return f"{number} {standard}"

            return match.group(0)

        result = re.sub(pattern, replace_unit, result)

        return result

    def get_alternative_queries(
        self, query: str, max_alternatives: int = 5
    ) -> List[str]:
        """
        Generate alternative query variations for better matching.

        Args:
            query: The original query
            max_alternatives: Maximum number of alternatives to generate

        Returns:
            List of alternative query strings
        """
        alternatives = []

        # Original query
        alternatives.append(query)

        # Corrected typos
        corrected = self.correct_query(query)
        if corrected != query.lower():
            alternatives.append(corrected)

        # Expanded abbreviations
        expanded = self.expand_abbreviations(query)
        if expanded != query.lower():
            alternatives.append(expanded)

        # Normalized units
        normalized = self.normalize_units(query)
        if normalized != query.lower():
            alternatives.append(normalized)

        # Try swapping word order for 2-word queries
        words = query.split()
        if len(words) == 2:
            swapped = f"{words[1]} {words[0]}"
            alternatives.append(swapped)

        # Remove duplicates and limit
        seen = set()
        unique_alternatives = []
        for alt in alternatives:
            alt_lower = alt.lower()
            if alt_lower not in seen:
                seen.add(alt_lower)
                unique_alternatives.append(alt)
                if len(unique_alternatives) >= max_alternatives:
                    break

        return unique_alternatives

    def suggest_corrections(
        self, query: str, known_terms: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Suggest corrections for potentially misspelled terms.

        Args:
            query: The search query
            known_terms: List of known valid terms

        Returns:
            List of (suggestion, confidence) tuples
        """
        suggestions = []
        words = query.lower().split()

        for word in words:
            # Skip if word is known
            if word in known_terms:
                continue

            # Find best match
            match = self.find_best_match(word, known_terms, threshold=0.5)

            if match and match.match_type != "exact":
                suggestions.append((match.matched, match.score))

        # Sort by confidence
        suggestions.sort(key=lambda x: x[1], reverse=True)

        return suggestions


# Factory function
def create_fuzzy_matcher() -> FuzzyMatcher:
    """Create and return FuzzyMatcher instance"""
    return FuzzyMatcher()

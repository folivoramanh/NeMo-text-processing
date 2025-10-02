"""
Vietnamese Phonetic Rules for Alternative Pronunciation Generation
================================================================

This module implements systematic rules for generating Vietnamese number pronunciation alternatives
based on linguistic patterns and existing TSV data files.
"""

from typing import List, Dict, Set, Tuple
import re

# Import existing utilities instead of reimplementing
from nemo_text_processing.text_normalization.vi.utils import get_abs_path, load_labels
from nemo_text_processing.text_normalization.vi.graph_utils import NEMO_SPACE


class VietnamesePhoneticRules:
    """
    Systematic rule-based generator for Vietnamese number pronunciation alternatives.
    Based on Vietnamese linguistic patterns and leverages existing TSV data files.
    """
    
    def __init__(self):
        # Load base data from TSV files using existing utilities
        self.base_digits = self._load_tsv_as_dict("data/numbers/digit.tsv")
        self.base_magnitudes = self._load_tsv_as_dict("data/numbers/magnitudes.tsv") 
        self.base_ties = self._load_tsv_as_dict("data/numbers/ties.tsv")
        self.base_teens = self._load_tsv_as_dict("data/numbers/teen.tsv")
        self.base_zero = self._load_tsv_as_dict("data/numbers/zero.tsv")
        self.ordinal_exceptions = self._load_tsv_as_dict("data/ordinal/ordinal_exceptions.tsv")
        
        # Load digit alternatives from existing digit_special.tsv
        self.digit_alternatives = self._load_digit_special()
        
        # Connector words for different contexts
        self.connectors = {
            "zero_after_hundred": ["lẻ", "không", "linh"],
            "magnitude_separators": ["nghìn", "ngàn"],  # thousand alternatives
            "tens_formats": ["mười", "mươi"]  # colloquial vs formal
        }
        
        # Magnitude system using TSV data + extensions
        self.magnitude_order = ["quadrillions", "trillions", "billions", "millions", "thousands", "units"]
        
        # Build magnitude system from TSV data
        self.magnitude_system = self._build_magnitude_system()

    def _load_digit_special(self) -> Dict[str, List[str]]:
        """Load digit alternatives from digit_special.tsv using existing utilities"""
        try:
            abs_path = get_abs_path("data/numbers/digit_special.tsv")
            labels = load_labels(abs_path)
            
            # Convert to format: {"1": ["một", "mốt"], "4": ["bốn", "tư"], ...}
            special = {}
            for parts in labels:
                if len(parts) >= 3:
                    digit = parts[0]
                    std = parts[1]
                    alt = parts[2]
                    special[digit] = [std, alt]
            
            return special
        except Exception as e:
            print(f"Warning: Could not load digit_special.tsv: {e}")
            # Fallback to hardcoded values
            return {
                "1": ["một", "mốt"],
                "4": ["bốn", "tư"],
                "5": ["năm", "lăm"],
            }

    def _remove_duplicates(self, items: List[str]) -> List[str]:
        """Remove duplicates while preserving order using standard pattern"""
        return list(dict.fromkeys(items)) if items else []

    def _load_tsv_as_dict(self, rel_path: str) -> Dict[str, str]:
        """Load TSV data as dictionary using existing utilities"""
        try:
            abs_path = get_abs_path(rel_path)
            labels = load_labels(abs_path)
            
            # Convert list of lists to dictionary
            data = {}
            for parts in labels:
                if len(parts) >= 2:
                    data[parts[0]] = parts[1]
            
            return data
        except FileNotFoundError:
            print(f"Warning: TSV file not found: {rel_path}")
            return {}
        except Exception as e:
            print(f"Warning: Could not load TSV file {rel_path}: {e}")
            return {}
    
    def _build_magnitude_system(self) -> Dict:
        """Build magnitude system from TSV data"""
        return {
            "units": {
                "range": (1, 999),
                "base_word": [],
                "connectors": self.connectors["zero_after_hundred"]
            },
            "thousands": {
                "range": (1000, 999999),
                "base_word": self.connectors["magnitude_separators"],
                "connectors": self.connectors["zero_after_hundred"]
            },
            "millions": {
                "range": (1000000, 999999999),
                "base_word": [self.base_magnitudes.get("million", "triệu")],
                "connectors": self.connectors["zero_after_hundred"]
            },
            "billions": {
                "range": (1000000000, 999999999999),
                "base_word": [self.base_magnitudes.get("billion", "tỷ")],
                "connectors": self.connectors["zero_after_hundred"]
            },
            "trillions": {
                "range": (1000000000000, 999999999999999),
                "base_word": [self.base_magnitudes.get("trillion", "nghìn tỷ")],
                "connectors": self.connectors["zero_after_hundred"]
            },
            "quadrillions": {
                "range": (1000000000000000, 999999999999999999),
                "base_word": [self.base_magnitudes.get("quadrillion", "triệu tỷ")],
                "connectors": self.connectors["zero_after_hundred"]
            }
        }

    def _determine_magnitude(self, number: int) -> str:
        """Determine which magnitude category a number belongs to"""
        for magnitude, config in self.magnitude_system.items():
            min_val, max_val = config["range"]
            if min_val <= number <= max_val:
                return magnitude
        return "units"  # Default fallback
    
    def _decompose_number_by_magnitude(self, number: int) -> List[Dict]:
        """Decompose a number into magnitude components
        
        Example: 1234567 -> [
            {"value": 1, "magnitude": "millions", "remainder": 234567},
            {"value": 234, "magnitude": "thousands", "remainder": 567}, 
            {"value": 567, "magnitude": "units", "remainder": 0}
        ]
        """
        components = []
        remaining = number
        
        # Process from largest to smallest magnitude
        magnitude_order = ["quadrillions", "trillions", "billions", "millions", "thousands", "units"]
        
        for magnitude in magnitude_order:
            config = self.magnitude_system[magnitude]
            min_val, max_val = config["range"]
            
            if magnitude == "units":
                if remaining > 0:
                    components.append({
                        "value": remaining,
                        "magnitude": magnitude,
                        "remainder": 0
                    })
                break
            
            # Calculate magnitude value (e.g., billions = remaining // 1000000000)
            magnitude_divisor = min_val
            if remaining >= magnitude_divisor:
                magnitude_value = remaining // magnitude_divisor
                components.append({
                    "value": magnitude_value,
                    "magnitude": magnitude,
                    "remainder": remaining % magnitude_divisor
                })
                remaining = remaining % magnitude_divisor
        
        return components
    
    def _generate_magnitude_alternatives(self, value: int, magnitude: str) -> List[str]:
        """Generate alternatives for a specific magnitude component"""
        config = self.magnitude_system[magnitude]
        base_words = config.get("base_word", [])
        
        if magnitude == "units":
            # Use existing unit generation logic
            return self._generate_unit_alternatives(value)
        
        # For thousands, millions, billions
        alternatives = []
        
        # Get alternatives for the value part (always < 1000)
        value_alternatives = self._generate_unit_alternatives(value)
        
        # Combine with magnitude base words
        for value_alt in value_alternatives:
            for base_word in base_words:
                alternatives.append(f"{value_alt} {base_word}")
        
        return alternatives
    
    def _generate_unit_alternatives(self, number: int) -> List[str]:
        """Generate alternatives for numbers 1-999 (units magnitude)"""
        if number <= 0 or number > 999:
            return [str(number)]
            
        # Use existing generate_alternatives logic but ensure it handles all cases
        return self._generate_small_number_alternatives(number, "general")
    
    def _combine_magnitude_components(self, components: List[Dict]) -> List[str]:
        """Combine magnitude components with appropriate connectors"""
        if not components:
            return []
        
        if len(components) == 1:
            # Single component
            comp = components[0]
            return self._generate_magnitude_alternatives(comp["value"], comp["magnitude"])
        
        # Multiple components - need to combine with connectors
        all_combinations = []
        
        # Generate alternatives for each component
        component_alternatives = []
        for comp in components:
            comp_alts = self._generate_magnitude_alternatives(comp["value"], comp["magnitude"])
            component_alternatives.append(comp_alts)
        
        # Generate all combinations
        from itertools import product
        
        for combination in product(*component_alternatives):
            # Join components with appropriate connectors
            result = self._join_with_connectors(combination, components)
            all_combinations.extend(result)
        
        return all_combinations
    
    def _join_with_connectors(self, parts: List[str], components: List[Dict]) -> List[str]:
        """Join magnitude parts with appropriate connectors"""
        if len(parts) <= 1:
            return list(parts)
        
        results = []
        
        # Use standard space separator
        basic_join = " ".join(parts)
        results.append(basic_join)
        
        # Add connector variations for specific patterns
        if len(parts) == 2:
            # Two parts: handle zero connectors
            if self._has_zero_pattern(components):
                # Use the connectors from the refactored system
                connectors = self.connectors["zero_after_hundred"]
                for connector in connectors:
                    alt_join = f"{parts[0]} {connector} {parts[1]}"
                    results.append(alt_join)
        
        return results
    
    def _has_zero_pattern(self, components: List[Dict]) -> bool:
        """Check if components have zero patterns (e.g., 1001, 2005)"""
        if len(components) < 2:
            return False
        
        # Check if the smaller magnitude component starts with zero-like pattern
        smaller_comp = components[1]
        return smaller_comp["value"] < 100  # Simplified check
    
    def generate_alternatives(self, number_str: str, context: str = "general") -> List[str]:
        """Enhanced generate_alternatives using magnitude system"""
        try:
            number = int(number_str)
        except ValueError:
            return [number_str]
        
        # Use magnitude system for large numbers
        if number >= 1000:
            return self._generate_large_number_alternatives_v2(number)
        
        # Use existing logic for numbers < 1000
        return self._generate_small_number_alternatives(number, context)
    
    def _generate_large_number_alternatives_v2(self, number: int) -> List[str]:
        """Generate alternatives for numbers >= 1000 using magnitude system"""
        try:
            # Decompose number into magnitude components
            components = self._decompose_number_by_magnitude(number)
            
            # Generate alternatives for the combination
            alternatives = self._combine_magnitude_components(components)
            
            # Ensure we have at least the basic form
            if not alternatives:
                alternatives = [str(number)]
            
            return alternatives
            
        except Exception as e:
            print(f"Warning: Could not generate large number alternatives for {number}: {e}")
            return [str(number)]
    
    def _generate_small_number_alternatives(self, number: int, context: str) -> List[str]:
        """Generate alternatives for numbers 1-999 using original logic"""
        if number == 0:
            return [self.base_zero.get("0", "không")]
        
        if 1 <= number <= 9:
            return self._get_digit_alternatives_in_context(number, context)
        elif 10 <= number <= 19:
            return self._generate_teen_alternatives(number)
        elif 20 <= number <= 99:
            return self._generate_two_digit_alternatives(number)
        elif 100 <= number <= 999:
            return self._generate_three_digit_alternatives(number, context)
        else:
            return [str(number)]

    def _generate_two_digit_alternatives(self, num: int) -> List[str]:
        """Generate alternatives for two-digit numbers (10-99)"""
        alternatives = []
        
        if 10 <= num <= 19:  # Teen numbers
            alternatives.extend(self._generate_teen_alternatives(num))
        elif 20 <= num <= 99:  # Tens
            alternatives.extend(self._generate_tens_alternatives(num))
            
        return alternatives

    def _generate_teen_alternatives(self, num: int) -> List[str]:
        """Generate alternatives for teen numbers (10-19)"""
        alternatives = []
        ones_digit = num % 10
        
        if ones_digit == 0:  # 10
            alternatives.append("mười")
        else:
            # Base form: mười + digit
            base_forms = ["mười"]
            digit_alts = self._get_digit_alternatives_in_context(ones_digit, "after_tens")
            
            for base in base_forms:
                for digit_alt in digit_alts:
                    alternatives.append(f"{base} {digit_alt}")
                    
        return alternatives

    def _generate_tens_alternatives(self, num: int) -> List[str]:
        """Generate alternatives for tens (20-99)"""
        alternatives = []
        tens_digit = num // 10
        ones_digit = num % 10
        
        # Get tens word
        tens_words = self._get_tens_word(tens_digit)
        
        if ones_digit == 0:
            # Just the tens word
            alternatives.extend(tens_words)
        else:
            # Tens + ones combinations
            ones_alts = self._get_digit_alternatives_in_context(ones_digit, "after_tens")
            
            for tens_word in tens_words:
                for ones_alt in ones_alts:
                    alternatives.append(f"{tens_word} {ones_alt}")
                    
        return alternatives

    def _generate_three_digit_alternatives(self, num: int, context: str) -> List[str]:
        """Generate alternatives for three-digit numbers (100-999)"""
        alternatives = []
        
        hundreds_digit = num // 100
        remainder = num % 100
        
        # Handle short forms for certain contexts
        if context in ["page", "room"]:
            alternatives.extend(self._generate_short_form_alternatives(num))
            
        # Standard full forms
        hundreds_word = self._get_hundreds_word(hundreds_digit)
        
        if remainder == 0:
            # Just hundreds (100, 200, etc.)
            alternatives.append(f"{hundreds_word} trăm")
        elif remainder < 10:
            # Handle 101-109 pattern with ALL possible variations
            zero_connectors = self.connectors["zero_after_hundred"]
            ones_alts = self._get_digit_alternatives_in_context(remainder, "after_hundred")
            
            for zero_connector in zero_connectors:
                for ones_alt in ones_alts:
                    alternatives.append(f"{hundreds_word} trăm {zero_connector} {ones_alt}")
        else:
            # Handle 110-199 pattern  
            remainder_alts = self._generate_two_digit_alternatives(remainder)
            for remainder_alt in remainder_alts:
                alternatives.append(f"{hundreds_word} trăm {remainder_alt}")
        
        # ALWAYS add short forms for ALL numbers, not just specific contexts
        short_forms = self._generate_short_form_alternatives(num)
        alternatives.extend(short_forms)
                
        return alternatives

    def _generate_short_form_alternatives(self, num: int) -> List[str]:
        """Generate ALL possible short form alternatives for ANY 3-digit number"""
        alternatives = []
        
        if 100 <= num <= 999:
            hundreds_digit = num // 100
            remainder = num % 100
            
            # Get hundreds word
            hundreds_word = self._get_hundreds_word(hundreds_digit)
            
            if remainder < 10:
                # X01-X09: Generate short forms for ALL hundreds
                # Examples: 201 -> hai lẻ một, 304 -> ba lẻ bốn, 507 -> năm không bảy
                ones_alts = self._get_digit_alternatives_in_context(remainder, "after_hundred")  # Use same context
                
                # All possible connectors
                connectors = self.connectors["zero_after_hundred"]
                
                for connector in connectors:
                    for ones_alt in ones_alts:
                        alternatives.append(f"{hundreds_word} {connector} {ones_alt}")
            
            elif 10 <= remainder <= 19:
                # X10-X19: Short forms for teens
                # Examples: 215 -> hai mười lăm, 314 -> ba mười tư
                teen_alts = self._generate_teen_alternatives(remainder)
                for teen_alt in teen_alts:
                    alternatives.append(f"{hundreds_word} {teen_alt}")
            
            elif remainder >= 20:
                # X20-X99: Short forms for tens
                # Examples: 225 -> hai hai lăm, 334 -> ba ba tư  
                tens_digit = remainder // 10
                ones_digit = remainder % 10
                
                if ones_digit == 0:
                    # X20, X30, etc. -> hai hai, ba ba
                    tens_word = self._get_digit_name(tens_digit)
                    alternatives.append(f"{hundreds_word} {tens_word}")
                else:
                    # X21, X34, etc. -> hai hai mốt, ba ba tư
                    tens_word = self._get_digit_name(tens_digit)
                    ones_alts = self._get_digit_alternatives_in_context(ones_digit, "after_tens")  # Use appropriate context
                    for ones_alt in ones_alts:
                        alternatives.append(f"{hundreds_word} {tens_word} {ones_alt}")
                        
        return alternatives

    def _get_digit_alternatives_in_context(self, digit: int, context: str) -> List[str]:
        """Get digit alternatives based on context using TSV data"""
        digit_str = str(digit)
        alternatives = []
        
        # Use base digit from TSV as default
        if digit_str in self.base_digits:
            alternatives.append(self.base_digits[digit_str])
        
        # Add enhanced alternatives for specific digits
        if digit_str in self.digit_alternatives:
            alternatives.extend(self.digit_alternatives[digit_str])
        
        # Context-specific handling
        if context == "after_hundred" and digit_str == "1":
            alternatives = ["một", "mốt"]  # Both valid after hundred
        elif context == "after_tens":
            if digit_str == "1":
                alternatives = ["mốt"]  # Only mốt after tens
            elif digit_str == "5":
                alternatives = ["lăm"]  # Only lăm after tens
            elif digit_str in self.digit_alternatives:
                alternatives = self.digit_alternatives[digit_str]
        
        # Remove duplicates while preserving order
        return self._remove_duplicates(alternatives) if alternatives else [digit_str]

    def _get_tens_word(self, tens_digit: int) -> List[str]:
        """Get tens word alternatives using TSV data"""
        tens_str = str(tens_digit)
        alternatives = []
        
        # Use base ties from TSV
        if tens_str in self.base_ties:
            alternatives.append(self.base_ties[tens_str])
        
        # Add colloquial alternatives for specific digits
        colloquial_map = {
            "2": "hai mười",
            "3": "ba mười", 
            "4": "tư mười",
            "5": "lăm mười"
        }
        
        if tens_str in colloquial_map:
            alternatives.append(colloquial_map[tens_str])
        
        return alternatives if alternatives else [f"{tens_digit} mười"]

    def _get_hundreds_word(self, hundreds_digit: int) -> str:
        """Get hundreds word using TSV data"""
        digit_str = str(hundreds_digit)
        return self.base_digits.get(digit_str, str(hundreds_digit))
    
    def _get_digit_name(self, digit: int) -> str:
        """Get basic digit name using TSV data"""
        digit_str = str(digit)
        if digit == 0:
            return self.base_zero.get("0", "không")
        return self.base_digits.get(digit_str, str(digit))

    def _generate_teen_alternatives(self, num: int) -> List[str]:
        """Generate alternatives for teen numbers using TSV data"""
        alternatives = []
        teen_str = str(num)
        
        # Use base teen from TSV if available
        if teen_str in self.base_teens:
            alternatives.append(self.base_teens[teen_str])
        
        # Generate additional alternatives for specific teens
        ones_digit = num % 10
        if ones_digit in [1, 4, 5]:  # Digits with alternatives
            base_forms = ["mười"]
            digit_alts = self._get_digit_alternatives_in_context(ones_digit, "after_tens")
            
            for base in base_forms:
                for digit_alt in digit_alts:
                    alt = f"{base} {digit_alt}"
                    if alt not in alternatives:
                        alternatives.append(alt)
        
        return alternatives if alternatives else [teen_str]

    def generate_ordinal_alternatives(self, number_str: str) -> List[str]:
        """Generate alternatives for ordinal numbers using TSV data"""
        alternatives = []
        
        # Check ordinal exceptions from TSV first
        if number_str in self.ordinal_exceptions:
            alternatives.append(self.ordinal_exceptions[number_str])
        
        # Apply general number rules
        general_alts = self.generate_alternatives(number_str, "general")
        alternatives.extend(general_alts)
        
        return self._remove_duplicates(alternatives)
    
    def generate_fraction_numerator_alternatives(self, number_str: str) -> List[str]:
        """Generate alternatives for fraction numerators"""
        return self.generate_alternatives(number_str, "general")
    
    def generate_fraction_denominator_alternatives(self, number_str: str) -> List[str]:
        """Generate alternatives for fraction denominators"""
        alternatives = []
        
        # Basic fraction denominators with alternatives
        fraction_alts = {
            "2": ["hai", "đôi"],
            "4": ["bốn", "tư"],
            "10": ["mười", "mười phần"],
            "100": ["trăm", "phần trăm"]
        }
        
        if number_str in fraction_alts:
            alternatives.extend(fraction_alts[number_str])
        
        general_alts = self.generate_alternatives(number_str, "general")
        alternatives.extend(general_alts)
        
        return self._remove_duplicates(alternatives)
    
    def generate_decimal_digit_alternatives(self, digit_str: str, position: str = "fractional") -> List[str]:
        """Generate alternatives for decimal digits"""
        alternatives = []
        
        if digit_str == "0" and position == "fractional":
            alternatives.extend(["không", "lẻ"])
        
        general_alts = self.generate_alternatives(digit_str, "general")
        alternatives.extend(general_alts)
        
        return self._remove_duplicates(alternatives)
    
    def get_systematic_range_alternatives(self, start: int, end: int, context: str = "general") -> Dict[str, List[str]]:
        """Generate alternatives for a range of numbers"""
        range_alternatives = {}
        
        for i in range(start, min(end + 1, start + 1000)):  # Limit to prevent memory issues
            number_str = str(i)
            alternatives = self.generate_alternatives(number_str, context)
            if alternatives:
                range_alternatives[number_str] = alternatives
                
        return range_alternatives


# Example usage and testing
if __name__ == "__main__":
    rules = VietnamesePhoneticRules()
    
    # Test basic functionality
    test_numbers = ["1", "4", "5", "15", "101", "1004"]
    
    print("=== Vietnamese Phonetic Rules Test (TSV-based) ===")
    for num in test_numbers:
        alternatives = rules.generate_alternatives(num, context="general")
        print(f"{num}: {len(alternatives)} alternatives - {alternatives[:3]}{'...' if len(alternatives) > 3 else ''}")

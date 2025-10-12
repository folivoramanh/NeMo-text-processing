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
        # Fixed: "lẻ/linh" only appear once, to replace zero positions
        self.connectors = {
            "zero_after_hundred": ["lẻ", "linh"],  # Removed "không" - not appropriate as connector
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
        """Remove duplicates while preserving order and filter out incomplete conversions"""
        if not items:
            return []
        
        # Filter out alternatives containing raw digits (incomplete conversion)
        filtered_items = []
        for item in items:
            if not any(char.isdigit() for char in item):
                filtered_items.append(item)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(filtered_items))

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
        
        # Check if the smaller magnitude component indicates a zero in tens place
        smaller_comp = components[1]
        value = smaller_comp["value"]
        
        # Only has zero pattern if value is single digit (0-9)
        # Examples: 2004 (value=4), 2005 (value=5) → True
        # Counter-examples: 2024 (value=24), 2034 (value=34) → False
        return value < 10
    
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
            # Special handling for 4-digit years (e.g., 2024)
            if 1000 <= number <= 9999:
                alternatives = []
                
                # Standard magnitude-based alternatives
                components = self._decompose_number_by_magnitude(number)
                standard_alts = self._combine_magnitude_components(components)
                alternatives.extend(standard_alts)
                
                # Add short forms for years (e.g., 2024 → "hai không hai tư")
                year_short_forms = self._generate_year_short_forms(number)
                alternatives.extend(year_short_forms)
                
                return self._remove_duplicates(alternatives)
            else:
                # Regular magnitude system for other large numbers
                components = self._decompose_number_by_magnitude(number)
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

    def _generate_year_short_forms(self, number: int) -> List[str]:
        """Generate short forms for 4-digit years (e.g., 2024 → hai không hai tư)"""
        alternatives = []
        
        if 1000 <= number <= 9999:
            # Convert to string and split into digits
            year_str = str(number)
            
            # Generate digit-by-digit reading
            digit_alternatives = []
            for digit_char in year_str:
                digit = int(digit_char)
                if digit == 0:
                    # Special handling for zero in year context
                    # Use only "không" to avoid excessive combinations
                    zero_alts = ["không"]  # Simplified to reduce alternatives
                    digit_alternatives.append(zero_alts)
                else:
                    digit_alts = self._get_digit_alternatives_in_context(digit, "general")
                    digit_alternatives.append(digit_alts)
            
            # Generate combinations of digit alternatives
            import itertools
            for combo in itertools.product(*digit_alternatives):
                short_form = " ".join(combo)
                alternatives.append(short_form)
        
        return alternatives

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
                
        return self._remove_duplicates(alternatives)

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
        """
        Get digit alternatives based on context using TSV data
        
        Rules:
        - "mốt": Only at end of tens (after_tens), NOT for standalone 1/01/11
        - "lăm": Only at end of tens (after_tens), NOT for standalone 5/05
        - No "x mười" patterns (hai mười, ba mười, etc.)
        """
        digit_str = str(digit)
        alternatives = []
        
        # Use base digit from TSV as default
        if digit_str in self.base_digits:
            alternatives.append(self.base_digits[digit_str])
        
        # Context-specific handling with proper Vietnamese rules
        if context == "after_hundred":
            # After hundred: use base forms only (no mốt/lăm)
            if digit_str in self.digit_alternatives:
                alts = self.digit_alternatives[digit_str].copy()
                # Remove mốt/lăm from after hundred context
                if digit_str == "1" and "mốt" in alts:
                    alts.remove("mốt")
                if digit_str == "5" and "lăm" in alts:
                    alts.remove("lăm")
                alternatives.extend(alts)
                    
        elif context == "after_tens":
            # After tens: use special forms for 1 and 5
            if digit_str == "1":
                # After tens: only "mốt" (never "một")
                # E.g., "hai mười mốt", "ba mười mốt"
                alternatives = ["mốt"]
            elif digit_str == "5":
                # After tens: only "lăm" (never "năm")
                # E.g., "hai mười lăm", "ba mười lăm"  
                alternatives = ["lăm"]
            else:
                # Other digits: use base form only
                alternatives = [self.base_digits.get(digit_str, digit_str)]
                
        elif context == "general":
            # General context: use base + TSV alternatives
            # But exclude "mốt" and "lăm" for standalone numbers
            if digit_str in self.digit_alternatives:
                alts = self.digit_alternatives[digit_str].copy()
                # Remove mốt/lăm from standalone numbers
                if digit_str == "1" and "mốt" in alts:
                    alts.remove("mốt")
                if digit_str == "5" and "lăm" in alts:
                    alts.remove("lăm")
                alternatives.extend(alts)
                
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
        # Note: Vietnamese doesn't have "hai mười", "ba mười", "bốn mười" formats
        # These are incorrect patterns that should be removed
        colloquial_map = {
            # Removed all "x mười" patterns - these don't exist in Vietnamese
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
        
        # Special exceptions: 11 should only be "mười một" (no "mười mốt")
        if num == 11:
            return ["mười một"]
        
        # Generate additional alternatives for specific teens (except 11)
        ones_digit = num % 10
        if ones_digit in [1, 4, 5]:  # Digits with alternatives
            base_forms = ["mười"]
            digit_alts = self._get_digit_alternatives_in_context(ones_digit, "after_tens")
            
            for base in base_forms:
                for digit_alt in digit_alts:
                    alt = f"{base} {digit_alt}"
                    if alt not in alternatives:
                        alternatives.append(alt)
        
        # Special handling for 15: should have both "mười lăm" and "mười năm"
        if num == 15:
            if "mười năm" not in alternatives:
                alternatives.append("mười năm")
        
        return alternatives if alternatives else [teen_str]

    def generate_date_day_alternatives(self, number_str: str, include_prefix: bool = True) -> List[str]:
        """Generate alternatives for date days - NO automatic prefixes to avoid duplicates"""
        # Always return base alternatives without any automatic prefix
        # Prefixes should only come from input patterns, not generated here
        base_alternatives = self.generate_alternatives(number_str, "general")
        
        if include_prefix:
            day_num = int(number_str)
            
            # Only "mùng", "mồng" prefixes for days 1-10, but ONLY if explicitly requested
            # This is used by prefix-preserving patterns, not standard patterns
            if 1 <= day_num <= 10:
                day_prefixes_small = ["mùng", "mồng"]
                prefixed_alternatives = []
                for prefix in day_prefixes_small:
                    for base_alt in base_alternatives:
                        prefixed_alternatives.append(f"{prefix} {base_alt}")
                
                # Return ONLY prefixed alternatives when include_prefix=True
                # This prevents mixing prefixed and non-prefixed in same FST
                return self._remove_duplicates(prefixed_alternatives)
        
        # For include_prefix=False or days > 10, return base alternatives only
        return self._remove_duplicates(base_alternatives)

    def generate_date_month_alternatives(self, number_str: str, include_prefix: bool = True) -> List[str]:
        """Generate alternatives for date months with Vietnamese prefixes"""
        alternatives = []
        
        # Get basic number alternatives
        base_alternatives = self.generate_alternatives(number_str, "general")
        
        if include_prefix:
            # Add "tháng" prefix
            for base_alt in base_alternatives:
                alternatives.append(f"tháng {base_alt}")
        else:
            # No prefix, just return base alternatives
            alternatives.extend(base_alternatives)
        
        return self._remove_duplicates(alternatives)

    def generate_date_year_alternatives(self, number_str: str, include_prefix: bool = True) -> List[str]:
        """Generate alternatives for date years with Vietnamese prefixes"""
        alternatives = []
        
        # Get basic number alternatives
        base_alternatives = self.generate_alternatives(number_str, "general")
        
        if include_prefix:
            # Add "năm" prefix
            for base_alt in base_alternatives:
                alternatives.append(f"năm {base_alt}")
        else:
            # No prefix, just return base alternatives
            alternatives.extend(base_alternatives)
        
        return self._remove_duplicates(alternatives)

    def generate_ordinal_alternatives(self, number_str: str) -> List[str]:
        """Generate alternatives for ordinal numbers using TSV data"""
        alternatives = []
        
        # Check ordinal exceptions from TSV first
        if number_str in self.ordinal_exceptions:
            alternatives.append(self.ordinal_exceptions[number_str])
        
        # Add specific ordinal alternatives
        if number_str == "1":
            alternatives.extend(["nhất", "một"])  # hạng nhất hoặc hạng một
        elif number_str == "2":
            alternatives.extend(["nhì", "hai"])   # hạng nhì hoặc hạng hai
        elif number_str == "4":
            alternatives.extend(["tư", "bốn"])    # hạng tư hoặc hạng bốn
        
        # Apply general number rules for other cases
        general_alts = self.generate_alternatives(number_str, "general")
        alternatives.extend(general_alts)
        
        return self._remove_duplicates(alternatives)
    
    def generate_fraction_numerator_alternatives(self, number_str: str) -> List[str]:
        """Generate alternatives for fraction numerators"""
        alternatives = []
        
        # Special handling for numerator "1" in fractions (1/x)
        if number_str == "1":
            alternatives.extend([
                "một",        # một phần x
                "",           # phần x (không có một)
                "chia"        # chia x (alternative reading)
            ])
        else:
            # For other numerators, use general alternatives
            alternatives = self.generate_alternatives(number_str, "general")
        
        return self._remove_duplicates(alternatives)
    
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

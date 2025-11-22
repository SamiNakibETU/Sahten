"""
Content Guard
Validates and sanitizes responses to ensure editorial compliance
"""

import logging
import re
from typing import Literal

from app.models.schemas import ScenarioContext
from app.models.config import settings

logger = logging.getLogger(__name__)


class ValidationResult:
    """Result of content validation"""

    def __init__(self):
        self.is_valid = True
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def add_error(self, message: str):
        """Add a validation error"""
        self.is_valid = False
        self.errors.append(message)

    def add_warning(self, message: str):
        """Add a validation warning"""
        self.warnings.append(message)


class ContentGuard:
    """
    Content guard that validates responses meet editorial constraints

    Checks:
    1. Language is French
    2. No ingredient lists/steps for OLJ scenarios (scenario 1)
    3. URLs are from allowed domain
    4. HTML format (no Markdown)
    5. Emoji count (1-3 max)
    6. Emoji types (food/emotion, no flags)
    7. Response length (~100 words, except Base 2 full recipes)
    8. Required links present where expected
    """

    # Flag emojis to reject
    FLAG_PATTERN = re.compile(r'[\U0001F1E6-\U0001F1FF]{2}')  # Flag emojis

    # Non-French word patterns (common English words that shouldn't appear)
    NON_FRENCH_PATTERNS = [
        r'\bthe\b', r'\band\b', r'\bor\b', r'\bwith\b', r'\bfor\b',
        r'\brecipe\b', r'\bcooking\b', r'\bingredients?\b',
    ]

    def __init__(self):
        pass

    def validate(self, html: str, scenario: ScenarioContext) -> ValidationResult:
        """
        Validate HTML response

        Returns ValidationResult with errors and warnings
        """
        result = ValidationResult()

        # 1. Check language (should be mostly French)
        if not self._is_french(html):
            result.add_warning("Response may not be in French")

        # 2. For OLJ scenarios, check NO ingredient lists/steps
        if scenario.scenario_id == 1 and not scenario.show_full_recipe:
            if self._contains_ingredient_list(html):
                result.add_error("OLJ scenario must not contain ingredient lists")
            if self._contains_steps_list(html):
                result.add_error("OLJ scenario must not contain cooking steps")

        # 3. Check URLs are from allowed domain
        if not self._all_urls_valid(html):
            result.add_error(f"Found URLs outside allowed domain: {settings.allowed_url_domain}")

        # 4. Check HTML format (no Markdown)
        if self._contains_markdown(html):
            result.add_warning("Response contains Markdown formatting, should be HTML only")

        # 5. Check emoji count
        emoji_count = self._count_emojis(html)
        if emoji_count > settings.max_emojis:
            result.add_error(f"Too many emojis ({emoji_count}), max is {settings.max_emojis}")

        # 6. Check for flag emojis
        if self._contains_flags(html):
            result.add_error("Response contains flag emojis (not allowed)")

        # 7. Check response length
        word_count = self._count_words(html)
        max_words = settings.max_response_words_recipe if scenario.show_full_recipe else settings.max_response_words

        if word_count > max_words + 50:  # Allow some buffer
            result.add_warning(f"Response is long ({word_count} words), target is ~{max_words}")

        # 8. Check required link presence
        if scenario.include_link:
            if not self._contains_link(html):
                result.add_error("Scenario requires a link but none found")

        return result

    def sanitize(self, html: str, scenario: ScenarioContext) -> str:
        """
        Sanitize HTML to ensure compliance

        Applies automatic fixes:
        - Remove excess emojis
        - Remove flag emojis
        - Trim excessive length (if needed)
        """
        sanitized = html

        # Remove flag emojis
        sanitized = self.FLAG_PATTERN.sub('', sanitized)

        # Limit emojis
        sanitized = self._limit_emojis(sanitized, settings.max_emojis)

        # Trim length if too long (last resort)
        max_words = settings.max_response_words_recipe if scenario.show_full_recipe else settings.max_response_words
        sanitized = self._trim_to_length(sanitized, max_words + 100)  # Allow buffer

        return sanitized

    def _is_french(self, text: str) -> bool:
        """Check if text appears to be in French"""
        # Remove HTML tags for analysis
        clean_text = re.sub(r'<[^>]+>', '', text).lower()

        # Check for non-French patterns
        for pattern in self.NON_FRENCH_PATTERNS:
            if re.search(pattern, clean_text, re.IGNORECASE):
                return False

        # French word indicators
        french_indicators = ['le', 'la', 'les', 'de', 'du', 'des', 'une', 'un', 'pour', 'avec']
        has_french = any(word in clean_text for word in french_indicators)

        return has_french

    def _contains_ingredient_list(self, html: str) -> bool:
        """Check if HTML contains what looks like an ingredient list"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', html).lower()

        # Patterns that indicate ingredient lists
        patterns = [
            r'ingrédients?\s*:',
            r'\d+\s*(g|ml|c\.\s*à\s*(soupe|café))',  # Quantities
            r'^\s*[\d•\-]\s*\d+.*?(grammes?|litres?)',  # List items with quantities
        ]

        return any(re.search(pattern, text, re.MULTILINE) for pattern in patterns)

    def _contains_steps_list(self, html: str) -> bool:
        """Check if HTML contains what looks like cooking steps"""
        text = re.sub(r'<[^>]+>', '', html).lower()

        # Patterns for cooking steps
        patterns = [
            r'(préparation|étapes?)\s*:',
            r'^\s*\d+\.\s*(faire|mettre|ajouter|mélanger|cuire)',  # Numbered steps
        ]

        return any(re.search(pattern, text, re.MULTILINE | re.IGNORECASE) for pattern in patterns)

    def _all_urls_valid(self, html: str) -> bool:
        """Check all URLs are from allowed domain"""
        # Extract all URLs
        url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
        urls = re.findall(url_pattern, html)

        for url in urls:
            if not url.startswith(settings.allowed_url_domain):
                logger.warning(f"Invalid URL domain: {url}")
                return False

        return True

    def _contains_markdown(self, html: str) -> bool:
        """Check if response contains Markdown instead of HTML"""
        # Common Markdown patterns
        markdown_patterns = [
            r'\*\*[^*]+\*\*',  # **bold**
            r'\*[^*]+\*',  # *italic*
            r'^\s*#\s+',  # # Headers
            r'^\s*-\s+',  # - List items (at line start)
            r'^\s*\d+\.\s+',  # 1. Numbered items
            r'\[([^\]]+)\]\(([^)]+)\)',  # [text](url)
        ]

        # But ignore cases where these are inside HTML tags
        text_without_tags = re.sub(r'<[^>]+>', '', html)

        return any(re.search(pattern, text_without_tags, re.MULTILINE) for pattern in markdown_patterns)

    def _count_emojis(self, text: str) -> int:
        """Count emojis in text"""
        # Emoji ranges
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )

        matches = emoji_pattern.findall(text)
        return len(matches)

    def _contains_flags(self, text: str) -> bool:
        """Check if text contains flag emojis"""
        return bool(self.FLAG_PATTERN.search(text))

    def _count_words(self, html: str) -> int:
        """Count words in HTML (excluding tags)"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', html)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Count words
        words = text.split()
        return len(words)

    def _contains_link(self, html: str) -> bool:
        """Check if HTML contains at least one link"""
        return bool(re.search(r'<a\s+[^>]*href=', html))

    def _limit_emojis(self, text: str, max_emojis: int) -> str:
        """Remove excess emojis"""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )

        emojis = emoji_pattern.findall(text)
        if len(emojis) <= max_emojis:
            return text

        # Remove excess emojis from the end
        result = text
        for i in range(len(emojis) - max_emojis):
            # Remove last emoji
            result = emoji_pattern.sub('', result, count=1)

        return result

    def _trim_to_length(self, html: str, max_words: int) -> str:
        """Trim HTML to maximum word count"""
        word_count = self._count_words(html)
        if word_count <= max_words:
            return html

        # Simple trimming: split by </p> and keep first paragraphs
        paragraphs = html.split('</p>')
        result = []
        current_words = 0

        for para in paragraphs:
            para_words = self._count_words(para)
            if current_words + para_words <= max_words:
                result.append(para + '</p>')
                current_words += para_words
            else:
                break

        return '\n'.join(result) if result else html

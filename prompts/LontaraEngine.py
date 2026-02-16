import re
import json

class LontaraEngine:

    # =========================
    # BASIC MAPPING
    # =========================

    LONTARA_TO_LATIN = {
        "ᨀ": "ka", "ᨁ": "ga", "ᨂ": "nga", "ᨃ": "pa",
        "ᨄ": "ba", "ᨅ": "ba", "ᨆ": "ma", "ᨇ": "da",
        "ᨈ": "ta", "ᨉ": "da", "ᨊ": "na", "ᨋ": "nya",
        "ᨌ": "ca", "ᨍ": "ra", "ᨎ": "la", "ᨏ": "wa",
        "ᨐ": "sa", "ᨑ": "ra", "ᨒ": "la", "ᨓ": "fa",
        "ᨔ": "kha", "ᨕ": "a", "ᨖ": "za",
    }

    DIACRITICS = {
        "ᨗ": "i",
        "ᨘ": "u",
        "ᨙ": "e",
        "ᨚ": "o",
        "ᨛ": "é"
    }

    LATIN_TO_LONTARA_BASE = {
        "k": "ᨀ", "g": "ᨁ", "ng": "ᨂ", "p": "ᨃ",
        "b": "ᨅ", "m": "ᨆ", "d": "ᨉ",
        "t": "ᨈ", "c": "ᨌ", "ny": "ᨋ",
        "y": "ᨌ", "r": "ᨑ", "l": "ᨒ", "w": "ᨏ",
        "s": "ᨐ", "h": "ᨒ", "f": "ᨓ",
        "a": "ᨕ", "n": "ᨊ"
    }

    VOWEL_MARK = {
        "i": "ᨗ",
        "u": "ᨘ",
        "e": "ᨙ",
        "o": "ᨚ",
        "é": "ᨛ"
    }

    # =========================
    # LONTARA → LATIN
    # =========================

    @classmethod
    def lontara_to_latin(cls, text: str) -> str:
        result = ""
        skip = False

        for i, char in enumerate(text):
            if skip:
                skip = False
                continue

            if char in cls.LONTARA_TO_LATIN:
                base = cls.LONTARA_TO_LATIN[char]

                # check diacritic
                if i + 1 < len(text) and text[i + 1] in cls.DIACRITICS:
                    vowel = cls.DIACRITICS[text[i + 1]]
                    result += base[:-1] + vowel
                    skip = True
                else:
                    result += base
            else:
                result += char

        return result.strip()

    # =========================
    # SIMPLE BUGIS RECONSTRUCTION
    # (can be expanded with dictionary)
    # =========================

    @staticmethod
    def reconstruct_bugis(text: str) -> str:
        # contoh sederhana
        text = re.sub(r"pe", "ppe", text)
        text = re.sub(r"ka", "kka", text)
        text = re.sub(r"nganare", "nganre", text)
        return text

    # =========================
    # LATIN BUGIS → LONTARA
    # =========================

    @classmethod
    def latin_to_lontara(cls, text: str) -> str:
        text = text.lower()
        result = ""
        i = 0
        
        while i < len(text):
            # Handle spaces and punctuation
            if text[i] == ' ':
                result += ' '
                i += 1
                continue
            
            # Check for digraph consonants (ng, ny) first
            if i + 1 < len(text) and text[i:i+2] in ["ng", "ny"]:
                consonant = text[i:i+2]
                base = cls.LATIN_TO_LONTARA_BASE[consonant]
                i += 2
                
                # Check for following vowel
                if i < len(text) and (text[i] in cls.VOWEL_MARK or text[i] == 'a'):
                    vowel = text[i]
                    if vowel != 'a':  # 'a' is inherent in the base
                        result += base + cls.VOWEL_MARK[vowel]
                    else:
                        result += base
                    i += 1
                else:
                    result += base
                continue
            
            char = text[i]
            
            # Handle consonants
            if char in cls.LATIN_TO_LONTARA_BASE and char != 'a':
                base = cls.LATIN_TO_LONTARA_BASE[char]
                i += 1
                
                # Check for following vowel
                if i < len(text) and (text[i] in cls.VOWEL_MARK or text[i] == 'a'):
                    vowel = text[i]
                    if vowel != 'a':  # 'a' is inherent in the base
                        result += base + cls.VOWEL_MARK[vowel]
                    else:
                        result += base
                    i += 1
                else:
                    result += base
            
            # Handle standalone vowels or initial vowels
            elif char in cls.VOWEL_MARK or char == 'a':
                if char == 'a':
                    result += cls.LATIN_TO_LONTARA_BASE['a']  # ᨕ
                else:
                    result += cls.LATIN_TO_LONTARA_BASE['a'] + cls.VOWEL_MARK[char]
                i += 1
            
            # Unknown character, pass through
            else:
                result += char
                i += 1
        
        return result

    # =========================
    # MAIN PIPELINES
    # =========================

    @classmethod
    def translate_lontara(cls, text: str) -> str:
        original_aksara = text

        latin = cls.lontara_to_latin(text)
        reconstructed = cls.reconstruct_bugis(latin)

        # dummy indonesia translation
        indonesia = reconstructed  # replace with real translator

        return json.dumps({
            "aksara": original_aksara,
            "latin": reconstructed,
            "indonesia": indonesia
        }, ensure_ascii=False)


    @classmethod
    def latin_to_aksara_json(cls, text: str) -> str:
        aksara = cls.latin_to_lontara(text)

        return json.dumps({
            "aksara": aksara,
            "latin": text,
            "indonesia": ""
        }, ensure_ascii=False)


    @classmethod
    def indonesia_to_aksara_json(cls, text: str) -> str:
        # STEP 1 translate indonesia → bugis (placeholder)
        bugis = text.lower()

        # STEP 2 reconstruct
        bugis = cls.reconstruct_bugis(bugis)

        # STEP 3 convert
        aksara = cls.latin_to_lontara(bugis)

        return json.dumps({
            "aksara": aksara,
            "latin": bugis,
            "indonesia": text
        }, ensure_ascii=False)
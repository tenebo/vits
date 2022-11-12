import re
from text import cleaners
from text.symbols import symbols
from .korean_dict import char_to_id, id_to_char

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")


def text_to_sequence(text, cleaner_names):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = []

    # Mappings from symbol to numeric ID and vice versa:
    _language, _symbol_to_id, _ = ("kr", char_to_id, id_to_char) if "korean_cleaners" in cleaner_names\
         else ("en", {s: i for i, s in enumerate(symbols)}, {i: s for i, s in enumerate(symbols)})

    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
        m = _curly_re.match(text)

        if not m:
            sequence += _symbols_to_sequence(_clean_text(text, cleaner_names), _symbol_to_id)
            break
        sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names), _symbol_to_id)
        sequence += _arpabet_to_sequence(m.group(2), _language, _symbol_to_id)
        text = m.group(3)

    return sequence


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbols, _symbol_to_id):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s, _symbol_to_id)]


def _arpabet_to_sequence(text, _language, _symbol_to_id):
    if _language == "kr":
        return _symbols_to_sequence([s for s in text.split()], _symbol_to_id)
    return _symbols_to_sequence(["@" + s for s in text.split()], _symbol_to_id)


def _should_keep_symbol(s, _symbol_to_id):
    return s in _symbol_to_id and s != "_" and s != "~"

def cleaned_text_to_sequence(cleaned_text):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = [char_to_id[symbol] for symbol in cleaned_text]
  return sequence

if __name__ == "__main__":
  from .korean import detokenize
  a=text_to_sequence("선 AB와 A'B'는 평행이다.",['korean_cleaners'])
  print(a)
  b=detokenize([id_to_char[i] for i in a])
  print(b)
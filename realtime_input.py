import re
import sys
import termios
import tty
from dataclasses import dataclass

from spell_checker import NGramSpellChecker

COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_CYAN = "\033[96m"
COLOR_RESET = "\033[0m"

WORD_PATTERN = re.compile(r"\w+", re.UNICODE)
BOUNDARY_PUNCTUATION = {".", "!", "?", ";", ":", ","}


@dataclass
class Token:
    kind: str
    value: str


def _tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    index = 0

    for match in re.finditer(r"\s+|\w+|[^\w\s]", text, re.UNICODE):
        if match.start() > index:
            tokens.append(Token(kind="other", value=text[index : match.start()]))

        value = match.group(0)
        if value.isspace():
            kind = "space"
        elif WORD_PATTERN.fullmatch(value):
            kind = "word"
        elif value in BOUNDARY_PUNCTUATION:
            kind = "punct"
        else:
            kind = "other"

        tokens.append(Token(kind=kind, value=value))
        index = match.end()

    if index < len(text):
        tokens.append(Token(kind="other", value=text[index:]))

    return tokens


def _apply_coloring(original: str, corrected: str) -> str:
    if original.lower() == corrected.lower():
        return f"{COLOR_GREEN}{corrected}{COLOR_RESET}"
    return f"{COLOR_YELLOW}{corrected}{COLOR_RESET}"


def render_corrected_text(
    text: str, checker: NGramSpellChecker, finalize: bool = False
) -> str:
    tokens = _tokenize(text)
    rendered: list[str] = []

    segment_indexes: list[int] = []
    segment_words: list[str] = []

    def flush_segment() -> None:
        nonlocal segment_indexes, segment_words
        if not segment_indexes:
            return

        corrected_words = checker.correct_sentence(" ".join(segment_words), top_k=1)
        best_words = corrected_words[0].split() if corrected_words else segment_words

        for idx, token_index in enumerate(segment_indexes):
            original_word = tokens[token_index].value
            corrected_word = best_words[idx] if idx < len(best_words) else original_word
            rendered[token_index] = _apply_coloring(original_word, corrected_word)

        segment_indexes = []
        segment_words = []

    for token in tokens:
        rendered.append(token.value)

    for token_index, token in enumerate(tokens):
        if token.kind == "word":
            segment_indexes.append(token_index)
            segment_words.append(token.value)
            continue

        if token.kind == "space":
            continue

        flush_segment()

    if finalize:
        flush_segment()
    return "".join(rendered)


def _read_key() -> str:
    char = sys.stdin.read(1)
    if char != "\x1b":
        return char

    next_char = sys.stdin.read(1)
    if next_char != "[":
        return char + next_char

    final_char = sys.stdin.read(1)
    return char + next_char + final_char


def run_realtime_input(checker: NGramSpellChecker) -> str:
    if not sys.stdin.isatty():
        raise RuntimeError("Chế độ realtime cần chạy trong terminal tương tác.")

    prompt = f"{COLOR_CYAN}Nhập văn bản: {COLOR_RESET}"
    buffer: list[str] = []
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    def redraw() -> None:
        visible_text = "".join(buffer)
        rendered = render_corrected_text(visible_text, checker, finalize=False)
        sys.stdout.write("\r\033[2K")
        sys.stdout.write(prompt + rendered)
        sys.stdout.flush()

    sys.stdout.write(prompt)
    sys.stdout.flush()

    try:
        tty.setraw(fd)
        while True:
            key = _read_key()

            if key in {"\r", "\n"}:
                break
            if key == "\x03":
                raise KeyboardInterrupt
            if key in {"\x7f", "\b"}:
                if buffer:
                    buffer.pop()
            elif key.startswith("\x1b"):
                continue
            elif key.isprintable():
                buffer.append(key)

            redraw()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        sys.stdout.write("\r\033[2K")
        sys.stdout.write(
            prompt + render_corrected_text("".join(buffer), checker, finalize=True)
        )
        sys.stdout.write("\n")
        sys.stdout.flush()

    return "".join(buffer)

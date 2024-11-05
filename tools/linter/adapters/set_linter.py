from __future__ import annotations

import dataclasses as dc
import json
import logging
import sys
import token
from argparse import ArgumentParser, Namespace
from enum import Enum
from functools import cached_property
from pathlib import Path
from tokenize import generate_tokens, TokenInfo
from typing import Any, Iterator, Sequence


BRACKETS = {
    "{": "}",
    "(": ")",
    "[": "]",
}
BRACKETS_INV = {j: i for i, j in BRACKETS.items()}
EMPTY_TOKENS = {
    token.COMMENT,
    token.DEDENT,
    token.ENCODING,
    token.INDENT,
    token.NEWLINE,
    token.NL,
}
ERROR = "Builtin `set` is deprecated"
IMPORT_LINE = "from torch.utils._ordered_set import OrderedSet\n"
OMIT = "# noqa: set_linter"


class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


@dc.dataclass
class LintMessage:
    path: str | None = None
    line: int | None = None
    char: int | None = None
    code: str = "SET_LINTER"
    severity: LintSeverity = LintSeverity.ERROR
    name: str = ""
    original: str | None = None
    replacement: str | None = None
    description: str | None = None

    asdict = dc.asdict


def main() -> None:
    args = get_args()

    for f in args.files:
        if errors := list(lint_file(f, args)):
            if args.edit:
                replacement = errors[-1].replacement
                assert replacement is not None
                Path(f).write_text(replacement)
                print("Rewrote", f)
            else:
                for error in errors:
                    print(json.dumps(error.asdict()))


@dc.dataclass(order=True)
class LintReplacement:
    line: int
    char: int
    length: int
    replacement: str
    name: str = ERROR


def lint_replacements(pl: PythonLines) -> Iterator[LintReplacement]:
    for b in pl.braced_sets:
        yield LintReplacement(*b[0].start, 1, "OrderedSet([")
        yield LintReplacement(*b[-1].start, 1, "])")

    for b in pl.sets:
        yield LintReplacement(*b.start, 3, "OrderedSet")

    if (pl.sets or pl.braced_sets) and (ins := pl.insert_import_line()) is not None:
        yield LintReplacement(ins, 0, 0, IMPORT_LINE, "Add import for OrderedSet")


def lint_file(path: str, args: Namespace) -> Iterator[LintMessage]:
    if args.verbose:
        print(path, "Reading")

    try:
        pl = PythonLines(Path(path))
    except IndentationError as e:
        msg, (name, lineno, column, _line) = e.args
        yield LintMessage(path, lineno, column, name="Indentation Error")
        return

    replacements = sorted(lint_replacements(pl), reverse=True)
    lines = pl.lines[:]
    for r in replacements:
        yield LintMessage(path, r.line, r.char, name=r.name)

        line = lines[r.line - 1]
        before, after = line[: r.char], line[r.char + r.length :]
        lines[r.line - 1] = f"{before}{r.replacement}{after}"

    if replacements:
        yield LintMessage(
            path,
            original="".join(pl.lines),
            replacement="".join(lines),
            name="Suggested fixes for set_linter",
        )


class ParseError(ValueError):
    def __init__(self, token: TokenInfo, *args: str) -> None:
        super().__init__(*args)
        self.token = token


@dc.dataclass
class TokenLine:
    """A logical line of Python tokens, terminated by a NEWLINE or the end of file"""

    tokens: list[TokenInfo]

    @cached_property
    def sets(self) -> list[TokenInfo]:
        """A list of tokens which use the built-in set symbol"""
        return [t for i, t in enumerate(self.tokens) if self.is_set(i)]

    @cached_property
    def braced_sets(self) -> list[list[TokenInfo]]:
        """A list of lists of tokens, each representing a braced set, like {1}"""
        return [
            self.tokens[b : e + 1]
            for b, e in self.bracket_pairs.items()
            if self.is_braced_set(b, e)
        ]

    @cached_property
    def bracket_pairs(self) -> dict[int, int]:
        braces: dict[int, int] = {}
        stack: list[int] = []

        def check(cond: Any, token: TokenInfo, *args: str) -> None:
            if not cond:
                raise ParseError(token, *args)

        for i, t in enumerate(self.tokens):
            if t.type == token.OP:
                if t.string in BRACKETS:
                    stack.append(i)
                elif inv := BRACKETS_INV.get(t.string):
                    check(stack, t, "Never opened")
                    begin = stack.pop()
                    braces[begin] = i

                    b = self.tokens[begin].string
                    check(b == inv, t, f"Mismatched braces '{b}' at {begin}")

        if self.tokens:
            check(not stack, t, "Left open")
        return braces

    def is_set(self, i: int) -> bool:
        t = self.tokens[i]
        after = i < len(self.tokens) - 1 and self.tokens[i + 1]
        if t.string == "Set" and t.type == token.NAME:
            return after and after.string == "[" and after.type == token.OP
        if not (t.string == "set" and t.type == token.NAME):
            return False
        if i and self.tokens[i - 1].string in ("def", "."):
            return False
        if after and after.string == "=" and after.type == token.OP:
            return False
        return True

    def is_braced_set(self, begin: int, end: int) -> bool:
        if begin + 1 == end or self.tokens[begin].string != "{":
            return False
        i = begin + 1
        empty = True
        while i < end:
            t = self.tokens[i]
            if t.type == token.OP and t.string in (":", "**"):
                return False
            if brace_end := self.bracket_pairs.get(i):
                # Skip to the end of a subexpression
                i = brace_end
            elif t.type not in EMPTY_TOKENS:
                empty = False
            i += 1
        return not empty


class PythonLines:
    """A list of lines of Python code represented by strings"""

    braced_sets: list[Sequence[TokenInfo]]
    lines: list[str]
    path: Path | None
    sets: list[TokenInfo]
    token_lines: list[TokenLine]
    tokens: list[TokenInfo]

    def __init__(self, contents: Path | str) -> None:
        if isinstance(contents, Path):
            text = contents.read_text()
            self.path = contents
        else:
            text = contents
            self.path = None
        self.lines = text.splitlines(keepends=True)
        self.tokens = list(generate_tokens(iter(self.lines).__next__))
        self.token_lines = list(self._split_into_token_lines())
        self.omitted = OmittedLines(self.lines)

        sets = [t for tl in self.token_lines for t in tl.sets]
        self.sets = [t for t in sets if not self.omitted([t])]

        braced_sets = [t for tl in self.token_lines for t in tl.braced_sets]
        self.braced_sets = [t for t in braced_sets if not self.omitted(t)]

    def _split_into_token_lines(self) -> Iterator[TokenLine]:
        token_line = TokenLine([])

        for t in self.tokens:
            if t.type != token.ENDMARKER:
                token_line.tokens.append(t)
                if t.type == token.NEWLINE:
                    yield token_line
                    token_line = TokenLine([])

        if token_line.tokens:
            yield token_line

    def insert_import_line(self) -> int | None:
        froms, imports = [], []

        for token_line in self.token_lines:
            tokens = token_line.tokens
            tokens = [t for t in tokens if t.type not in (token.COMMENT, token.NL)]
            t = tokens[0]
            if t.type == token.INDENT:
                break
            if not (t.type == token.NAME and t.string in ("from", "import")):
                continue
            if any(i.type == token.NAME and i.string == "OrderedSet" for i in tokens):
                return None
            if t.string == "from":
                froms.append(tokens)
            else:
                imports.append(tokens)

        if section := froms or imports:
            return section[-1][-1].start[0] + 1
        return 0


class OmittedLines:
    def __init__(self, lines: Sequence[str]) -> None:
        self.lines = lines
        self.omitted = {i + 1 for i, s in enumerate(lines) if s.rstrip().endswith(OMIT)}

    def __call__(self, tokens: Sequence[TokenInfo]) -> bool:
        # A token_line might span multiple physical lines
        lines = sorted(i for t in tokens for i in (t.start[0], t.end[0]))
        lines_covered = list(range(lines[0], lines[-1] + 1)) if lines else []
        return bool(self.omitted.intersection(lines_covered))


def expand_file_patterns(file_paths: list[str]) -> Iterator[str]:
    for fp in file_paths:
        for f in fp.split(":"):
            if f == "--":
                pass
            elif f.startswith("@"):
                yield from Path(f[1:]).read_text().splitlines()
            else:
                yield f


def get_args(argv: list[str] | None = None) -> Namespace:
    parser = ArgumentParser()
    add = parser.add_argument

    add("files", nargs="*", help="Files or directories to include")
    add("-v", "--verbose", action="store_true", help="Print more info")
    add(
        "-e",
        "--edit",
        action="store_true",
        help="Edit the files without using lintrunner",
    )

    args = parser.parse_args(argv)
    args.files = list(expand_file_patterns(args.files))

    if args.verbose:
        level = logging.NOTSET
    elif len(args.files) < 1000:
        level = logging.DEBUG
    else:
        level = logging.INFO
    fmt = "<%(threadName)s:%(levelname)s> %(message)s"
    logging.basicConfig(format=fmt, level=level, stream=sys.stderr)

    return args


if __name__ == "__main__":
    main()

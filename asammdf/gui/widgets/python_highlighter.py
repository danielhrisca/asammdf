#!/usr/bin/env python

"""PySide6 QPlainTextEdit syntax highlight for Python source code.
The original code if found here
https://wiki.python.org/moin/PyQt/Python%20syntax%20highlighting
"""

import builtins
import keyword

from PySide6 import QtCore, QtGui


def format(color, style=""):
    """Return a QTextCharFormat with the given attributes."""
    _color = QtGui.QColor()
    _color.setNamedColor(color)

    _format = QtGui.QTextCharFormat()
    _format.setForeground(_color)
    if "bold" in style:
        _format.setFontWeight(QtGui.QFont.Bold)
    if "italic" in style:
        _format.setFontItalic(True)

    return _format


# Syntax styles that can be shared by all languages
STYLES = {
    "keyword": format("orange"),
    "operator": format("red"),
    "brace": format("darkGray"),
    "defclass": format("orange", "bold"),
    "string": format("green"),
    "string2": format("green"),
    "comment": format("lightBlue", "italic"),
    "self": format("darkViolet", "italic"),
    "numbers": format("blue"),
    "builtins": format("darkOrange"),
}


class PythonHighlighter(QtGui.QSyntaxHighlighter):
    """Syntax highlighter for the Python language."""

    # Python keywords
    keywords = keyword.kwlist + keyword.softkwlist

    builtins = dir(builtins)

    # Python operators
    operators = [
        "=",
        # Comparison
        "==",
        "!=",
        "<",
        "<=",
        ">",
        ">=",
        # Arithmetic
        "\+",
        "-",
        "\*",
        "/",
        "//",
        "\%",
        "\*\*",
        # In-place
        "\+=",
        "-=",
        "\*=",
        "/=",
        "\%=",
        # Bitwise
        "\^",
        "\|",
        "\&",
        "\~",
        ">>",
        "<<",
    ]

    # Python braces
    braces = [
        "\{",
        "\}",
        "\(",
        "\)",
        "\[",
        "\]",
    ]

    def __init__(self, parent: QtGui.QTextDocument) -> None:
        super().__init__(parent)

        # Multi-line strings (expression, flag, style)
        self.tri_single = (QtCore.QRegularExpression("'''"), 1, STYLES["string2"])
        self.tri_double = (QtCore.QRegularExpression('"""'), 2, STYLES["string2"])

        rules = []

        # Keyword, operator, and brace rules
        rules += [
            (r"\b%s\b" % w, 0, STYLES["keyword"]) for w in PythonHighlighter.keywords
        ]
        # rules += [
        #     (r"%s" % o, 0, STYLES["operator"]) for o in PythonHighlighter.operators
        # ]
        # rules += [(r"%s" % b, 0, STYLES["brace"]) for b in PythonHighlighter.braces]
        rules += [
            (r"%s" % b, 0, STYLES["builtins"]) for b in PythonHighlighter.builtins
        ]

        # All other rules
        rules += [
            # 'self'
            (r"\bself\b", 0, STYLES["self"]),
            # 'def' followed by an identifier
            (r"\bdef\b\s*(\w+)", 1, STYLES["defclass"]),
            # 'class' followed by an identifier
            (r"\bclass\b\s*(\w+)", 1, STYLES["defclass"]),
            # Numeric literals
            (r"\b[+-]?[0-9]+\b", 0, STYLES["numbers"]),
            (r"\b[+-]?0[xX][0-9A-Fa-f]+\b", 0, STYLES["numbers"]),
            (r"\b[+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\b", 0, STYLES["numbers"]),
            # Double-quoted string, possibly containing escape sequences
            (r'"[^"\\]*(\\.[^"\\]*)*"', 0, STYLES["string"]),
            # Single-quoted string, possibly containing escape sequences
            (r"'[^'\\]*(\\.[^'\\]*)*'", 0, STYLES["string"]),
            # From '#' until a newline
            (r"#[^\n]*", 0, STYLES["comment"]),
        ]

        # Build a QRegularExpression for each pattern
        self.rules = [
            (QtCore.QRegularExpression(pat), index, fmt) for (pat, index, fmt) in rules
        ]

    def highlightBlock(self, text):
        """Apply syntax highlighting to the given block of text."""
        self.tripleQuoutesWithinStrings = []
        # Do other syntax formatting
        for expression, nth, format in self.rules:
            match = expression.match(text, 0)
            index = match.capturedStart()
            if index >= 0:
                # if there is a string we check
                # if there are some triple quotes within the string
                # they will be ignored if they are matched again
                if expression.pattern() in [
                    r'"[^"\\]*(\\.[^"\\]*)*"',
                    r"'[^'\\]*(\\.[^'\\]*)*'",
                ]:
                    match = self.tri_single[0].match(text, index + 1)
                    innerIndex = match.capturedStart()

                    if innerIndex == -1:
                        match = self.tri_single[0].match(text, index + 1)
                        innerIndex = match.capturedStart()

                    if innerIndex != -1:

                        tripleQuoteIndexes = range(innerIndex, innerIndex + 3)
                        self.tripleQuoutesWithinStrings.extend(tripleQuoteIndexes)

            while index >= 0:
                # skipping triple quotes within strings
                if index in self.tripleQuoutesWithinStrings:
                    index += 1
                    match = expression.match(text, index)
                    index = match.capturedStart()
                    continue

                # We actually want the index of the nth match
                iterator = expression.globalMatch(text)
                ith = 0

                while iterator.hasNext():
                    ith += 1
                    match = iterator.next()

                    if ith == nth:
                        break

                length = match.capturedEnd() - match.capturedStart()
                self.setFormat(index, length, format)

                match = expression.match(text, index + length)
                index = match.capturedStart()

        self.setCurrentBlockState(0)

        # Do multi-line strings
        in_multiline = self.match_multiline(text, *self.tri_single)
        if not in_multiline:
            in_multiline = self.match_multiline(text, *self.tri_double)

    def match_multiline(self, text, delimiter, in_state, style):
        """Do highlighting of multi-line strings. ``delimiter`` should be a
        ``QRegularExpression`` for triple-single-quotes or triple-double-quotes, and
        ``in_state`` should be a unique integer to represent the corresponding
        state changes when inside those strings. Returns True if we're still
        inside a multi-line string when this function is finished.
        """
        # If inside triple-single quotes, start at 0
        if self.previousBlockState() == in_state:
            start = 0
            add = 0
        # Otherwise, look for the delimiter on this line
        else:
            match = delimiter.match(text)
            start = match.capturedStart()
            # skipping triple quotes within strings
            if start in self.tripleQuoutesWithinStrings:
                return False
            # Move past this match
            add = match.capturedLength()

        # As long as there's a delimiter match on this line...
        while start >= 0:
            # Look for the ending delimiter
            match = delimiter.match(text, start + add)
            end = match.capturedStart()
            # Ending delimiter on this line?
            if end >= add:
                length = end - start + add + match.capturedLength()
                self.setCurrentBlockState(0)
            # No; multi-line string
            else:
                self.setCurrentBlockState(in_state)
                length = len(text) - start + add
            # Apply formatting
            self.setFormat(start, length, style)
            # Look for the next match
            match = delimiter.match(text, start + length)
            start = match.capturedStart()

        # Return True if still inside a multi-line string, False otherwise
        if self.currentBlockState() == in_state:
            return True
        else:
            return False

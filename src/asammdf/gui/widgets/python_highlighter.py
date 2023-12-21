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
    _color = QtGui.QColor(color)

    _format = QtGui.QTextCharFormat()
    _format.setForeground(_color)
    if "bold" in style:
        _format.setFontWeight(QtGui.QFont.Weight.Bold)
    if "italic" in style:
        _format.setFontItalic(True)

    _format.setBackground(QtGui.QBrush(QtGui.QColor("#131314")))

    return _format


# Syntax styles that can be shared by all languages
STYLES = {
    "keyword": format("#ec6529"),
    "operator": format("#ff0000"),
    "brace": format("#ff0000"),
    "defclassname": format("#9e5fdd", "bold"),
    "defclass": format("#ec8549", "bold"),
    "string": format("#2f8f3d"),
    "string2": format("#2f8f3d"),
    "comment": format("#65c2e5", "italic"),
    "self": format("#93548b", "italic"),
    "numbers": format("#33ccff"),
    "builtins": format("#567ac5"),
}


class PythonHighlighter(QtGui.QSyntaxHighlighter):
    """Syntax highlighter for the Python language."""

    # Python keywords
    keywords = keyword.kwlist + getattr(keyword, "softkwlist", [])

    builtins = dir(builtins)

    # Python operators
    operators = [
        r"=",
        # Comparison
        r"==",
        "!=",
        "<",
        "<=",
        ">",
        ">=",
        # Arithmetic
        r"\+",
        "-",
        r"\*",
        "/",
        "//",
        r"\%",
        r"\*\*",
        # In-place
        r"\+=",
        "-=",
        r"\*=",
        "/=",
        r"\%=",
        # Bitwise
        r"\^",
        r"\|",
        r"\&",
        r"\~",
        ">>",
        "<<",
    ]

    # Python braces
    braces = [
        r"\{",
        r"\}",
        r"\(",
        r"\)",
        r"\[",
        r"\]",
    ]

    def __init__(self, parent: QtGui.QTextDocument) -> None:
        super().__init__(parent)

        # Multi-line strings (expression, flag, style)
        self.tri_single = (QtCore.QRegularExpression("'''"), 1, STYLES["string2"])
        self.tri_double = (QtCore.QRegularExpression('"""'), 2, STYLES["string2"])

        rules = []

        # Keyword, operator, and brace rules
        rules += [(rf"\b{w}\b", 0, STYLES["keyword"]) for w in PythonHighlighter.keywords]
        rules += [(o, 0, STYLES["operator"]) for o in PythonHighlighter.operators]
        rules += [(b, 0, STYLES["brace"]) for b in PythonHighlighter.braces]
        rules += [(rf"\b{b}\b", 0, STYLES["builtins"]) for b in PythonHighlighter.builtins]

        # All other rules
        rules += [
            # 'self'
            (r"\bself\b", 0, STYLES["self"]),
            # 'def' followed by an identifier
            (r"\bdef\b\s*(\w+)", 0, STYLES["defclassname"]),
            (r"\bdef\b", 0, STYLES["defclass"]),
            # 'class' followed by an identifier
            (r"\bclass\b\s*(\w+)", 0, STYLES["defclassname"]),
            (r"\bclass\b", 0, STYLES["defclass"]),
            # Numeric literals
            (r"\b[+-]?[0-9]+\b", 0, STYLES["numbers"]),
            (r"\b[+-]?0[xX][0-9A-Fa-f]+\b", 0, STYLES["numbers"]),
            (r"\b[+-]?[0-9]+(\.[0-9]+)\b", 0, STYLES["numbers"]),
            (r"\b[+-]?[0-9]+([eE][+-]?[0-9]+)\b", 0, STYLES["numbers"]),
            (r"\b[+-]?[0-9]+(\.[0-9]+)([eE][+-]?[0-9]+)\b", 0, STYLES["numbers"]),
            # Double-quoted string, possibly containing escape sequences
            (r'"[^"\\]*(\\.[^"\\]*)*"', 0, STYLES["string"]),
            # Single-quoted string, possibly containing escape sequences
            (r"'[^'\\]*(\\.[^'\\]*)*'", 0, STYLES["string"]),
            # From '#' until a newline
            (r"#[^\n]*", 0, STYLES["comment"]),
        ]

        # Build a QRegularExpression for each pattern
        self.rules = [(QtCore.QRegularExpression(pat), index, fmt) for (pat, index, fmt) in rules]

    def highlightBlock(self, text):
        """Apply syntax highlighting to the given block of text."""

        self.tripleQuoutesWithinStrings = []
        # Do other syntax formatting
        for expression, nth, format in self.rules:
            iterator = expression.globalMatch(text)

            while iterator.hasNext():
                match = iterator.next()

                length = match.capturedEnd() - match.capturedStart()
                index = match.capturedStart()
                self.setFormat(index, length, format)

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

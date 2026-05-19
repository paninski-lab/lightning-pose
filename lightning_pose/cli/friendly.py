"""Customized argparse classes with improved help formatting for the litpose CLI."""

import argparse
import shutil
import sys
from typing import Any


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, **kwargs: Any) -> None:
        """Initialize ArgumentParser with custom formatting and epilog."""
        super().__init__(
            formatter_class=_HelpFormatter,
            epilog="documentation: \n"
            "  https://lightning-pose.readthedocs.io/en/latest/source/user_guide/index.html",
            **kwargs,
        )
        self.is_sub_parser = False

    def print_help(self, with_welcome: bool = True, **kwargs: Any) -> None:
        """Print help text, optionally preceded by a welcome message.

        Args:
            with_welcome: if True and this is the top-level parser, print a welcome banner.
            **kwargs: additional keyword arguments forwarded to the parent ``print_help``.
        """
        if with_welcome and not self.is_sub_parser:
            print("Welcome to the lightning-pose CLI!\n")
        super().print_help(**kwargs)

    def error(self, message: str) -> None:  # type: ignore[override]
        """Print a formatted error message, display help, and exit.

        Args:
            message: the error message to display in red.
        """
        red = "\033[91m"
        end = "\033[0m"

        sys.stderr.write(red + f"error:\n{message}\n\n" + end)

        width = shutil.get_terminal_size().columns
        sys.stderr.write("-" * width + "\n")
        self.print_help(with_welcome=False)
        sys.exit(2)


class ArgumentSubParser(ArgumentParser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize ArgumentSubParser and mark it as a subparser."""
        super().__init__(*args, **kwargs)
        self.is_sub_parser = True


class _HelpFormatter(argparse.HelpFormatter):
    """Modifications on help text formatting for easier readability."""

    def _split_lines(self, text: str, width: int) -> list[str]:
        """Modified to preserve newlines and long words."""
        # First split into paragraphs, then wrap each separately:
        # https://docs.python.org/3/library/textwrap.html#textwrap.TextWrapper.replace_whitespace
        paragraphs = text.splitlines()
        import textwrap

        lines: list[str] = []
        for p in paragraphs:
            p_lines = textwrap.wrap(p, width, break_long_words=False, break_on_hyphens=False)
            # An empty paragraph should result in a newline.
            if not p_lines:
                p_lines = [""]
            lines.extend(p_lines)
        return lines

    def _fill_text(self, text: str, width: int, indent: str) -> str:
        """Fill text with indentation, preserving newlines.

        Args:
            text: the text to fill.
            width: maximum line width.
            indent: string prepended to each output line.

        Returns:
            Filled text with each line prefixed by ``indent``.
        """
        return "\n".join(indent + line for line in self._split_lines(text, width - len(indent)))

    def _format_action(self, *args: Any, **kwargs: Any) -> str:
        """Modified to add a newline after each argument, for better readability."""
        return super()._format_action(*args, **kwargs) + "\n"

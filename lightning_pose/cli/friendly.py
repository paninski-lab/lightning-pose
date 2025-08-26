import argparse
import shutil
import sys


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        super().__init__(
            formatter_class=_HelpFormatter,
            epilog="documentation: \n"
            "  https://lightning-pose.readthedocs.io/en/latest/source/user_guide/index.html",
            **kwargs,
        )
        self.is_sub_parser = False

    def format_help(self):
        """Modified to remove the "run_app" argument from the help text
        while it's still under development."""
        h = super().format_help()
        return h.replace(",run_app", "")

    def print_help(self, with_welcome=True, **kwargs):
        if with_welcome and not self.is_sub_parser:
            print("Welcome to the lightning-pose CLI!\n")
        super().print_help(**kwargs)

    def error(self, message):
        red = "\033[91m"
        end = "\033[0m"

        # Remove run_app while it's still in development.
        message = message.replace(", run_app", "")

        sys.stderr.write(red + f"error:\n{message}\n\n" + end)

        width = shutil.get_terminal_size().columns
        sys.stderr.write("-" * width + "\n")
        self.print_help(with_welcome=False)
        sys.exit(2)


class ArgumentSubParser(ArgumentParser):
    def __init__(self, *args, **kwargs):
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
            p_lines = textwrap.wrap(
                p, width, break_long_words=False, break_on_hyphens=False
            )
            # An empty paragraph should result in a newline.
            if not p_lines:
                p_lines = [""]
            lines.extend(p_lines)
        return lines

    def _fill_text(self, text: str, width: int, indent: str) -> str:
        return "\n".join(
            indent + line for line in self._split_lines(text, width - len(indent))
        )

    def _format_action(self, *args, **kwargs):
        """Modified to add a newline after each argument, for better readability."""
        return super()._format_action(*args, **kwargs) + "\n"

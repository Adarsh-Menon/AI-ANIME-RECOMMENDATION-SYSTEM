"""Application-wide custom exception with automatic source-location capture.

Wrapping a caught exception in `CustomException` attaches the file and line
where the failure occurred to the message, so a log line alone is enough to
locate the fault without re-reading the full traceback.

Typical use:
    try:
        risky_call()
    except Exception as e:
        raise CustomException("failed to load config", e)
"""

import sys                      # for sys.exc_info(), which exposes the traceback being handled
from types import TracebackType  # type-only import: annotates the traceback object


class CustomException(Exception):
    """Exception carrying the originating file and line alongside the message.

    Attributes:
        error_message (str): Fully formatted message — user message, the
            underlying error, and the source location — also passed to
            `Exception.__init__` so `args[0]` matches.
    """

    def __init__(self, message: str, error_detail: Exception = None) -> None:
        """Build the enriched message and initialise the base Exception.

        Args:
            message (str): Human-readable description of what was being
                attempted when the failure occurred.
            error_detail (Exception, optional): The original caught exception.
                Defaults to None when raising without an underlying cause.
        """
        # str: composed once at construction time, while the traceback is still live.
        self.error_message: str = self.get_detailed_error_message(message, error_detail)
        super().__init__(self.error_message)  # keeps str(e) and e.args consistent

    @staticmethod
    def get_detailed_error_message(message: str, error_detail: Exception) -> str:
        """Format the message with the location of the currently-handled exception.

        Args:
            message (str): Context supplied by the caller.
            error_detail (Exception): The original exception, or None.

        Returns:
            str: "<message> | Error: <detail> | File: <path> | Line: <n>"
        """
        # sys.exc_info() -> (type, value, traceback); index [2] takes just the traceback.
        exc_tb: TracebackType | None = sys.exc_info()[2]

        if exc_tb is not None:
            # Descend to the deepest frame — that's where the error actually
            # occurred, not where the enclosing try block was written.
            while exc_tb.tb_next is not None:
                exc_tb = exc_tb.tb_next
            file_name = exc_tb.tb_frame.f_code.co_filename  # str: source file of the failure
            line_number = exc_tb.tb_lineno                  # int: line within that file
        else:
            # No exception currently being handled (e.g. raised outside an except block).
            file_name, line_number = "Unknown File", "Unknown Line"

        return f"{message} | Error: {error_detail} | File: {file_name} | Line: {line_number}"

    def __str__(self) -> str:
        """Return the enriched message (what `print(e)` and most log calls show).

        Returns:
            str: The value of `self.error_message`.
        """
        return self.error_message
    
    
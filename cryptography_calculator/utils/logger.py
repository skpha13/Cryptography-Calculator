from logging import Logger


class LogStack:
    """Class that accumulates log messages and logs them as a single entry."""

    def __init__(self, logger: Logger):
        self.messages = []
        self.logger = logger

    def add_message(self, message: str = ""):
        """Add a new log message to the stack."""
        if message is None:
            message = ""

        self.messages.append(message)

    def empty_messages(self) -> None:
        self.messages.clear()

    def display_logs(self):
        """Display all accumulated logs as a single log entry."""

        if self.messages:
            full_message = "\n".join(self.messages)
            self.logger.info(full_message)
        else:
            self.logger.info("No logs to display.")

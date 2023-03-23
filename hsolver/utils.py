"""Utilities module"""


class ProgressPrinter:
    """Class for printing the progress of calculations

    :param min_value: Min iterator value
    :param max_value: Max iterator value
    :param title: Title of the process
    :param verbose: Set False to disable printing
    """

    def __init__(self, min_value: float, max_value: float, title: str, verbose: bool = True):
        self.min_value = min_value
        self.max_value = max_value
        self.title = title
        self.verbose = verbose

        self.current_progress = 0
        self.num_steps = 0
        self.print(new_line=False)

    def print(self, new_line: bool):
        """Update the line with current status

        :param new_line: Place the caret on the new line
        """
        if not self.verbose:
            return
        end = "\n" if new_line else ""
        print(f"\r{self.get_line()}", end=end)

    def get_percentage(self, value: float) -> int:
        """Computes the progress

        :param value: Current iterator value
        :return: The progress in percents
        """
        return round((value - self.min_value) / (self.max_value - self.min_value) * 100)

    def get_line(self):
        """Returns the line to print

        :return: Line to print
        """
        return f"{self.title}: {self.current_progress}% ({self.num_steps} steps)"

    def update(self, value: float):
        """Updates the progress with new iterator value

        :param value: Iterator value
        """
        self.num_steps += 1
        progress = self.get_percentage(value)
        if progress == self.current_progress:
            return
        self.current_progress = progress
        self.print(new_line=False)

    def stop(self):
        """Stops the progress printer"""
        self.print(new_line=True)

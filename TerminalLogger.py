import sys
import os
import traceback
from datetime import datetime


class TerminalLogger:

    def __init__(self, folder="TerminalLogs"):

        os.makedirs(folder, exist_ok=True)

        timestamp = datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

        self.filename = os.path.join(
            folder,
            f"experiment_{timestamp}.txt"
        )

        self.terminal = sys.stdout
        self.original_stderr = sys.stderr

        self.log_file = open(
            self.filename,
            "w",
            encoding="utf-8"
        )

    def write(self, message):

        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

        return len(message)

    def flush(self):

        self.terminal.flush()
        self.log_file.flush()

    def __getattr__(self, name):

        return getattr(self.terminal, name)

    def start(self):

        sys.stdout = self
        sys.stderr = self

        print(f"Experiment started: {datetime.now()}")
        print(f"Terminal log: {self.filename}")

    def stop(self):

        print(f"\nExperiment ended: {datetime.now()}")

        sys.stdout = self.terminal
        sys.stderr = self.original_stderr

        self.log_file.close()

        print(f"\nTerminal output saved to: {self.filename}")

    def run(self, function):

        self.start()

        try:

            function()

        except Exception:

            traceback.print_exc()
            raise SystemExit(1)

        finally:

            self.stop()
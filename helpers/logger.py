import sys
from pathlib import Path
import tempfile
import json
import datetime
from collections import OrderedDict
from typing import Optional, Any


DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40

DISABLED = 50


class KVWriter(object):

    def writekvs(self, kvs):
        raise NotImplementedError(f"do whatever with {kvs}")


class SeqWriter(object):

    def writeseq(self, seq):
        raise NotImplementedError(f"do whatever with {seq}")


class HumanOutputFormat(KVWriter, SeqWriter):

    def __init__(self, path_or_textiofilething):
        self.file = path_or_textiofilething
        if isinstance(path_or_textiofilething, Path):
            self.write_fn = self.file.write_text
            self.own_file = True
        else:  # must be a textiofilething or assert error
            assert_msg = (
                f"invalid type, got {type(self.file)};"
                "must at least have a `write` method"
            )
            assert hasattr(self.file, "write"), assert_msg
            self.write_fn = self.file.write
            self.own_file = False

    def writekvs(self, kvs):
        # Create strings for printing
        key2str = {}
        for (key, val) in kvs.items():
            valstr = f"{val:<8.3g}" if isinstance(val, float) else str(val)
            key2str[self._truncate(key)] = self._truncate(valstr)

        # Find max widths
        if len(key2str) == 0:
            # empty key-value dict; not sending warning nor stopping
            return
        keywidth = max(map(len, key2str.keys()))
        valwidth = max(map(len, key2str.values()))

        # Write out the data
        dashes = "-" * (keywidth + valwidth + 7)
        lines = [dashes]
        for (key, val) in key2str.items():
            key_space = " " * (keywidth - len(key))
            val_space = " " * (valwidth - len(val))
            lines.append(f"| {key}{key_space} | {val}{val_space} |")
        lines.append(dashes)
        self.write_fn("\n".join(lines) + "\n")

    @staticmethod
    def _truncate(s):
        thres_len = 43
        return s[:40] + "..." if len(s) > thres_len else s

    def writeseq(self, seq):
        for arg in seq:
            self.write_fn(arg)
        self.write_fn("\n")


class JSONOutputFormat(KVWriter):

    def __init__(self, path):
        self.file = path

    def writekvs(self, kvs):
        for k, v in kvs.items():
            if hasattr(v, "dtype"):
                kvs[k] = float(v.tolist())
        self.file.write_text(json.dumps(kvs), newline="\n")


class CSVOutputFormat(KVWriter):

    def __init__(self, path):
        self.file = path
        self.keys = []
        self.sep = ","

    def writekvs(self, kvs):
        # Add our current row to the history
        extra_keys = kvs.keys() - self.keys
        if extra_keys:
            self.keys.extend(extra_keys)
            with self.file.open() as f:
                lines = f.readlines()
            for (i, k) in enumerate(self.keys):
                if i > 0:
                    self.file.write_text(",")
                self.file.write_text(k)
            self.file.write("\n")
            for line in lines[1:]:
                self.file.write_text(line[:-1])
                self.file.write_text(self.sep * len(extra_keys))
                self.file.write_text("", newline="\n")
        for (i, k) in enumerate(self.keys):
            if i > 0:
                self.file.write_text(",")
            v = kvs.get(k)
            if v:
                self.file.write_text(str(v))
        self.file.write_text("", newline="\n")


def make_output_format(formatting, dir_, suffix=""):
    dir_ = Path(dir_)
    dir_.mkdir(parents=True, exist_ok=True)
    match formatting:  # python version >3.10 needed
        case "stdout":
            return HumanOutputFormat(sys.stdout)
        case "log":
            return HumanOutputFormat(dir_ / f"log{suffix}.txt")
        case "json":
            return JSONOutputFormat(dir_ / f"progress{suffix}.json")
        case "csv":
            return CSVOutputFormat(dir_ / f"progress{suffix}.csv")
        case _:
            raise ValueError(f"unknown formatting specified: {formatting}")


# Frontend

def logkv(key, val):
    """Log a key-value pair with the current logger.
    This method should be called every iteration for the quantities to monitor.
    """
    if Logger.CURRENT is not None:
        Logger.CURRENT.logkv(key, val)


def logkvs(d):
    """Log a dictionary of key-value pairs with the current logger"""
    for (k, v) in d.items():
        logkv(k, v)


def dumpkvs():
    """Write all the key-values pairs accumulated in the logger
    to the write ouput format(s) (then flush the dictionary.
    """
    if Logger.CURRENT is not None:
        Logger.CURRENT.dumpkvs()


def getkvs():
    """Return the key-value pairs accumulated in the current logger"""
    if Logger.CURRENT is not None:
        return Logger.CURRENT.name2val
    return None


def log(*args, level=INFO):
    """Write the sequence of args, with no separators, to the console
    and output files (if an output file has been configured).
    """
    if Logger.CURRENT is not None:
        Logger.CURRENT.log(*args, level=level)


# Create distinct functions fixed at all the values taken by `level`

def debug(*args):
    log(*args, level=DEBUG)


def info(*args):
    log(*args, level=INFO)


def warn(*args):
    log(*args, level=WARN)


def error(*args):
    log(*args, level=ERROR)


def set_level(level):
    """Set logging threshold on current logger"""
    if Logger.CURRENT is not None:
        Logger.CURRENT.set_level(level)


def get_dir():
    """Get directory to which log files are being written"""
    if Logger.CURRENT is not None:
        return Logger.CURRENT.get_dir()
    return None


# Define aliases for higher-level language
record_tabular = logkv
dump_tabular = dumpkvs


# Backend

class Logger(object):

    DEFAULT: Optional[Any] = None
    CURRENT: Optional[Any] = None

    def __init__(self, dir_, output_formats):
        self.name2val = OrderedDict()  # values this iteration
        self.level = INFO
        self.dir_ = dir_
        self.output_formats = output_formats

    def logkv(self, key, val):
        self.name2val.update({key: val})

    def dumpkvs(self):
        if self.level == DISABLED:
            return
        for output_format in self.output_formats:
            if isinstance(output_format, KVWriter):
                output_format.writekvs(self.name2val)
        self.name2val.clear()

    def log(self, *args, level=INFO):
        if self.level <= level:
            # If the current logger level is higher than
            # the `level` argument, don"t print to stdout
            self._log(args)

    def set_level(self, level):
        self.level = level

    def get_dir(self):
        return self.dir_

    def _log(self, args):
        for output_format in self.output_formats:
            if isinstance(output_format, SeqWriter):
                output_format.writeseq(map(str, args))


def configure(dir_=None, format_strs=None):
    """Configure logger (called in configure_default_logger)"""
    if dir_ is None:
        dir_ = Path(tempfile.gettempdir())
        dir_ /= datetime.datetime.now(tz=datetime.timezone.utc).strftime(
            "%Y-%m-%d-%H-%M-%S-%f_temp_log")
    else:
        # make sure the directory is provided as a string
        assert isinstance(dir_, str), f"wrong type: {type(dir_)} > must be str"
        # make sure the provided directory exists
        Path(dir_).mkdir(parents=True, exist_ok=True)
    if format_strs is None:
        format_strs = []
    # setup the output formats
    output_formats = [make_output_format(f, dir_) for f in format_strs]
    Logger.CURRENT = Logger(dir_=dir_, output_formats=output_formats)


def configure_default_logger():
    """Configure default logger"""
    # Write to stdout by default
    format_strs = ["stdout"]
    # Configure the current logger
    configure(format_strs=format_strs)  # makes Logger.CURRENT be not None anymore
    # Logging successful configuration of default logger
    log("configuring default logger for each worker (logging to stdout only by default)")
    # Define the default logger with the current logger
    Logger.DEFAULT = Logger.CURRENT


def reset():
    if Logger.CURRENT is not Logger.DEFAULT:
        Logger.CURRENT = Logger.DEFAULT
        log("resetting logger")


# Configure a logger by default
configure_default_logger()

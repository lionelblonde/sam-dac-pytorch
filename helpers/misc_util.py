import time
import argparse
from typing import Optional
from enum import IntEnum
from contextlib import contextmanager

from helpers import logger


def zipsame(*seqs):
    """Verify that all the sequences in `seqs` are the same length, then zip them together"""
    assert seqs, "empty input sequence"
    ref_len = len(seqs[0])
    assert all(len(seq) == ref_len for seq in seqs[1:])
    return zip(*seqs)


def prettify_time(seconds: int):
    """Print the number of seconds in human-readable format.
    e.g. "2 days", "2 hours and 37 minutes", "less than a minute".
    """
    minutes = seconds // 60
    seconds %= 60
    hours = minutes // 60
    minutes %= 60
    days = hours // 24
    hours %= 24

    def helper(count, name):
        trailing_s = "s" if count > 1 else ""
        return f"{count} {name}{trailing_s}"

    # display only the two greater units (days and hours, hours and minutes, minutes and seconds)
    if days > 0:
        message = helper(days, "day")
        if hours > 0:
            message += " and " + helper(hours, "hour")
        return message
    if hours > 0:
        message = helper(hours, "hour")
        if minutes > 0:
            message += " and " + helper(minutes, "minute")
        return message
    if minutes > 0:
        return helper(minutes, "minute")

    # finally, if none of the previous conditions is valid
    return "less than a minute"


def boolean_flag(parser: argparse.ArgumentParser,
                 name: str,
                 *,
                 default: Optional[bool] = False,
                 hint: Optional[str] = None):
    """Add a boolean flag to argparse parser"""
    dest = name.replace("-", "_")
    parser.add_argument("--" + name, action="store_true", default=default, dest=dest, help=hint)
    parser.add_argument("--no-" + name, action="store_false", dest=dest)


class Colors(IntEnum):
    GRAY = 30
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    MAGENTA = 35
    CYAN = 36
    WHITE = 37
    CRIMSON = 38


def colorize(string: str, color: Colors, *, bold=False, highlight=False) -> str:
    attr = []
    num: int = Colors[color.name]
    if highlight:
        num += 10
    bold_indicator = "1" if bold else ""
    attr = f"{num};{bold_indicator}"
    return f"\x1b[{attr}m{string}\x1b[0m"


@contextmanager
def timed(text: str):
    pre_mess = f"::{text}::"
    logger.info(colorize(pre_mess, Colors.GRAY))
    tstart = time.time()
    yield
    tot_time = time.time() - tstart
    post_mess = f"done in {tot_time:.3f} seconds".rjust(50, ":")
    logger.info(colorize(post_mess, Colors.CYAN))


def log_iter_info(cur_iter: int, tot_num_iters: int, tstart: float):
    """Display the current iteration and elapsed time"""
    elapsed = prettify_time(int(time.time() - tstart))
    mess = f"iter [{cur_iter}/{tot_num_iters}] <- elapsed time: {elapsed}".rjust(75, ":")
    logger.info(colorize(mess, Colors.CRIMSON))

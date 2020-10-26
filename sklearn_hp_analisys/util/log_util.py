import logging


def log_and_print(msg, log_level):
    getattr(logging, log_level)(msg)
    print(msg)

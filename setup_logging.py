import logging


def setup_logging(loglevel):
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)

    logger = logging.getLogger("program_scheduling")
    logger.setLevel(numeric_level)

    ch = logging.StreamHandler()
    ch.setLevel(numeric_level)
    # to include time, you could add `%(asctime)s`
    formatter = logging.Formatter('(%(module)s) %(levelname)s: %(message)s')
    ch.setFormatter(formatter)

    logger.addHandler(ch)

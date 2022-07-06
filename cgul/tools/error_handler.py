import logging
import typing as T


def error_handler(
    message: str,
    logger: logging.Logger,
    err: T.Union[Exception, str] = "",
    warn_extra: str = "",
    error_mode: str = "warn",
) -> None:
    if error_mode == "ignore":
        pass
    elif error_mode == "raise":
        if err:
            message = f"{message}\nTraceback:\n{err}"
        raise RuntimeError(message)
    else:
        if warn_extra:
            message = f"{message} {warn_extra}"
        if err:
            message = f"{message}\nTraceback:\n{err}"
        logger.warning(message)

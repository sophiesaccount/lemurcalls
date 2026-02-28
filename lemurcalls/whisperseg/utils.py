import os


RATIO_DECODING_TIME_STEP_TO_SPEC_TIME_STEP = 2


def create_if_not_exists(folder):
    """Create directory if it does not exist and return its path.

    Args:
        folder: Path to the directory.

    Returns:
        str: The same folder path.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


def get_lr(optimizer):
    """Return the current learning rate(s) of the optimizer.

    Args:
        optimizer: A torch optimizer.

    Returns:
        list: Learning rate for each param group.
    """
    return [param_group["lr"] for param_group in optimizer.param_groups]

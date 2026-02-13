import os


RATIO_DECODING_TIME_STEP_TO_SPEC_TIME_STEP = 2


def create_if_not_exists(folder):
    """Create directory if it does not exist.

    Args:
        folder: Path to the directory.

    Returns:
        The folder path.
    """
    if not os.path.exists(folder):
        os.makedirs( folder )
    return folder


def get_lr(optimizer):
    """Return current learning rate(s) from an optimizer.

    Args:
        optimizer: PyTorch optimizer.

    Returns:
        List of learning rates for each param group.
    """
    return [param_group['lr'] for param_group in optimizer.param_groups ]
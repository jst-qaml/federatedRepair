"""Berkley Deep Drive (BDD100K).

cf. https://bdd-data.berkeley.edu/index.html
"""

from pathlib import Path

from repair.core import dataset

from . import prepare, prepare_exp2


class BDDObjects(dataset.RepairDataset):
    """API for DNN with BDD."""

    def __init__(self):
        """Initialize."""
        self.target_label = "weather"

    def _get_input_shape(self):
        """Set the input_shape and classes of BDD."""
        return (32, 32, 3), 13

    def prepare(self, input_dir: Path, output_dir: Path, divide_rate, random_state):
        """Prepare BDD objecets dataset.

        Parameters
        ----------
        input_dir : Path
        output_dir : Path
        divide_rate : float
        random_state : int, optional

        """
        # Make output directory if not exist
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        if "exp2" in str(input_dir):
            prepare_exp2.prepare(input_dir, output_dir, divide_rate, random_state)
        else:
            prepare.prepare(input_dir, output_dir, divide_rate, random_state)

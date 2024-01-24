"""Command Line Interface (CLI).

handling commands with fire.

"""

import importlib
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np

import cv2
import fire
from tqdm import tqdm

from repair.core.model import load_model_from_tf, target, test, train

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["KMP_AFFINITY"] = "noverbose"

tf = importlib.import_module("tensorflow")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class CLI:
    """Command Line Interface for automated repair of DNN models.

    Handling commands as defined below.

    """

    def prepare(
        self,
        dataset,
        input_dir="inputs/",
        output_dir="outputs/",
        divide_rate=0.2,
        random_state=None,
        **kwargs,
    ):
        """Prepare dataset to train.

        Parameters
        ----------
        dataset : str
           Dataset name
        input_dir : str, default="inputs"
            Path to input directory containing train and test datasets
        output_dir : str, default="outputs"
            Path to output directory
        divide_rate : float, default=0.2
            The ratio of dividing training data for using repair
        random_state : int, optional
            The seed value of random sampling
            for reproducibility of dividing training data

        """
        dataset = _get_dataset(dataset, kwargs=kwargs)
        dataset.prepare(Path(input_dir), Path(output_dir), divide_rate, random_state)

    def train(
        self,
        model="base_cnn_model",
        epochs=5,
        gpu=False,
        data_dir="outputs/",
        output_dir="outputs/",
        **kwargs,
    ):
        """Train dataset prepared previously.

        Parameters
        ----------
        model : str, default="base_cnn_model"
            Model name
        epochs : int, default=5
            The number of epochs
        gpu : bool, default=False
            True or False
        data_dir : str, default="outputs"
            Path to directory containing train datasets directory
        output_dir : str, default="outputs"
            Path to output directory
        **kwargs
            Additional options specific for each models.

        """
        kwargs["data_dir"] = data_dir
        model = _get_model(model, kwargs=kwargs)
        train(model, epochs, Path(data_dir), Path(output_dir), gpu)

    def test(
        self,
        batch_size=32,
        model_dir="outputs/",
        data_dir="outputs/",
        target_data="test.h5",
    ):
        """Test DNN model generated previously.

        Parameters
        ----------
        batch_size : str, default=32
            A size of batches
        model_dir : str, default="outputs"
            Path to directory containing DNN model
        data_dir : str, default="outputs"
            Path to directory containing test dataset
        target_data : str, default="test.h5"
            Filename for target dataset

        Returns
        -------
        float
            accuracy

        """
        score = test(Path(model_dir), Path(data_dir), target_data, batch_size)

        return score

    def target(self, batch_size=32, model_dir=r"outputs/", data_dir=r"outputs/"):
        """Find target dataset to reproduce failures on DNN models.

        Parameters
        ----------
        batch_size : int, default=32
            A size of batches
        model_dir : str, default="outputs"
            Path to directory containing DNN model
        data_dir : str, default="outputs"
            Path to directory containing test dataset

        """
        target(Path(model_dir), Path(data_dir), batch_size)

    def localize(
        self,
        method,
        model_dir="outputs",
        target_data_dir="outputs/negative/0",
        verbose=1,
        **kwargs,
    ):
        """Repair DNN model with target and test data.

        Parameters
        ----------
        method : str
            Repair method name
        model_dir : str, default="outputs"
            Path to directory containing DNN model
        target_data_dir : str, default="outputs/negative/0"
            Path to directory containing target test dataset
        verbose : int, default=1
            Log level
        kwargs
            `batch_size`:
            a size of batches
            `train_data_dir`:
            (TBA) path to directory containing train dataset
            `num_grad`:
            (for Arachne only) number of neural weight candidates
            to choose based on gradient loss

        """
        # Parse optionals

        pos_input_dir = None
        output_dir=None
        pos_data = None

        try:
            if kwargs["train_data_dir"] is not None:
                raise NotImplementedError("To be implemented: 'train_data_dir'")  # noqa
        except KeyError:
            pass

        method = _get_repair_method(method, kwargs)
        print(kwargs)

        if kwargs['positive_inputs_dir'] is not None:
            pos_input_dir = kwargs['positive_inputs_dir']
            print(pos_input_dir)
        if kwargs['output_dir'] is not None:
            output_dir = kwargs['output_dir']


        model = load_model_from_tf(Path(model_dir))
        target_test_data = method.load_input_neg(Path(target_data_dir))

        if pos_input_dir is not None:
            pos_data = method.load_input_pos(Path(pos_input_dir))
        else:
            raise Exception("Missing positive inputs for FL")

        method.localize(model, target_test_data, Path(output_dir), verbose,input_pos = pos_data,model_dir=model_dir)

    def optimize(
        self,
        method,
        model_dir="outputs",
        target_data_dir="outputs/negative/0",
        positive_inputs_dir="outputs/positive",
        output_dir=None,
        verbose=1,
        **kwargs,
    ):
        """Optimize neuron weights to repair.

        Parameters
        ----------
        method : str
            Repair method name
        model_dir : str, default="outputs"
            Path to directory containing DNN model
        target_data_dir : str, default="outputs/negative/0"
            Path to directory containing dataset for unexpected behavior
        positive_inputs_dir : str, default="outputs/positive"
            Path to directory containing dataset for correct behavior
        output_dir : str, optional
            Path to directory where analysis results are saved
        verbose : Log level
        kwargs
            `batch_size`: a size of batches

        """
        # Instantiate
        method = _get_repair_method(method, kwargs)
        if output_dir is None:
            output_dir = target_data_dir

        # Load
        model = load_model_from_tf(Path(model_dir))
        weights = method.load_weights(Path(target_data_dir))
        target_data = method.load_input_neg(Path(target_data_dir))
        positive_inputs = method.load_input_pos(Path(positive_inputs_dir))

        if 'num_gen' in kwargs:
            num_gen = int(kwargs['num_gen'])
        else:
            num_gen = 2

        if 'num_pop' in kwargs:
            num_pop = int(kwargs['num_pop'])
        else:
            num_pop = 10

        # Analyze
        method.optimize(
            model,
            model_dir,
            weights,
            target_data,
            positive_inputs,
            Path(output_dir),
            verbose,
            num_gen=num_gen,
            num_pop=num_pop
        )

    def evaluate(
        self,
        dataset,
        method,
        model_dir="outputs",
        target_data_dir="outputs/negative/0",
        positive_inputs_dir="outputs/positive",
        output_dir=None,
        num_runs=10,
        verbose=1,
        **kwargs,
    ):
        """Evaluate repairing performance.

        Parameters
        ----------
        dataset : str
            Dataset name
        method : str
            Repair method name
        model_dir : str, default="outputs"
            Path to directory containing DNN model
        target_data_dir : str, default="outputs/negative/0"
            Path to directory containing dataset for unexpected behavior
        positive_inputs_dir : str, default="outputs/positive"
            Path to directory containing dataset for correct behavior
        output_dir : str, optional
            Path to directory where analysis results are saved
        num_runs : int, default=10
            Number of repair attempts
        verbose : int, default=1
            Log level
        **kwargs
            `batch_size` : int, default=32
                A size of batches

        """
        # Instantiate
        # NOTE: currently dataset is used to just log its name at Arachne
        dataset = _get_dataset(dataset)
        method = _get_repair_method(method, kwargs)
        if output_dir is None:
            output_dir = target_data_dir

        # Load
        target_data = dataset.load_repair_data(target_data_dir)
        positive_inputs = dataset.load_repair_data(positive_inputs_dir)

        # Evaluate
        method.evaluate(
            dataset,
            Path(model_dir),
            target_data,
            Path(target_data_dir),
            positive_inputs,
            Path(positive_inputs_dir),
            Path(output_dir),
            num_runs,
            verbose,
        )

    def utils(self, dataset, **kwargs):
        """Utilities.

        This command is for dataset-specific functions.

        Parameters
        ----------
        dataset : str
            Dataset name
        **kwargs
            Additional options specific for each utilities

        """
        dataset = _get_dataset(dataset)
        dataset.utils(kwargs)

    def create_category_dir(
        self, label_file, image_dir, output_dir="categories", **kwargs
    ):
        """Extract all the object images from the original BDD images.

        This function creates image directories by the object categories.

        Parameters
        ----------
        label_file : str
            Path to the label file
            (e.g. inputs/labels/det_20/det_train.json)
        image_dir : str
            Path to the image directory
            (e.g. inputs/images/100k/train)
        output_dir : str, default="categories"
            Path to the output directory
        **kwargs : dict
            Extra args

        """
        with open(label_file) as f:
            labels = json.load(f)

        extract_results = {}
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()

        for image in tqdm(labels):
            if "labels" in image:
                for label in image["labels"]:
                    output_category = output_dir / label["category"]
                    if label["category"] not in extract_results:
                        extract_results[label["category"]] = {
                            "extracted": 0,
                            "removed": 0,
                        }
                    if not output_category.exists():
                        output_category.mkdir()

                    _record_result(
                        image,
                        Path(image_dir),
                        label,
                        output_dir,
                        output_category,
                        extract_results,
                        kwargs,
                    )
            else:
                print("There are no labels in " + image["name"])
        _create_image_info(output_dir)

    def create_image_subset(
        self,
        category_dir,
        output_dir="outputs",
        info_file=None,
        num=None,
        excluded_labels=None,
        category_min=None,
        random_state=None,
        resize_to=None,
        **kwargs,
    ):
        """Create image_info subset used for extracting image subset.

        Parameters
        ----------
        category_dir : str
            Path to directory created by create_category_dir.
        output_dir : str, default="outputs"
            Path to directory where extracted images are saved.
        info_file : str, optional
            Path to a file that stores arguments.
        num : int, optional
            Minimum number of images to be extracted.
        excluded_labels : optional
            Labels to be excluded.
        category_min : int, optional
            Minimum number of images to be extracted in each category.
        random_state : int, optional
            Seed value of random image sampling.
        resize_to : int, optional
            length of one side of images to be resized.
        **kwargs
            Extra config

        """
        category_dir = Path(category_dir)
        output_dir = Path(output_dir)
        with open(category_dir / "image_info.json") as f:
            image_info = json.load(f)

        if category_dir == output_dir:
            raise ValueError("Output directory should not be same as category direcotry.")

        if output_dir.is_dir():
            shutil.rmtree(output_dir)

        output_dir.mkdir()

        if info_file is not None:
            with open(info_file) as f:
                extract_args = json.load(f)["args"]
                arg_num = extract_args["num"]
                arg_category_min = extract_args["category_min"]
                arg_random_state = extract_args["random_state"]
                arg_resize_to = extract_args["resize_to"]
                arg_exc_labels = extract_args["excluded_labels"]
        else:
            arg_num = num or 5000
            arg_category_min = category_min or 0
            arg_random_state = random_state or 0
            arg_resize_to = resize_to
            arg_exc_labels = []

        if excluded_labels is not None:
            arg_exc_labels = excluded_labels.split(",")

        sample_info_list, nums = _create_sublist_intend(
            image_info, arg_exc_labels, arg_num, arg_category_min, arg_random_state
        )
        _copy_images(sample_info_list, category_dir, output_dir, arg_resize_to)

        extract_args = {
            "num": arg_num,
            "category_min": arg_category_min,
            "random_state": arg_random_state,
            "resize_to": arg_resize_to,
            "excluded_labels": arg_exc_labels,
        }

        extract_info = {"args": extract_args, "results": nums, "images": sample_info_list}
        with open(output_dir / "image_info.json", "w") as f:
            json.dump(extract_info, f, indent=4)


def _adjast_extract_img(img, label):
    """Extract an object based on the label info.

    Parameters
    ----------
    img : np.ndarray
        Target image
    label : dict
        Object's position on the image

    Returns
    -------
    np.ndarray
        Extracted image

    """
    height, width, channels = img.shape[:3]
    x1 = int(label["box2d"]["x1"])
    x2 = int(label["box2d"]["x2"])
    y1 = int(label["box2d"]["y1"])
    y2 = int(label["box2d"]["y2"])

    # expand the extracted img to adjast the aspect ratio
    width_sub = (x2 - x1) - (y2 - y1)
    if width_sub > 0:
        y2 = int(y2 + width_sub / 2)
        y1 = int(y1 - width_sub / 2)
        if y2 > height:
            y1 -= y2 - height
            y2 = height
        elif y1 < 0:
            y2 -= y1
            y1 = 0
    else:
        x2 = int(x2 - width_sub / 2)
        x1 = int(x1 + width_sub / 2)
        if x2 > width:
            x1 -= x2 - width
            x2 = width
        elif x1 < 0:
            x2 -= x1
            x1 = 0

    return img[y1:y2, x1:x2]


def _record_result(
    image,
    image_dir: Path,
    label,
    output_dir: Path,
    output_category_dir: Path,
    extract_results,
    kwargs,
):
    """Extracts and stores an object from the input image info.

    Parameters
    ----------
    image : np.ndarray
        Info of the image from which objects are extracted
    image_dir : Path
        Directory where the actual image stored
    label :
        Dict object that includes object info in the input image
    output_dir : Path
        Directory that final results to be stored
    output_category_dir : Path
        Label of the input image
    extract_results :
        Dict object that stores results
    kwargs :
        Options while extracting

    """
    # Option1: Whether to exclude truncated objects
    if "remove_truncated" in kwargs:
        remove_truncated = True
    else:
        remove_truncated = False

    # Option2: Whether to exclude occluded objects
    if "remove_occluded" in kwargs:
        remove_occluded = True
    else:
        remove_occluded = False

    # Option3: Whether to exclude objects under the minimum size
    if "min_area" in kwargs:
        min_area = kwargs["min_area"]
    else:
        min_area = -1

    image_file_name = image["name"]
    img = cv2.imread(str(image_dir / image_file_name))
    extract_img_name = f"{label['id']}_{image_file_name}"
    output_filename = output_category_dir / extract_img_name

    if not output_filename.exists():
        extract_img = _adjast_extract_img(img, label)
        height, width, channels = extract_img.shape[:3]
        area = width * height

        if extract_img is not None:
            # Check whether the target object is to be excluded
            if (
                (remove_occluded and label["attributes"]["occluded"])
                or (remove_truncated and label["attributes"]["truncated"])
                or (area < min_area and min_area != -1)
            ):
                # Store the object image in the directory named removed
                output_category_dir = output_category_dir / "removed"
                output_filename = output_category_dir / extract_img_name
                if not output_category_dir.exists():
                    output_category_dir.mkdir()
                extract_results[label["category"]]["removed"] += 1
            else:
                # Store the object image in its category directory
                extract_results[label["category"]]["extracted"] += 1
            cv2.imwrite(str(output_filename), extract_img)
            with open(output_dir / "extract_results.json", "w") as f:
                json.dump(extract_results, f, indent=4)


def _create_image_info(category_dir: Path):
    """Create info file for extracting image subset.

    Parameters
    ----------
    category_dir : Path
        Directory containing categories and their images

    """
    image_dirs = list(filter(lambda d: d.is_dir(), category_dir.iterdir()))
    image_json_list = []
    value_mean_list = []
    area_list = []
    for image_dir in tqdm(image_dirs, desc="Process category"):
        image_files = list(filter(lambda f: f.is_file(), image_dir.iterdir()))

        for image_file in tqdm(image_files, desc="Extract image info"):
            img = cv2.imread(str(image_file))
            if img is None:
                print(f"{image_file} is not a image file")
                continue

            height, width, channels = img.shape[:3]
            area = width * height
            value_mean = _extract_value_mean(img)
            image_json_list.append(
                {
                    "name": image_file.name,
                    "label": image_dir.name,
                    "value_mean": value_mean,
                    "area": area,
                }
            )
            area_list.append(area)
            value_mean_list.append(value_mean)
        with open(category_dir / "image_info.json", "w") as f:
            json.dump(image_json_list, f, indent=4)


def _copy_images(image_info, category_dir: Path, output_dir: Path, resize_to=None):
    """Copy the extracted images.

    Parameters
    ----------
    image_info
        Info of the target images
    category_dir : Path
        Source directory
    output_dir : Path
        Target directory
    resize_to : int, optional
        image size to be resized

    """
    if resize_to is not None:
        for info in image_info:
            img = cv2.imread(str(category_dir / info["label"] / info["name"]))
            resize_img = cv2.resize(img, dsize=(resize_to, resize_to))
            cv2.imwrite(str(output_dir / info["name"]), resize_img)
    else:
        for info in image_info:
            shutil.copy(
                category_dir / info["label"] / info["name"],
                output_dir / info["name"],
            )


def _create_sublist_intend(
    image_info,
    excluded_labels=None,
    dict_size=5000,
    category_min=0,
    random_state=None,
):
    """Sampling images.

    Parameters
    ----------
    image_info : np.ndarray
        Info of the target images
    excluded_labels : dict, optional
        Labels to be excluded
    dict_size : int, default=5000
        The number images to be extracted
    category_min : int, default=0
        The minimum number of images in each category
    random_state : int, optional
        Seed value of random sampling

    """
    sample_info_list = []
    categ_dict = {}
    label_num_dict = {}
    if excluded_labels is None:
        excluded_labels = {}
    for info in image_info:
        if info["label"] not in excluded_labels:
            if info["label"] in categ_dict:
                categ_dict[info["label"]].append(info)
            else:
                categ_dict[info["label"]] = [info]
    print("|               label|   num|")
    print("|--------------------|------|")
    for categ in categ_dict:
        info_sublist = categ_dict[categ]
        target_num = min(
            max(
                int(len(info_sublist) / len(image_info) * dict_size) + 1,
                int(category_min),
            ),
            len(info_sublist),
        )
        if random_state is not None:
            random.seed(random_state)
        sample_info_list += random.sample(info_sublist, target_num)
        print("|%20s|%6d|" % (categ, target_num))
        label_num_dict[categ] = target_num

    print("|               total|%6d|" % (len(sample_info_list)))
    return sample_info_list, label_num_dict


def _extract_value_mean(img):
    """Extract value mean.

    Parameters
    ----------
    img : np.ndarray
        Target image

    Returns
    -------
    mean

    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    return np.mean(v)


def _get_dataset(name, **kwargs):
    """Get dataset.

    Parameters
    ----------
    name : str
        Identifier of dataset
    **kwargs : dict
        Extra config available for specific dataset
           (e.g. target_label for BDD dataset)

    Returns
    -------
    repair.core.dataset.RepairDataset
        Dataset

    """
    dataset = _load_instance("dataset", "dataset", name, Path("settings"))
    if dataset is None:
        errmsg = f"Invalid dataset: {name!r}"
        raise KeyError(errmsg)
    dataset.set_extra_config(**kwargs)
    return dataset


def _get_model(name, **kwargs):
    """Get model.

    Parameters
    ----------
    name : str
        Identifier of model

    Returns
    -------
    repair.core.model.RepairModel
        Model

    """
    model = _load_instance("model", "model", name, Path("settings"))
    if model is None:
        errmsg = f"Invalid model: {name!r}"
        raise KeyError(errmsg)
    model.set_extra_config(**kwargs)
    return model


def _get_repair_method(name, kwargs):
    """Get repair method.

    Parameters
    ----------
    name : str
        Identifier of repair method
    kwargs : dict
        Repair method options

    Returns
    -------
    Repair method

    """
    # Instantiate repair method
    method = _load_instance("method", "methods", name, Path("settings"))
    if method is None:
        raise AttributeError(f"Invalid method: {name!r}")
    # Set optional parameters
    method.set_options(**kwargs)

    return method


def _load_instance(key, parent_module, name, settings_dir: Path):
    """Load settings.

    Loading a instance of repair methods, datasets, or models
    Settings are in "settings.json" under a given directory.

    Parameters
    ----------
    key : str
        Key of settings.json (e.g. method)
    parent_module : str
        Directory containing modules (e.g. repair)
    name : str
        Available option name (e.g. Arachne for method)
    settings_dir : str
        Path to directory containing settings.json

    Returns
    -------
        Instance of repair methods, datasets, or models

    """
    importlib.invalidate_caches()
    file = settings_dir / "settings.json"
    with open(file) as f:
        try:
            settings = json.load(f)
            if name in settings[key]:
                name = settings[key][name]
            _names = name.split(".")
            module_name = name.split("." + _names[-1])[0]
            methodclass = importlib.import_module(
                f".{module_name}", f"repair.{parent_module}"
            )

            instance = getattr(methodclass, _names[-1])()
        # TODO: refactor after this commit
        except BaseException as e:  # noqa: BLE001
            print(e)
            return None
        else:
            return instance


def main():
    """Entrypoint of repair cli."""
    fire.Fire(CLI)

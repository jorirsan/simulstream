import argparse
import logging

import simulstream
from simulstream.metrics.logger import setup_metrics_logger, METRICS_LOGGER
from simulstream.server.message_processor import MessageProcessor
from simulstream.server.speech_processors import build_speech_processor, SpeechProcessor
from pathlib import Path

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
LOGGER = logging.getLogger('simulstream.inference')

#Downloads and prepares evaluation datasets una funcion de preparar dataset

def get_earnings21(dataset_path: Path):
    
def main(dataset: str, dataset_path: Path):
    dataset_path.mkdir(parents=True, exist_ok=True)

    fn_name = f"get_{dataset}"
    fn = globals().get(fn_name)

    if fn is None:
        raise ValueError(
            f"No dataset preparation function found for '{dataset}'. "
            f"Expected a function named '{fn_name}'."
        )
    LOGGER.info(f"Preparing dataset '{dataset}' at {dataset_path}")
    fn(dataset_path)

def cli_main():
    """
    Simulstream command-line interface (CLI) to download and prepare various evaluation datasets.
    This will be created
    Various datasets recipies are avaialble to download and prepare the folders and expected YAML formats for long-form ASR/ST and short form ST are given.

    """
    LOGGER.info(f"Simulstream version: {simulstream.__version__}")
    parser = argparse.ArgumentParser("simulstream_inference")
    parser.add_argument("--dataset", type=str, required=True,
        choices = [
            #Long form (>10 mins on average)

            #Long form
            "earnings21",
            "earnings22",
            "tedlium",

            #Short Form ST
            "covost2",
            "europarl_st",

            #Long Form ST
            "acl_6060",
            "mcif",
            "custom"
            
        ],
        help=f'Get a dataset. If "custom" is chosen, a module path to the function will be executed. Function name is expected to follow the pattern get_$DATASET ')
    parser.add_argument("--dataset-path", type=Path, default=Path("./datasets"), help="Path where to save datasets")
    #Pass vars instead of args so that we can define in function and get type annotation
    main(**vars(parser.parse_args()))


if __name__ == "__main__":
    cli_main()

import argparse
import itertools
import logging
from re import L

from websockets import datastructures

import simulstream
from simulstream.metrics.logger import setup_metrics_logger, METRICS_LOGGER
from simulstream.server.message_processor import MessageProcessor
from simulstream.server.speech_processors import build_speech_processor, SpeechProcessor
from pathlib import Path
from tqdm import tqdm
import requests
import os
import shutil
from typing import final, override
from abc import ABC, abstractmethod

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
LOGGER = logging.getLogger('simulstream.inference')

#Downloads and prepares evaluation datasets una funcion de preparar dataset
SIMULEVAL_AUDIO_CACHE = Path(os.getenv("SIMULEVAL_AUDIO_CACHE", f"{os.getenv('HOME')}/.cache/simuleval/"))


def ensure_symlink(link: Path, target) -> None:
    """
    Ensure `link` is a symlink pointing to `target`.
    """
    link.parent.mkdir(parents=True, exist_ok=True)

    if link.exists() or link.is_symlink():
        if link.is_dir() and not link.is_symlink():
            shutil.rmtree(link)
        else:
            link.unlink(missing_ok=True)

    # Create symlink pointing to `target`
    link.symlink_to(target)


def get_earnings21():
    raise NotImplementedError()

def get_earnings22():
    raise NotImplementedError()

def get_tedlium():
    raise NotImplementedError()

def get_covost2():
    raise NotImplementedError()

def get_europarl_st(**kwargs):
    raise NotImplementedError()

def get_mcif(**kwargs):
    raise NotImplementedError()


def download(url, fname):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
            desc=fname,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
    shutil.unpack_archive(fname, Path(fname).with_suffix(''))
    os.remove(fname)

class Dataset(ABC):
    dataset: str  = ""
    def __init__(self, cache_path: Path = SIMULEVAL_AUDIO_CACHE, force_download: bool = False, force_preprocess: bool = False) -> None:
        self.cache_path: Path = cache_path
        self.path : Path= (cache_path/self.dataset)
        self.force_download= force_download
        self.force_preprocess= force_preprocess
        self.path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def get(self) -> None:
        pass

    @abstractmethod
    def prepare(self) -> None:
        pass
        
    def __call__(self, **kwargs) -> None:
        downloaded = Path(self.path/"downloaded.ok")
        prepared = Path(self.path/"prepared.ok")

        if not downloaded.exists() or self.force_download:
            LOGGER.info(f"'{self.dataset}': Downloading...")
            downloaded.unlink(missing_ok=True)
            self.get()
            Path(self.cache_path/self.dataset/"downloaded.ok").touch()
        else:
            LOGGER.info(f"'{self.dataset}' has already been downloaded.")

        if not prepared.exists() or self.force_preprocess:
            LOGGER.info(f"'{self.dataset}': Preprocessing...")
            prepared.unlink(missing_ok=True)
            self.prepare()
            Path(self.cache_path/self.dataset/"prepared.ok").touch()
        else:
            LOGGER.info(f"'{self.dataset}' has already been preprocessed.")


import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
import yaml
import pandas as pd
import re

@final
class acl_6060_dataset(Dataset):
    dataset = "acl_6060"
    splits  = ["dev", "eval"]
    src_langs =["en"]
    tgt_langs = ["ar", "de","fa", "fr","ja","nl","pt","ru","tr","zh"]
    url = "https://aclanthology.org/attachments/2023.iwslt-1.2.dataset.zip"
    lang_pairs = [
        (src, tgt)
        for src, tgt in itertools.product(src_langs, tgt_langs) 
        if src != tgt
    ]

    @override
    def get(self) -> None:
        download(self.url, str(self.cache_path/self.dataset/"2023.iwslt-1.2.zip"), )

    #TODO Change so that are links are relative paths to SIMULEVAL_AUDIO_CACHE
    @override
    def prepare(self) -> None:
        for split in self.splits:
            wav_path = self.path/f"{split}_wavs_list.txt"
            wav_files = self.path/f"2023.iwslt-1.2/2/acl_6060/{split}/FILE_ORDER"
            ensure_symlink(wav_path, wav_files) if not wav_path.exists() else None
            with open(wav_files, 'r') as f:
                for wav in f.readlines():
                    print(wav)
                    tmp = (self.path/f"2023.iwslt-1.2/2/acl_6060/{split}/full_wavs"/wav.rstrip())
                    ensure_symlink(self.path/wav.rstrip(),  tmp.with_suffix(tmp.suffix+".wav"))
        LOGGER.info(f"Created wav lists for splits dev and eval at {self.path}/dev_wavs_list.txt and {self.path}/eval_wavs_list.txt")

        #TODO Finish linking text references.
        with open("/home/jorirsan/trabajo/git/simulstream/ACL.ACLdev2023.en-xx.gold_segments.yaml", 'r') as file:
            df = pd.DataFrame(yaml.safe_load(file))
        for split in self.splits:
            for src, tgt in self.lang_pairs:
                xml_file = self.path/f"2023.iwslt-1.2/2/acl_6060/{split}/text/xml/ACL.6060.{split}.{src}-xx.{tgt}.xml"
                #print(self.load_segments_from_xml(xml_file)[0])


    def load_segments_from_xml(self, xml_file):
        with open(xml_file, 'r', encoding='utf-8') as f:
            xml_content = f.read()
    
        # Remove all <abstract>...</abstract> blocks bacause test sets abstract have weird characthers that break XML parsing
        xml_content = re.sub(r'<abstract>.*?</abstract>', '', xml_content, flags=re.DOTALL)
        root = ET.fromstring(xml_content)

        docs = defaultdict(dict)
        for d in root.findall(".//doc"):
            docid = d.attrib["docid"]
            for s in d.findall(".//seg"):
                segid = s.attrib["id"]
                text = s.text
                docs[docid][segid] = text

        return docs



def get_dataset(dataset_name, cache_path, force_download, force_preprocess) -> Dataset:
    cls_name = f"{dataset_name}_dataset"
    cls = globals().get(cls_name)
    if not isinstance(cls, type) or not issubclass(cls, Dataset):
        raise ValueError(
            f"No dataset preparation class found for '{dataset_name}'. Class '{cls_name}' must be a defined subclass of Dataset."
        )
    ds: Dataset = cls(cache_path, force_download, force_preprocess)
    return ds


def main(dataset: str, cache_path: Path, force_download: bool, force_preprocess: bool, **kwargs):
    LOGGER.info(f"Will save dataset at {cache_path}")
    ds = get_dataset(dataset, cache_path, force_download, force_preprocess)
    ds()
    LOGGER.info(f"Finished downloading and preparing dataset '{dataset}' at {ds.path}")


def cli_main():
    """
    Simulstream command-line interface (CLI) to download and prepare various evaluation datasets.
    This will be created
    Various datasets recipies are avaialble to download and prepare the folders and expected YAML formats for long-form ASR/ST and short form ST are given.
    By default, prepares all available splits and language directions for a dataset
    Resulting folders are in the following format ./datasets/$DATASET/
    and will create under the directory:
        -The corresponding wavs
        -The txt list of wavs to be feed to simulstream_inference --wav-list-file with format $DATASETNAME_$SPLIT_$SRC_$TGT.txt
    If no target reference on the dataset, $TGT value will be set to the string noref

    """
    LOGGER.info(f"Simulstream version: {simulstream.__version__}")
    parser = argparse.ArgumentParser("simulstream_inference")
    parser.add_argument("--dataset", type=str, required=True,
        choices = [
            #Long form (>10 mins on average)

            #Long form ASR (no reference)
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
    parser.add_argument("--cache-path", type=Path, default=SIMULEVAL_AUDIO_CACHE, help="Path where to save datasets")
    parser.add_argument("--force-download", "-fd", action="store_true", help="Forces redownload of dataset even if folder is already present in cache")
    parser.add_argument("--force-preprocess", "-fp", action="store_true", help="Forces pre-process of dataset even if preocessdfolder is already present in cache")
    #Pass vars instead of args so that we can define in function and get type annotation
    main(**vars(parser.parse_args()))


if __name__ == "__main__":
    cli_main()

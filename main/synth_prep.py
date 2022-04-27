# import datetime as dt

# today = dt.datetime.today()
# LOG_FNAME = f"{today.day:02d}-{today.month:02d}-{today.year}.log"
# LOG_FORMAT = '%(asctime)s: %(name)s - %(levelname)s - %(message)s'
# DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

# formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

# numba_logger = logging.getLogger('numba')
# numba_logger.setLevel(logging.WARNING)

# logger = logging.getLogger("Preprocess Logger")

# file_handler = logging.FileHandler(f"./log/{LOG_FNAME}")
# file_handler.setLevel(logging.INFO)
# file_handler.setFormatter(formatter)

# logger.addHandler(file_handler)

# logger.info(f"Project Name: RTVC - Model Name: Synthesizer - Stage Name: Preprocessing")

#---------------------------------------------------------
import torch
from enc import embed_frames_batch

#----------------------------------------------------
from glob import glob

import numpy as np

import re
from unidecode import unidecode

import inflect
inflect = inflect.engine()

data = {}
with open("../data/LibriSpeech_text.txt", "r") as fh: 
    for line in fh.readlines(): 
        data[line.split(": ")[0]] = line.split(": ")[1]

with open("../data/LibriTTS_text.txt", "r") as fh: 
    for line in fh.readlines(): 
        data[line.split(": ")[0]] = line.split(": ")[1]


symbols = list("_~ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'\"(),-.:;? ")

symbol_to_id = {s: i for i, s in enumerate(symbols)}

abbreviations = [(re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1]) for x in [
    ("mrs", "misess"),
    ("mr", "mister"),
    ("dr", "doctor"),
    ("st", "saint"),
    ("co", "company"),
    ("jr", "junior"),
    ("maj", "major"),
    ("gen", "general"),
    ("drs", "doctors"),
    ("rev", "reverend"),
    ("lt", "lieutenant"),
    ("hon", "honorable"),
    ("sgt", "sergeant"),
    ("capt", "captain"),
    ("esq", "esquire"),
    ("ltd", "limited"),
    ("col", "colonel"),
    ("ft", "fort"),
]]

def expand_dollars(m):
    parts = m.group(1).split(".")

    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0 

    if dollars and cents:
        return "%s %s, %s %s" % (dollars, "dollars", cents, "cents")
    elif dollars:
        return "%s %s" % (dollars, "dollars")
    elif cents:
        return "%s %s" % (cents, "cents")
    else:
        return "zero dollars"

def expand_number(m):
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return "two thousand"
        elif num > 2000 and num < 2010:
            return "two thousand " + inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return inflect.number_to_words(num // 100) + " hundred"
        else:
            return inflect.number_to_words(num, andword="", zero="oh", group=2).replace(", ", " ")
    else:
        return inflect.number_to_words(num, andword="")


def normalize_numbers(text):
    text = re.sub(re.compile(r"([0-9][0-9\,]+[0-9])"), lambda m:m.group(1).replace(",", ""), text)
    text = re.sub(re.compile(r"£([0-9\,]*[0-9]+)"), r"\1 pounds", text)
    text = re.sub(re.compile(r"\$([0-9\.\,]*[0-9]+)"), expand_dollars, text)
    text = re.sub(re.compile(r"([0-9]+\.[0-9]+)"), lambda m:m.group(1).replace(".", " point "), text)
    text = re.sub(re.compile(r"[0-9]+(st|nd|rd|th)"), lambda m:inflect.number_to_words(m.group(0)), text)
    text = re.sub(re.compile(r"[0-9]+"), expand_number, text)
    return text

def clean(text):
    text = unidecode(text)
    text = text.lower()
    text = normalize_numbers(text)
    for regex, replacement in abbreviations:
        text = re.sub(regex, replacement, text)
    text = re.sub(re.compile(r"\s+"), " ", text)
    return text

def text_to_seq(text):   
    text = clean(text)
    seq = [symbol_to_id[s] for s in text if s in symbol_to_id and s not in ("_", "~")]
    seq.append(symbol_to_id["~"])
    return seq 


for path in glob("../data/audio/*/*/*"):
    idx = path.split("/")[-1].split(".")[0]

    text = np.array(text_to_seq(data[idx]))

    arr = np.load(path)

    embeds = embed_frames_batch(arr)

    print(arr.shape, text.shape, embed.shape)

    break

    



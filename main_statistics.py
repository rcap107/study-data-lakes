"""GITTABLES
This script combines the scripts that were used to prepare the statistics used to run observations on the repository.

"""
import argparse
import glob
import json
import os.path as osp
from collections import Counter

from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Digits, Whitespace
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        help="Select the command to execute.", required=True
    )

    parser_encode = subparsers.add_parser(
        "encode", help="Encode tokens in a corpus using given tokenizer."
    )

    parser_encode.add_argument(
        "src_dir",
        action="store",
        required=True,
        help="Source folder of corpus files. The path can be expanded with * flag.",
        default=None,
    )
    parser_encode.add_argument(
        "--tokenizer_path",
        action="store",
        default="data/tokenizer.json",
        help="Set tokenizer path.",
    )

    parser_tally = subparsers.add_parser(
        "tally",
        help="Tally tokens in the given corpus, according to the given tokenizer.",
    )
    parser_tally.add_argument(
        "src_dir",
        action="store",
        required=True,
        help="Source folder of corpus files. The path can be expanded with * flag.",
    )
    
    parser_tally.add_argument("--recursive", "-r", action="store_true", help="Search the provided path recursively.")
    
    parser_tally.add_argument(
        "--tokenizer_path",
        action="store",
        default="data/tokenizer.json",
        help="Read tokenizer from the given path. ",
    )
    parser_tally.add_argument(
        "--counter_path",
        action="store",
        default="data/global_counter.json",
        help="Set global counter path.",
    )


def encode_corpus(args):
    # corpus_files = glob.glob("data/zenodo/txt_tables/*/*.txt")
    corpus_files = glob.glob(args.src_dir, recursive=args.recursive)

    normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits()])
    # Creating the tokenizer with the assigned normalizer and pre-tokenizer
    tokenizer = Tokenizer(BPE())
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre_tokenizer

    trainer = BpeTrainer()  # No special tokens for now

    tokenizer.train(corpus_files, trainer)

    tokenizer.save(args.tokenizer_path)


def tally_tokens(args):
    def tally_tokens(target_tab, tokenizer):
        with open(target_tab, "r") as fp:
            lines = fp.readlines()
            batch_ = tokenizer.encode_batch(lines)
            tokens = [token for enc in batch_ for token in enc.tokens]
        return tokens

    tokenizer_path = args.tokenizer_path
    assert osp.exists(tokenizer_path)

    tokenizer = Tokenizer.from_file(tokenizer_path)

    table_glob = glob.glob(args.src_dir)
    assert len(table_glob) > 0

    global_counter = Counter(
        dict(zip(tokenizer.get_vocab(), [0 for _ in range(tokenizer.get_vocab_size())]))
    )

    print("Starting tally operation.")

    for idx, target_tab in tqdm(enumerate(table_glob), total=len(table_glob)):
        try:
            global_counter.update(tally_tokens(target_tab, tokenizer))
        except KeyboardInterrupt:
            print("Interrupting...")
            break

    json.dump(global_counter, open(args.counter_path, "w"))


if __name__ == "__main__":
    args = parse_args()
    args.func(args)

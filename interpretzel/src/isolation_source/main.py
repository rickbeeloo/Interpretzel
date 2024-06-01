from .class_gen import DescGen
from .class_pred import LLMPred
import argparse
import asyncio


def pretzel_iso_gen():
    parser = argparse.ArgumentParser(prog='Interpretzel iso generator V1')
    parser.add_argument('-i', help='File with class examples', type = str)
    parser.add_argument('-o', help='Where to write class output json', type = str)
    parser.add_argument('-m', nargs="?", default=None, help='Model to use', type = str, const="meta-llama/Meta-Llama-3-8B-Instruct")
    args = parser.parse_args()
    gen = DescGen(args.m)
    gen.process(args.i, args.o)

def pretzel_iso_pred():
    parser = argparse.ArgumentParser(prog='Interpretzel iso predictor V1')
    parser.add_argument('-i', help='File with queries', type = str)
    parser.add_argument('-o', help='Where to write class predictions', type = str)
    parser.add_argument('-c', help='Class json', type = str)
    parser.add_argument('-m', nargs="?", help='Model to use', type = str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument('-v', nargs="?", help='verbose output', type = bool, default=False)
    parser.add_argument('-p', nargs="?", help='percentage cutoff', type = float, default=99.9)
    args = parser.parse_args()
    gen = LLMPred(model_name = args.m, verbose = args.v)
    asyncio.run(gen.predict(args.i, args.c, args.o, cut_off=args.p))
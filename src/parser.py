from argparse import Namespace
from argparse import ArgumentParser


def argparser() -> Namespace:
    """Parse command-line arguments for the function calling system.

    Returns:
        Namespace object with parsed arguments.
    """
    parser = ArgumentParser()
    parser.add_argument(
        '--input', type=str,
        default="data/input/function_calling_tests.json")
    parser.add_argument(
        '--functions_definition', type=str,
        default="data/input/functions_definition.json")
    parser.add_argument(
        '--model', type=str,
        default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        '--output', type=str,
        default="data/output/function_calling_results.json")

    return parser.parse_args()

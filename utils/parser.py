import argparse


def parse_args():
    parser = argparse.ArgumentParser(prog='LinkSeg', description='Structural segmentation')
    parser.add_argument('audio_files', metavar='FILE', type=str, nargs='+',
                        help='audio files to process')
    parser.add_argument('-m', '--model_name', type=str, default="Harmonix_full", choices=["Harmonix_full"],
                        help='choice of the model')
    parser.add_argument('-o', '--output', metavar='DIR', type=str, default=None,
                        help='directory to save the output predictions')
    parser.add_argument('-e', '--export_format', type=str, default=["jams"], choices=["jams"], nargs='+')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='the index of the GPU to use, -1 for CPU')
    return parser.parse_args()
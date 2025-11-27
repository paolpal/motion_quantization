import argparse
import json
from pathlib import Path
from utils.plot import animate_skeleton_from_dataset
from quantization.codebook import PoseCodebook


def animate_command(args):
    """Comando per animare uno scheletro dal dataset."""
    # Carica il codebook

    data_path = Path(args.data_path)
    dataset_folder = data_path / "datasets"
    split_name = 'train' if args.train else 'dev' if args.dev else 'test'
    split_folder = dataset_folder / split_name
    codebook_path = dataset_folder / "codebook.pkl"
    codebook = PoseCodebook.load(codebook_path)

    dataset_path = split_folder / f"{split_name}_dataset.jsonl"

    # Leggi il sample dal dataset
    with open(dataset_path, 'r') as f:
        for i, line in enumerate(f):
            if i == args.index:
                sample = json.loads(line)
                break
        else:
            raise ValueError(f"Index {args.index} not found in dataset")
    
    # Crea l'animazione
    animate_skeleton_from_dataset(
        sample=sample,
        codebook=codebook,
        skeleton_type=args.skeleton_type,
        fps=args.fps,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Utility per visualizzazione e analisi del dataset pose"
    )
    split = parser.add_mutually_exclusive_group(required=True)
    split.add_argument('--train', action='store_true', help='Process train dataset')
    split.add_argument('--dev', action='store_true', help='Process dev dataset')
    split.add_argument('--test', action='store_true', help='Process test dataset')

    subparsers = parser.add_subparsers(dest='command', help='Comandi disponibili')
    
    # Comando animate
    animate_parser = subparsers.add_parser(
        'animate',
        help='Anima lo scheletro da un sample del dataset'
    )
    animate_parser.add_argument(
        '--data_path', 
        required=True, 
        help='Path to the data folder'
    )

    animate_parser.add_argument(
        '--index',
        type=int,
        default=0,
        help='Indice del sample nel dataset (default: 0)'
    )
    animate_parser.add_argument(
        '--skeleton-type',
        type=str,
        choices=['full', 'torso', 'upper_body'],
        default=None,
        help='Tipo di scheletro (default: auto-rilevato)'
    )
    animate_parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Frame per secondo (default: 30)'
    )
    animate_parser.set_defaults(func=animate_command)
    
    # Parse argomenti
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
    else:
        args.func(args)


if __name__ == "__main__":
    main()

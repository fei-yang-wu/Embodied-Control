"""Build a stance/bridge reference candidate for simulation rehearsal."""
import argparse

from embodied_control.lowlevel.bundle import PolicyBundle
from embodied_control.lowlevel.reference_compose import compose_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', required=True)
    parser.add_argument('--motion', required=True)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--hold-seconds', type=float, default=2.0)
    parser.add_argument('--bridge-seconds', type=float, default=3.0)
    args = parser.parse_args()
    name = compose_reference(args.reference_root, args.motion, PolicyBundle.load(args.bundle),
                             args.model, args.output, hold_seconds=args.hold_seconds,
                             bridge_seconds=args.bridge_seconds)
    print(f'motion: {name}')
    print('Candidate only: rehearse this exact reference with each selected checkpoint.')


if __name__ == '__main__':
    main()

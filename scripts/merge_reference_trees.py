"""Merge compatible reference arrays into one BONES motion collection."""
import argparse

from embodied_control.lowlevel.reference_catalog import merge_reference_trees


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', nargs='+')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    manifest = merge_reference_trees(args.sources, args.output)
    for name, info in manifest['motions'].items():
        tier = 'deployable' if info['deployable'] else 'training'
        print(f"{tier:12} {name}: {'; '.join(info['reasons'])}")
    motions = manifest['motions']
    print(f"{len(motions)} motions; {sum(m['deployable'] for m in motions.values())} pass endpoint screening")


if __name__ == '__main__':
    main()

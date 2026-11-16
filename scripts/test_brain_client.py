"""Protocol-only health probe; no robot or CUDA on the client."""

import argparse
from rlt_online.broker import BrainBroker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", default="ws://127.0.0.1:6016")
    args = parser.parse_args()
    broker = BrainBroker(args.uri, timeout=10)
    try:
        print(broker.submit({"command": "health"}).result(timeout=12))
    finally:
        broker.close()


if __name__ == "__main__":
    main()

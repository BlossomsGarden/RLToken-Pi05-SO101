"""Hardware-free request/shape smoke test against a running OpenPI SFT server."""

import argparse

import numpy as np

from sft.deploy_openpi import encode_jpeg, validate_actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--prompt", default="Pick up the red block and place it on top of the green block.")
    args = parser.parse_args()
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    client = WebsocketClientPolicy(host=args.host, port=args.port)
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    state = np.zeros(6, dtype=np.float32)
    result = client.infer({"transport/front_image_jpeg": encode_jpeg(frame, 90),
                           "transport/side_image_jpeg": encode_jpeg(frame, 90),
                           "observation/state": state, "prompt": args.prompt})
    actions = validate_actions(result["actions"], state)
    if actions.shape != (50, 6):
        raise RuntimeError(f"Expected (50,6), got {actions.shape}")
    print("SMOKE TEST PASSED: finite (50,6) actions; synthetic images, no robot motion")


if __name__ == "__main__":
    main()

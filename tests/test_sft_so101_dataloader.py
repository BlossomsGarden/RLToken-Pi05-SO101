def main():
    from openpi.training import config, data_loader
    from sft.so101_config import NAME, register

    register()
    cfg = config.get_config(NAME)

    print("Creating DataLoader...")

    loader = data_loader.create_data_loader(
        cfg,
        num_batches=1,
        skip_norm_stats=True,
        shuffle=False,
    )

    print("Loading first batch...")

    obs, actions = next(iter(loader))

    print("\n===== DATALOADER OK =====")

    print("\nState shape:")
    print(obs.state.shape)

    print("\nImage shapes:")
    for k, v in obs.images.items():
        print(k, v.shape)

    print("\nImage masks:")
    for k, v in obs.image_masks.items():
        print(k, v.shape)

    print("\nAction shape:")
    print(actions.shape)

    print("\nFirst sample state:")
    print(obs.state[0])

    print("\nFirst sample first action:")
    print(actions[0, 0])

    # Prompt-related fields can differ slightly between OpenPI versions.
    if hasattr(obs, "tokenized_prompt"):
        print("\nTokenized prompt shape:")
        print(obs.tokenized_prompt.shape)

    if hasattr(obs, "tokenized_prompt_mask"):
        print("\nTokenized prompt mask shape:")
        print(obs.tokenized_prompt_mask.shape)


if __name__ == "__main__":
    main()

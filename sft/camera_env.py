"""Resolve the two supported names for each Windows camera assignment."""

from collections.abc import Mapping


def camera_setting(values: Mapping[str, str | None], canonical: str, alias: str) -> tuple[str, str | None]:
    primary = values.get(canonical)
    alternative = values.get(alias)
    if primary and alternative and primary != alternative:
        raise ValueError(f"Conflicting camera assignments: {canonical}={primary} and {alias}={alternative}")
    return (canonical, primary) if primary is not None else (alias, alternative)

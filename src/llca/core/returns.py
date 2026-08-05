"""Shared type contract for periodic return conventions."""

from __future__ import annotations

from typing import Literal

type ReturnType = Literal["simple", "log"]

RETURN_TYPES: tuple[ReturnType, ...] = ("simple", "log")

"""Small fake transformer metadata for block registry tests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FakeBlock:
    tensor_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FakeLayer:
    mha: FakeBlock
    ffn: FakeBlock


@dataclass(frozen=True, slots=True)
class FakeTransformer:
    layers: tuple[FakeLayer, ...]


class UnsupportedTransformer:
    pass


def make_fake_transformer(num_layers: int = 2) -> FakeTransformer:
    return FakeTransformer(
        layers=tuple(
            FakeLayer(
                mha=FakeBlock(
                    tensor_names=(
                        f"layers.{layer_index}.mha.q_proj.weight",
                        f"layers.{layer_index}.mha.o_proj.weight",
                    )
                ),
                ffn=FakeBlock(
                    tensor_names=(
                        f"layers.{layer_index}.ffn.gate_proj.weight",
                        f"layers.{layer_index}.ffn.down_proj.weight",
                    )
                ),
            )
            for layer_index in range(num_layers)
        )
    )

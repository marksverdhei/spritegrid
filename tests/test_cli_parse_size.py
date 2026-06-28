"""Tests for spritegrid.cli.parse_size."""

from __future__ import annotations

import argparse

import pytest

from spritegrid.cli import parse_size


class TestParseSize:
    def test_single_integer_returns_square(self):
        assert parse_size("32") == (32, 32)

    def test_single_integer_64(self):
        assert parse_size("64") == (64, 64)

    def test_wxh_format(self):
        assert parse_size("32x48") == (32, 48)

    def test_wxh_uppercase_x(self):
        assert parse_size("16X24") == (16, 24)

    def test_wxh_same_dims_is_square(self):
        assert parse_size("8x8") == (8, 8)

    def test_non_numeric_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_size("abc")

    def test_wxh_non_numeric_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_size("axb")

    def test_too_many_parts_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_size("32x48x16")

    def test_single_zero(self):
        assert parse_size("0") == (0, 0)

    def test_wxh_asymmetric_values(self):
        w, h = parse_size("100x200")
        assert w == 100
        assert h == 200

    def test_single_float_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_size("3.14")

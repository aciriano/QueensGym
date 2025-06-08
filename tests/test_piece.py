from pathlib import Path

import pytest

from queensgym.board import Square
from queensgym.piece import Queen


class TestChessPieceRegistry:
    """
    Tests suit for queensgym.piece.ChessPieceRegistry.
    """


class TestQueen:
    """
    Tests suit for queensgym.piece.Queen.
    """

    def test_get_id(self) -> None:
        assert Queen.get_id() == 1

    def test_get_value(self) -> None:
        assert Queen.get_value() == 9.0

    def test_get_symbol(self) -> None:
        assert Queen.get_symbol() == "♕"

    def test_get_icon(self) -> None:
        assert isinstance(Queen.get_icon(), Path)

    @pytest.mark.parametrize(
        argnames="sq1, sq2, expected",
        argvalues=[
            (Square(1, 1), Square(3, 2), False),
            (Square(1, 1), Square(2, 8), False),
            (Square(1, 1), Square(8, 2), False),
            (Square(4, 5), Square(6, 7), True),
            (Square(4, 5), Square(1, 8), True),
            (Square(4, 5), Square(2, 3), True),
            (Square(4, 5), Square(7, 2), True),
            (Square(1, 1), Square(1, 1), False),
        ],
    )
    def test_attack(self, sq1: Square, sq2: Square, expected: bool) -> None:
        assert Queen.attack(sq1, sq2) is expected

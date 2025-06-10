from pathlib import Path
from typing import Any
from typing import Type

import pytest

from queensgym.board import Square
from queensgym.exceptions import InvalidPieceError
from queensgym.exceptions import PieceDefinitionError
from queensgym.piece import Bishop
from queensgym.piece import ChessPiece
from queensgym.piece import ChessPieceRegistry
from queensgym.piece import King
from queensgym.piece import Knight
from queensgym.piece import Queen
from queensgym.piece import Rook


def create_piece(
    name: str, _id: Any, _value: Any, _symbol: Any, _icon: Any = None
) -> Type[ChessPiece]:
    """Creates a new ChessPiece based on a template."""
    return type(
        name,
        (ChessPiece,),
        {
            "get_id": classmethod(lambda cls: _id),
            "get_value": classmethod(lambda cls: _value),
            "get_symbol": classmethod(lambda cls: _symbol),
            "get_icon": classmethod(lambda cls: _icon),
            "_attack": classmethod(lambda cls, _from, _to: True),
        },
    )


class TestChessPieceRegistry:
    """
    Tests suit for queensgym.piece.ChessPieceRegistry.
    """

    def test_add_duplicated_id(self) -> None:
        with pytest.raises(InvalidPieceError):
            piece = create_piece("MyPiece", 1, 1.0, "A")
            ChessPieceRegistry.add(piece)

    def test_add_invalid_class(self) -> None:
        with pytest.raises(InvalidPieceError):
            ChessPieceRegistry.add(int)

    @pytest.mark.parametrize(argnames="piece", argvalues=[Queen, Rook, Bishop, Knight, King])
    def test_exists_main_pieces(self, piece: Any) -> None:
        assert ChessPieceRegistry.exists(piece)

    @pytest.mark.parametrize(argnames="piece", argvalues=[1, 2, 3, 4, 5])
    def test_exists_main_pieces_by_id(self, piece: Any) -> None:
        assert ChessPieceRegistry.exists(piece)

    @pytest.mark.parametrize(argnames="piece", argvalues=["", {}, list(), list])
    def test_non_exists(self, piece: Any) -> None:
        assert not ChessPieceRegistry.exists(piece)

    @pytest.mark.parametrize(argnames="piece", argvalues=[Queen, Rook, Bishop, Knight, King])
    def test_get(self, piece: Any) -> None:
        assert ChessPieceRegistry.get(piece.get_id()) == piece

    @pytest.mark.parametrize(argnames="piece_id", argvalues=[{}, ""])
    def test_get_invalid_piece(self, piece_id: Any) -> None:
        with pytest.raises(InvalidPieceError):
            ChessPieceRegistry.get(piece_id)


class TestChessPiece:
    """
    Tests suit for queensgym.piece.ChessPiece.
    """

    @pytest.mark.parametrize(argnames="_id", argvalues=["", 0])
    def test_invalid_id(self, _id: Any) -> None:
        with pytest.raises(PieceDefinitionError):
            create_piece(name=f"piece_{_id}", _id=_id, _value=1.0, _symbol="a")

    @pytest.mark.parametrize(argnames="_value", argvalues=["", -1.0])
    def test_invalid_value(self, _value: Any) -> None:
        with pytest.raises(PieceDefinitionError):
            create_piece(name=f"piece_{_value}", _id=1000, _value=_value, _symbol="a")

    @pytest.mark.parametrize(argnames="_symbol", argvalues=[None, -1.0, "AB"])
    def test_invalid_symbol(self, _symbol: Any) -> None:
        with pytest.raises(PieceDefinitionError):
            create_piece(name=f"piece_{_symbol}", _id=1000, _value=1.0, _symbol=_symbol)

    @pytest.mark.parametrize(argnames="_icon", argvalues=[[], -1.0, "AB"])
    def test_invalid_icon(self, _icon: Any) -> None:
        with pytest.raises(PieceDefinitionError):
            create_piece(name=f"piece_{_icon}", _id=1000, _value=1.0, _symbol="A", _icon=_icon)

    def test_attack_type_error(self) -> None:
        my_class = create_piece("MyClass", 100, 1.0, "A")
        with pytest.raises(TypeError):
            my_class.attack(1, Square(1, 1))
        with pytest.raises(TypeError):
            my_class.attack(Square(1, 1), 1)


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
            (Square(4, 5), Square(4, 1), True),
            (Square(4, 5), Square(4, 7), True),
            (Square(4, 5), Square(1, 5), True),
            (Square(4, 5), Square(7, 5), True),
            (Square(1, 1), Square(1, 1), False),
        ],
    )
    def test_attack(self, sq1: Square, sq2: Square, expected: bool) -> None:
        assert Queen.attack(sq1, sq2) is expected


class TestRook:
    """
    Tests suit for queensgym.piece.Rook.
    """

    def test_get_id(self) -> None:
        assert Rook.get_id() == 2

    def test_get_value(self) -> None:
        assert Rook.get_value() == 5.0

    def test_get_symbol(self) -> None:
        assert Rook.get_symbol() == "♖"

    def test_get_icon(self) -> None:
        assert isinstance(Rook.get_icon(), Path)

    @pytest.mark.parametrize(
        argnames="sq1, sq2, expected",
        argvalues=[
            (Square(1, 1), Square(3, 2), False),
            (Square(1, 1), Square(2, 8), False),
            (Square(1, 1), Square(8, 2), False),
            (Square(4, 5), Square(4, 1), True),
            (Square(4, 5), Square(4, 7), True),
            (Square(4, 5), Square(1, 5), True),
            (Square(4, 5), Square(7, 5), True),
            (Square(1, 1), Square(1, 1), False),
        ],
    )
    def test_attack(self, sq1: Square, sq2: Square, expected: bool) -> None:
        assert Rook.attack(sq1, sq2) is expected


class TestBishop:
    """
    Tests suit for queensgym.piece.Bishop.
    """

    def test_get_id(self) -> None:
        assert Bishop.get_id() == 3

    def test_get_value(self) -> None:
        assert Bishop.get_value() == 3.0

    def test_get_symbol(self) -> None:
        assert Bishop.get_symbol() == "♗"

    def test_get_icon(self) -> None:
        assert isinstance(Bishop.get_icon(), Path)

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
        assert Bishop.attack(sq1, sq2) is expected


class TestKnight:
    """
    Tests suit for queensgym.piece.Knight.
    """

    def test_get_id(self) -> None:
        assert Knight.get_id() == 4

    def test_get_value(self) -> None:
        assert Knight.get_value() == 3.0

    def test_get_symbol(self) -> None:
        assert Knight.get_symbol() == "♘"

    def test_get_icon(self) -> None:
        assert isinstance(Knight.get_icon(), Path)

    @pytest.mark.parametrize(
        argnames="sq1, sq2, expected",
        argvalues=[
            (Square(3, 5), Square(7, 1), False),
            (Square(3, 5), Square(3, 8), False),
            (Square(3, 5), Square(4, 3), True),
            (Square(3, 5), Square(5, 4), True),
            (Square(3, 5), Square(8, 8), False),
            (Square(3, 5), Square(5, 6), True),
            (Square(3, 5), Square(8, 1), False),
            (Square(3, 5), Square(7, 7), False),
        ],
    )
    def test_attack(self, sq1: Square, sq2: Square, expected: bool) -> None:
        assert Knight.attack(sq1, sq2) is expected


class TestKing:
    """
    Tests suit for queensgym.piece.King.
    """

    def test_get_id(self) -> None:
        assert King.get_id() == 5

    def test_get_value(self) -> None:
        assert King.get_value() == 4.0

    def test_get_symbol(self) -> None:
        assert King.get_symbol() == "♔"

    def test_get_icon(self) -> None:
        assert isinstance(King.get_icon(), Path)

    @pytest.mark.parametrize(
        argnames="sq1, sq2, expected",
        argvalues=[
            (Square(3, 5), Square(3, 6), True),
            (Square(3, 5), Square(3, 4), True),
            (Square(3, 5), Square(4, 5), True),
            (Square(3, 5), Square(2, 5), True),
            (Square(3, 5), Square(4, 6), True),
            (Square(3, 5), Square(4, 4), True),
            (Square(3, 5), Square(2, 6), True),
            (Square(3, 5), Square(2, 4), True),
            (Square(3, 5), Square(7, 7), False),
            (Square(3, 5), Square(1, 1), False),
            (Square(3, 5), Square(6, 1), False),
            (Square(3, 5), Square(10, 1), False),
        ],
    )
    def test_attack(self, sq1: Square, sq2: Square, expected: bool) -> None:
        assert King.attack(sq1, sq2) is expected

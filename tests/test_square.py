from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from queensgym.square import Color
from queensgym.square import Square


class TestSquare:
    """
    Tests suit for queensgym.square.Square.
    """

    def test_init(self) -> None:
        square = Square(1, 2)
        assert square.file == 1
        assert square.rank == 2
        assert square.column == 1
        assert square.row == 2

    @pytest.mark.parametrize(argnames="file, rank", argvalues=[(1, None), (None, 1)])
    def test_type_error(self, file: Any, rank: Any) -> None:
        with pytest.raises(TypeError):
            Square(file=file, rank=rank)

    @pytest.mark.parametrize(argnames="file, rank", argvalues=[(0, 1), (1, 0), (1, -1)])
    def test_value_error(self, file: Any, rank: Any) -> None:
        with pytest.raises(ValueError):
            Square(file=file, rank=rank)

    def test_is_frozen(self) -> None:
        with pytest.raises(FrozenInstanceError):
            sq = Square(1, 1)
            sq.file = 2

    def test_is_hashable(self) -> None:
        square = Square(1, 2)
        assert len({square}) == 1
        assert len({square: 1}) == 1
        assert isinstance(hash(square), int)

    def test_color(self) -> None:
        square = Square(1, 2)
        assert square.color is Color.WHITE
        assert square.symbol == "☐"

        square = Square(1, 1)
        assert square.color is Color.BLACK
        assert square.symbol == "◼︎"

    @pytest.mark.parametrize(
        argnames="sq1, sq2, expected",
        argvalues=[
            (Square(1, 2), Square(1, 3), True),
            (Square(1, 2), Square(2, 2), False),
            (Square(3, 4), Square(3, 5), True),
            (Square(5, 6), Square(6, 6), False),
        ],
    )
    def test_in_same_file(self, sq1: Square, sq2: Square, expected: bool) -> None:
        """Test the in_same_file method."""
        assert sq1.in_same_file(sq2) == expected

    def test_in_same_file_type_error(self) -> None:
        with pytest.raises(NotImplementedError):
            Square(1, 1).in_same_file("Not a Square.")

    @pytest.mark.parametrize(
        argnames="sq1, sq2, expected",
        argvalues=[
            (Square(1, 2), Square(1, 3), False),
            (Square(1, 2), Square(2, 2), True),
            (Square(3, 4), Square(3, 5), False),
            (Square(5, 6), Square(6, 6), True),
        ],
    )
    def test_in_same_rank(self, sq1: Square, sq2: Square, expected: bool) -> None:
        """Test the in_same_rank method."""
        assert sq1.in_same_rank(sq2) == expected

    def test_in_same_rank_type_error(self) -> None:
        with pytest.raises(NotImplementedError):
            Square(1, 1).in_same_rank("Not a Square.")

    @pytest.mark.parametrize(
        argnames="sq1, sq2, expected",
        argvalues=[
            (Square(4, 8), Square(7, 5), True),
            (Square(4, 8), Square(1, 5), True),
            (Square(4, 8), Square(2, 7), False),
            (Square(5, 6), Square(6, 6), False),
            (Square(5, 5), Square(7, 4), False),
            (Square(5, 5), Square(2, 2), True),
            (Square(5, 5), Square(3, 7), True),
            (Square(5, 5), Square(8, 8), True),
        ],
    )
    def test_in_same_diagonal(self, sq1: Square, sq2: Square, expected: bool) -> None:
        """Test the in_same_diagonal method."""
        assert sq1.in_same_diagonal(sq2) == expected

    def test_in_same_diagonal_type_error(self) -> None:
        with pytest.raises(NotImplementedError):
            Square(1, 1).in_same_diagonal("Not a Square.")

    @pytest.mark.parametrize(
        argnames="sq1, sq2, expected",
        argvalues=[
            (Square(1, 2), Square(1, 3), 1),
            (Square(1, 2), Square(1, 2), 0),
            (Square(1, 1), Square(8, 8), 7),
            (Square(5, 6), Square(6, 6), 1),
        ],
    )
    def test_distance(self, sq1: Square, sq2: Square, expected: int) -> None:
        """Test the distance method."""
        assert sq1.distance(sq2) == expected

    def test_distance_type_error(self) -> None:
        with pytest.raises(NotImplementedError):
            Square(1, 1).distance("Not a Square.")

    @pytest.mark.parametrize(
        argnames="offset, expected",
        argvalues=[
            ((1, 1), [Square(*i) for i in [(5, 6)]]),
            ((-1, -1), [Square(*i) for i in [(3, 4), (2, 3), (1, 2)]]),
            ((1, -1), [Square(*i) for i in [(5, 4), (6, 3)]]),
            ((-1, 1), [Square(*i) for i in [(3, 6)]]),
            ((1, 0), [Square(*i) for i in [(5, 5), (6, 5)]]),
            ((0, 1), [Square(*i) for i in [(4, 6)]]),
        ],
    )
    def test_related(self, offset: tuple[int, int], expected: list[Square]) -> None:
        sq = Square(4, 5)
        assert sq.get_related(limit=6, offset=offset) == expected

    def test_related_errors(self) -> None:
        sq = Square(4, 5)

        with pytest.raises(ValueError):
            sq.get_related(limit=8.0, offset=(1, 1))

        with pytest.raises(ValueError):
            sq.get_related(limit=0, offset=(1, 1))

        with pytest.raises(ValueError):
            sq.get_related(limit=3, offset=(1, 1))

        with pytest.raises(TypeError):
            sq.get_related(limit=8, offset=[1, 1])

        with pytest.raises(TypeError):
            sq.get_related(limit=8, offset=(1, 1, 1))

        with pytest.raises(TypeError):
            sq.get_related(limit=8, offset=(1, 1.0))

        with pytest.raises(ValueError):
            sq.get_related(limit=8, offset=(0, 0))

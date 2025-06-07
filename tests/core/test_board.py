import pytest

from queensgym.core.board import Board
from queensgym.core.board import Color
from queensgym.core.board import Square


class TestSquare:
    """
    Tests suit for queensgym.core.board.Square.
    """

    def test_init(self) -> None:
        square = Square(1, 2)
        assert square.file == 1
        assert square.rank == 2
        assert square.column == 1
        assert square.row == 2

    def test_type_error(self) -> None: ...

    def test_value_error(self) -> None: ...

    def test_is_frozen(self) -> None: ...

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

    def test_complex(self) -> None: ...

    @pytest.mark.parametrize(
        argnames="sq_1, sq_2, expected",
        argvalues=[
            (Square(1, 2), Square(1, 3), True),
            (Square(1, 2), Square(2, 2), False),
            (Square(3, 4), Square(3, 5), True),
            (Square(5, 6), Square(6, 6), False),
        ],
    )
    def test_in_same_file(self, sq_1: Square, sq_2: Square, expected: bool) -> None:
        """Test the in_same_file method."""
        assert sq_1.in_same_file(sq_2) == expected

    @pytest.mark.parametrize(
        argnames="sq_1, sq_2, expected",
        argvalues=[
            (Square(1, 2), Square(1, 3), False),
            (Square(1, 2), Square(2, 2), True),
            (Square(3, 4), Square(3, 5), False),
            (Square(5, 6), Square(6, 6), True),
        ],
    )
    def test_in_same_rank(self, sq_1: Square, sq_2: Square, expected: bool) -> None:
        """Test the in_same_rank method."""
        assert sq_1.in_same_rank(sq_2) == expected

    @pytest.mark.parametrize(
        argnames="sq_1, sq_2, expected",
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
    def test_in_same_diagonal(self, sq_1: Square, sq_2: Square, expected: bool) -> None:
        """Test the in_same_diagonal method."""
        assert sq_1.in_same_diagonal(sq_2) == expected

    @pytest.mark.parametrize(
        argnames="sq_1, sq_2, expected",
        argvalues=[
            (Square(1, 2), Square(1, 3), 1),
            (Square(1, 2), Square(1, 2), 0),
            (Square(1, 1), Square(8, 8), 7),
            (Square(5, 6), Square(6, 6), 1),
        ],
    )
    def test_distance(self, sq_1: Square, sq_2: Square, expected: int) -> None:
        """Test the distance method."""
        assert sq_1.distance(sq_2) == expected


class TestBoard:
    """
    Tests suit for queensgym.core.board.Board.
    """

    def test_init(self) -> None:
        board = Board(n=Board.max)
        assert board.n == Board.max
        assert not len(board.squares)

    def test_type_error(self) -> None:
        with pytest.raises(TypeError):
            Board(n=1.0)

    @pytest.mark.parametrize("n", [0, Board.max + 1])
    def test_value_error(self, n: int) -> None:
        with pytest.raises(ValueError):
            Board(n=n)

    def test_square_to_numpy(self):
        square = Square(file=1, rank=2)
        Board(n=3).square_to_numpy(square) == 1, 0

    def test_numpy_to_square(self):
        square = Square(file=1, rank=2)
        Board(n=3).numpy_to_square((1, 0)) == square

    def test_square_to_seq(self):
        square = Square(file=1, rank=2)
        Board(n=3).square_to_seq(square) == 4

    def test_seq_to_square(self):
        square = Square(file=1, rank=2)
        Board(n=3).seq_to_square(4) == square

    def test_get(self) -> None: ...

    def test_put(self) -> None: ...

    def test_remove(self) -> None: ...

    def test_matrix(self) -> None: ...

    def test_heatmap(self) -> None: ...

    def test_dynamics(self) -> None: ...

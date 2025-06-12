import numpy as np
import pytest

from queensgym.board import Board
from queensgym.board import SafeBoard
from queensgym.exceptions import InvalidDimensionError
from queensgym.exceptions import InvalidPieceError
from queensgym.exceptions import OccupiedSquareError
from queensgym.exceptions import UnsafePlacementError
from queensgym.piece import Knight
from queensgym.piece import Queen
from queensgym.piece import Rook
from queensgym.square import Square


class TestBoard:
    """
    Tests suit for queensgym.board.Board.
    """

    def test_init(self) -> None:
        board = Board(n=Board.max_size())
        assert board.n == Board.max_size()
        assert not board.total_placed()

    def test_init_type_error(self) -> None:
        with pytest.raises(InvalidDimensionError):
            Board(n=1.0)

    @pytest.mark.parametrize("n", [0, Board.max_size() + 1])
    def test_init_value_error(self, n: int) -> None:
        with pytest.raises(InvalidDimensionError):
            Board(n=n)

    def test_repr(self) -> None:
        board = Board(n=8)
        assert repr(board) == "Board(n=8, state=[])"

        board.put(Square(3, 4), Queen)
        board.put(Square(5, 1), Rook)
        assert repr(board) == "Board(n=8, state=[('Queen', 3, 4), ('Rook', 5, 1)])"

    def test_str(self) -> None:
        board = Board(n=8)
        assert str(board) == "[]"

        board.put(Square(3, 4), Queen)
        board.put(Square(5, 1), Rook)
        assert str(board) == "[('Queen', 3, 4), ('Rook', 5, 1)]"

    def test_tuples(self) -> None:
        board = Board(n=8)
        assert board.tuples == []

        board.put(Square(3, 4), Queen)
        board.put(Square(5, 1), Rook)
        assert board.tuples == [("Queen", 3, 4), ("Rook", 5, 1)]

    def test_pprint(self) -> None:
        board = Board(n=3)
        board.put(Square(1, 1), Queen)
        assert board.pprint() == "◼︎ ☐ ◼︎ \n☐ ◼︎ ☐ \n♕ ☐ ◼︎ \n"

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

    def test_get(self) -> None:
        board = Board(n=5)
        assert board.get(None) is None
        assert board.get(Square(1, 1)) is None

    def test_put(self) -> None:
        board = Board(n=5)
        assert board.get(Square(1, 1)) is None

        board.put(Square(1, 1), Queen)
        assert board.get(Square(1, 1)) is Queen
        assert board.total_placed() == 1

    def test_put_type_error(self) -> None:
        board = Board(n=5)
        with pytest.raises(TypeError):
            board.put(None, Queen)

    def test_put_invalid_piece_error(self) -> None:
        board = Board(n=5)
        with pytest.raises(InvalidPieceError):
            board.put(Square(1, 1), dict())

    def test_put_occupied_square_error(self) -> None:
        board = Board(n=5)
        board.put(Square(1, 1), Queen)
        with pytest.raises(OccupiedSquareError):
            board.put(Square(1, 1), Queen)

    def test_remove(self) -> None:
        board = Board(n=5)
        board.put(Square(1, 1), Queen)
        assert board.get(Square(1, 1)) is Queen
        assert board.total_placed() == 1

        # Now, drop the piece which is placed on the square.
        board.remove(square=Square(1, 1))
        assert board.get(Square(1, 1)) is None
        assert board.total_placed() == 0

    def test_reset(self) -> None:
        board = Board(n=5)
        board.put(Square(1, 1), Queen)
        board.put(Square(3, 2), Queen)
        assert board.total_placed() == 2

        board.reset()
        assert board.total_placed() == 0

    def test_array(self) -> None:
        # Set up a simple board.
        board = Board(n=4)
        board.put(Square(1, 1), Queen)
        board.put(Square(3, 2), Rook)

        # Check the conditions.
        exp_arr = np.array([1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        assert np.array_equal(board.as_array(), exp_arr)

    def test_matrix(self) -> None:
        # Set up a simple board.
        board = Board(n=4)
        board.put(Square(1, 1), Queen)
        board.put(Square(3, 2), Rook)

        # Check the conditions.
        exp_mat = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 2, 0], [1, 0, 0, 0]])
        assert np.array_equal(board.as_matrix(), exp_mat)

    def test_heatmap(self) -> None:
        # Set up a simple board.
        board = Board(n=4)
        board.put(Square(1, 1), Queen)
        board.put(Square(3, 2), Rook)

        # Check the conditions.
        exp_mat = np.array([[1, 0, 1, 1], [1, 0, 2, 0], [2, 2, 0, 1], [0, 1, 2, 1]])
        assert np.array_equal(board.as_heat_map(), exp_mat)

    def test_get_empty_square(self) -> None:
        board = Board(n=2)
        board.put(Square(1, 1), Queen)
        board.put(Square(1, 2), Queen)
        board.put(Square(2, 1), Queen)
        assert board.get_empty_square() == Square(2, 2)

        board.put(Square(2, 2), Queen)
        assert board.get_empty_square() is None

    def test_get_safe_square(self) -> None:
        board = Board(n=4)
        board.put(Square(1, 1), Rook)
        board.put(Square(2, 2), Rook)
        board.put(Square(3, 3), Rook)
        assert board.get_safe_square() == Square(4, 4)

        board.put(Square(4, 4), Rook)
        assert board.get_safe_square() is None


class TestSafeBoard:
    """
    Tests suit for queensgym.board.SafeBoard.
    """

    def test_put_unsafe(self) -> None:
        board = SafeBoard(n=5)
        assert board.get(Square(1, 1)) is None

        board.put(Square(1, 1), Queen)
        with pytest.raises(UnsafePlacementError):
            board.put(Square(2, 2), Queen)

        board.put(Square(3, 4), Queen)
        with pytest.raises(UnsafePlacementError):
            board.put(Square(5, 3), Knight)

        assert board.total_placed() == 2

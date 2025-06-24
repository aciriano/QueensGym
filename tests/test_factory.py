import pytest

from queensgym.board import Board
from queensgym.board import SafeBoard
from queensgym.exceptions import GenerationError
from queensgym.exceptions import InvalidDimensionError
from queensgym.exceptions import InvalidPieceError
from queensgym.factory import _BoardFactory
from queensgym.factory import BoardFactory
from queensgym.factory import RandomBoardFactory
from queensgym.factory import RandomSafeBoardFactory
from queensgym.factory import SafeBoardFactory
from queensgym.piece import Queen
from queensgym.square import Square


class TestBoardFactory:
    """
    Tests suit for queensgym.factory.BoardFactory.
    """

    def test_init(self) -> None:
        fact = BoardFactory(n=6)
        assert fact.n == 6

    def test_init_type_error(self) -> None:
        with pytest.raises(InvalidDimensionError):
            BoardFactory(n=[1, 2, 3])

    def test_init_max_dimension_error(self) -> None:
        with pytest.raises(InvalidDimensionError):
            BoardFactory(n=Board.max_size() + 1)

    def test_new(self) -> None:
        fact = BoardFactory(n=6)
        board = fact.new()
        assert isinstance(board, Board)
        assert board.total_placed() == 0
        assert board.n == 6


class TestSafeBoardFactory:
    """
    Tests suit for queensgym.factory.SafeBoardFactory.
    """

    def test_init(self) -> None:
        fact = SafeBoardFactory(n=6)
        assert fact.n == 6
        assert issubclass(SafeBoardFactory, _BoardFactory)

    def test_new(self) -> None:
        fact = SafeBoardFactory(n=6)
        board = fact.new()
        assert isinstance(board, SafeBoard)
        assert board.total_placed() == 0
        assert board.n == 6


class TestRandomBoardFactory:
    """
    Tests suit for queensgym.factory.RandomBoardFactory.
    """

    def test_init(self) -> None:
        fact = RandomBoardFactory(n=6, pieces=[Queen.get_id()], preplaced=(1, 1))
        assert fact.n == 6
        assert issubclass(RandomBoardFactory, _BoardFactory)

    def test_init_invalid_preplaced(self) -> None:
        for i in [None, (1, 1, 1), (1, "")]:
            with pytest.raises(TypeError):
                RandomBoardFactory(n=6, pieces=[Queen.get_id()], preplaced=i)

        for i in [(1, 37), (-1, 1)]:
            with pytest.raises(ValueError):
                RandomBoardFactory(n=6, pieces=[Queen.get_id()], preplaced=i)

    def test_init_invalid_pieces(self) -> None:
        for i in [None, ["Queen"], []]:
            with pytest.raises(TypeError):
                RandomBoardFactory(n=6, pieces=i, preplaced=(1, 5))

        for i in [[0, 1], [1, 6700000]]:
            with pytest.raises(InvalidPieceError):
                RandomBoardFactory(n=6, pieces=i, preplaced=(1, 5))

    def test_new_empty(self) -> None:
        fact = RandomBoardFactory(n=6, pieces=[Queen.get_id()], preplaced=(0, 0))
        board = fact.new()
        assert board.total_placed() == 0

    def test_new_full(self) -> None:
        fact = RandomBoardFactory(n=6, pieces=[Queen.get_id()], preplaced=(36, 36))
        board = fact.new()
        assert board.total_placed() == 36
        assert set(board.state.values()) == {Queen}

    def test_configure_board(self) -> None:
        """
        This test is artifically forced, you should not invoke the method
        `configure_board`. This is run by the factory internally.
        """
        # First, set up a board with 9/16 squares occupied by queens.
        my_board = Board(n=4)
        for i in range(1, 4):
            for j in range(1, 4):
                my_board.put(Square(i, j), Queen)

        # RandomBoardFactory cannot configure a board if the remaining empty
        # squares are less than the minimum.
        with pytest.raises(GenerationError):
            fact = RandomBoardFactory(n=4, pieces=[Queen.get_id()], preplaced=(8, 8))
            fact.configure_board(my_board)

        # If the number of empty squares is lower than the chosen number of preplaced
        # pieces, the factory must trunc it to n_preplaced = len(candidates).
        # TODO


class TestRandomSafeBoardFactory:
    """
    Tests suit for queensgym.factory.RandomSafeBoardFactory.
    """

    def test_init(self) -> None:
        fact = RandomSafeBoardFactory(n=6, pieces=[Queen.get_id()], preplaced=(1, 1))
        assert fact.errors == 6
        assert fact.n == 6
        assert issubclass(RandomSafeBoardFactory, _BoardFactory)

    def test_init_invalid_iters(self) -> None:
        with pytest.raises(ValueError):
            RandomSafeBoardFactory(n=6, pieces=[Queen.get_id()], preplaced=(1, 1), errors=0)

    def test_new_empty(self) -> None:
        fact = RandomSafeBoardFactory(n=6, pieces=[Queen.get_id()], preplaced=(0, 0))
        board = fact.new()
        assert board.total_placed() == 0
        assert isinstance(board, SafeBoard)

    def test_new(self) -> None:
        fact = RandomSafeBoardFactory(n=10, pieces=[Queen.get_id()], preplaced=(1, 3))
        for _ in range(100):
            board = fact.new()
            assert 1 <= board.total_placed() <= 3

    def test_drop_random_pieces(self) -> None:
        fact = RandomSafeBoardFactory(n=10, pieces=[Queen.get_id()], preplaced=(4, 4))
        board = fact.new()
        squares_before_deletion = set(board.squares)
        board = fact._drop_random_pieces(board, k=1)
        assert board.total_placed() == 3
        assert set(board.squares).issubset(squares_before_deletion)

    def test_configure_board(self) -> None:
        """
        This test is artifically forced, you should not invoke the method
        `configure_board`. This is run by the factory internally.
        """

        def create_solution() -> SafeBoard:
            my_board = SafeBoard(n=4)
            my_board.put(square=Square(1, 3), piece=Queen)
            my_board.put(square=Square(2, 1), piece=Queen)
            my_board.put(square=Square(3, 4), piece=Queen)
            my_board.put(square=Square(4, 2), piece=Queen)
            return my_board

        # The current arrangement is a solution for the n-queens problem
        # for n = 4. This implies that no more queens can be placed and,
        # if one is dropped, the position is still safe.
        fact = RandomSafeBoardFactory(n=4, pieces=[Queen.get_id()], preplaced=(5, 5), errors=1)
        with pytest.raises(GenerationError):
            fact.configure_board(board=create_solution())

        # The number of errors does not matter if the condition is impossible to get.
        fact = RandomSafeBoardFactory(n=4, pieces=[Queen.get_id()], preplaced=(5, 5), errors=2)
        with pytest.raises(GenerationError):
            fact.configure_board(board=create_solution())

        # If we modify the minimum number of preplaced pieces, when the factory chooses
        # 5 pieces to preplace it must drop 2 pieces and return a safe board with 3 queens in it.
        while True:
            fact = RandomSafeBoardFactory(n=4, pieces=[Queen.get_id()], preplaced=(3, 5), errors=1)
            initial_board = create_solution()
            initial_squares = set(initial_board.squares)
            new_board = fact.configure_board(board=initial_board)
            new_squares = set(new_board.squares)

            # If both boards are the same, then the factory has chosen 3 or 4 as preplaced.
            # We iter again until it chooses 5, which is impossible. In this situation, it
            # must drop 2 queens from the board.
            if initial_squares != new_squares:
                assert new_board.total_placed() == 3
                assert new_squares.issubset(initial_squares)
                break
            else:
                assert new_board.total_placed() == 4

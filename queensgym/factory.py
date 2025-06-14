"""
Example:
>>> from queensgym.piece import Queen, Rook
>>> from queensgym.factory import RandomBoardFactory
>>> preplaced = (1, 3)
>>> pieces = [Queen.get_id(), Rook.get_id()]
>>> factory = RandomBoardFactory(n=6, pieces=pieces, preplaced=preplaced)
>>> factory.new().as_matrix()
array([[0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]], dtype=int8)
>>> factory.new().as_matrix()
array([[0, 0, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]], dtype=int8)
"""
import abc
from random import choice
from random import randint
from typing import Type

from typing_extensions import override

from queensgym.board import Board
from queensgym.board import SafeBoard
from queensgym.exceptions import GenerationError
from queensgym.exceptions import InvalidDimensionError
from queensgym.exceptions import InvalidPieceError
from queensgym.exceptions import NoMoreSafeSquaresError
from queensgym.exceptions import UnsafePlacementError
from queensgym.piece import ChessPiece
from queensgym.piece import ChessPieceRegistry
from queensgym.square import Square

__all__ = ("BoardFactory", "SafeBoardFactory", "RandomBoardFactory", "RandomSafeBoardFactory")


class _BoardFactory(abc.ABC):
    """
    A generic class which creates new empty board.Board objects.
    Subclasses can replace the `new` and `configure` methods to
    initiate the boards with an arragement of pieces based on a
    predefined criteria.

    The process of building new boards follows these steps:
        1. The method _init_board() creates a new Board object.
        2. The method _configure() creates and places an arragement
            of piece.
        3. Both methods are invoked by the entrypoint new().
    """

    def __init__(self, n: int) -> None:
        if not Board.valid_dimension(n):
            raise InvalidDimensionError(
                f"Dimension {n} is invalid for Board objects. "
                f"Must be a positive integer in range [1, {Board.max_size()}]"
            )
        else:
            self.n = n

    @abc.abstractmethod
    def init_board(self) -> Board:
        raise NotImplementedError

    @abc.abstractmethod
    def configure_board(self, board: Board) -> Board:
        raise NotImplementedError

    def new(self) -> Board:
        board = self.init_board()
        return self.configure_board(board=board)


class BoardFactory(_BoardFactory):
    def init_board(self) -> Board:
        return Board(n=self.n)

    def configure_board(self, board: Board) -> Board:
        return board


class SafeBoardFactory(BoardFactory):
    @override
    def init_board(self) -> Board:
        return SafeBoard(n=self.n)


class RandomBoardFactory(BoardFactory):
    def validate(self) -> None:
        if (not isinstance(self.preplaced, tuple)) or len(self.preplaced) != 2:
            raise TypeError(f"Preplaced must be a tuple of two int. Received: {self.preplaced}.")
        elif not all(isinstance(i, int) for i in self.preplaced):
            raise TypeError(f"Preplaced must be a tuple of two int. Received: {self.preplaced}.")
        elif not (0 <= self.preplaced[0] <= self.preplaced[1] <= self.n**2):
            raise ValueError("Condition 0 <= min <= max <= n**2 is not met.")
        elif (not isinstance(self.pieces, list)) or not self.pieces:
            raise TypeError(f"Pieces must be a list of integers. Received: {self.pieces}.")
        elif not all(isinstance(p, int) for p in self.pieces):
            raise TypeError("All pieces must be integers.")
        elif not all(ChessPieceRegistry.exists(p) for p in self.pieces):
            raise InvalidPieceError("All pieces must exist in ChessPieceRegistry.")

    @override
    def __init__(self, n: int, pieces: list[int], preplaced: tuple[int, int]) -> None:
        super().__init__(n)
        self.pieces = pieces
        self.preplaced = preplaced
        self.validate()

    def choose_preplaced(self) -> int:
        return randint(self.preplaced[0], self.preplaced[1])

    def choose_piece(self) -> Type[ChessPiece]:
        return ChessPieceRegistry.get(choice(self.pieces))

    def choose_square(self, board: Board) -> Square | None:
        return board.get_empty_square()

    @override
    def configure_board(self, board: Board) -> Board:
        n_preplaced = self.choose_preplaced()
        while board.total_placed() < n_preplaced:
            square = self.choose_square(board=board)
            if square is None:
                # Then, there is no more empty squares. Board is returned.
                # NOTE: this is impossible as max(preplaced) <= n**2
                return board
            else:
                # Choose a piece and put in the board.
                piece = self.choose_piece()
                board.put(square=square, piece=piece)
        return board


class RandomSafeBoardFactory(RandomBoardFactory):
    @override
    def __init__(
        self, n: int, pieces: list[int], preplaced: tuple[int, int], iters: int | None = None
    ):
        iters = iters if isinstance(iters, int) else n
        if iters < 1:
            raise ValueError(f"Iters must be a positive integer. Value {iters} is invalid.")
        else:
            super().__init__(n, pieces, preplaced)
            self.iters = iters

    @override
    def choose_square(self, board: Board) -> Square | None:
        return board.get_safe_square()

    @override
    def init_board(self) -> Board:
        return SafeBoard(n=self.n)

    @override
    def configure_board(self, board: Board) -> Board:
        iters = 0
        n_preplaced = self.choose_preplaced()
        while board.total_placed() < n_preplaced:
            try:
                square = self.choose_square(board=board)
                if square is None:
                    raise NoMoreSafeSquaresError
                else:
                    piece = self.choose_piece()
                    board.put(square=square, piece=piece)
            except (UnsafePlacementError, NoMoreSafeSquaresError):
                # UnsafePlacementError: This can happen if the new piece has
                # not symetrical attack relation with the rest of the pieces.
                # For example, to place a queen in a safe square of a board where
                # only queens are placed is, 'a priori' and 'a posteriori', safe.
                # But, to place a knight in the same board could potentially
                # be unsafe 'a posteriori'.
                iters += 1
                if iters < self.iters:
                    # Then, reset the board and try again.
                    board.reset()
                else:
                    if board.total_placed() >= self.preplaced[0]:
                        # Then, drop random pieces and return the board at the end
                        # to avoid returning a blocked board.
                        to_delete = board.total_placed() - self.preplaced[0]
                        for _ in range(to_delete):
                            sq_to_delete = choice(board.squares)
                            board.remove(square=sq_to_delete)
                        return board
                    else:
                        raise GenerationError(
                            f"Total number of iterations have been reached ({self.iters}) without "
                            f"placing a minimum of {self.preplaced[0]} pieces in safe squares."
                        )
        return board

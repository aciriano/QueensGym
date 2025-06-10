from random import choice
from random import randint
from typing import Type

from queensgym.board import Board
from queensgym.exceptions import InvalidPieceError
from queensgym.piece import ChessPiece
from queensgym.piece import ChessPieceRegistry
from queensgym.square import Square


class BoardFactory:
    """
    A generic class which creates new empty board.Board objects.
    Subclasses can replace the `new` to initiate the boards with
    an arragement of pieces based on a predefined criteria.
    """

    def __init__(self, n: int) -> None:
        if not isinstance(n, int):
            raise TypeError(f"Board dimension must be a positive integer. Received: {n}.")
        elif not (1 <= n <= Board.max_size()):
            raise ValueError(
                f"Board dimension must be 1<=n<={Board.max_size()}. " f"Received: {n}."
            )
        else:
            self.n = n

    def new(self) -> Board:
        return Board(n=self.n)


class RandomBoardFactory(BoardFactory):
    """

    Example:
    >>> from queensgym.piece import Queen, Rook
    >>> from queensgym.factory import RandomBoardFactory
    >>> factory = RandomBoardFactory(n=6, pieces=[Queen.get_id(), Rook.get_id()], preplaced=(1, 3))
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

    def __init__(self, n: int, pieces: list[int], preplaced: tuple[int, int]) -> None:
        super().__init__(n)
        self.pieces = pieces
        self.preplaced = preplaced

        # Validate the configuration of the factory.
        self.validate_preplaced()
        self.validate_pieces()

    def validate_preplaced(self) -> None:
        type_err = TypeError(f"Preplaced must be a tuple of two int. Received: {self.preplaced}.")
        if (not isinstance(self.preplaced, tuple)) or len(self.preplaced) != 2:
            raise type_err
        elif not all(isinstance(i, int) for i in self.preplaced):
            raise type_err
        elif not (0 <= self.preplaced[0] <= self.preplaced[0] <= self.n**2):
            raise ValueError("Condition 0 <= min <= max <= n**2 is not met.")

    def validate_pieces(self) -> None:
        if not isinstance(self.pieces, list):
            raise TypeError(f"Pieces must be a list of integers. Received: {self.pieces}.")
        elif not all(isinstance(p, int) for p in self.pieces):
            raise TypeError("All pieces must be integers.")
        elif not all(ChessPieceRegistry.exists(p) for p in self.pieces):
            raise InvalidPieceError("All pieces must exist in ChessPieceRegistry.")

    def choose_preplaced(self) -> int:
        return randint(self.preplaced[0], self.preplaced[1])

    def choose_piece(self) -> Type[ChessPiece]:
        return ChessPieceRegistry.get(choice(self.pieces))

    def choose_square(self) -> Square:
        return Square(file=randint(1, self.n), rank=randint(1, self.n))

    def new(self) -> Board:
        board = super().new()
        n_preplaced = self.choose_preplaced()
        while len(board.squares) != n_preplaced:
            square = self.choose_square()
            piece = self.choose_piece()
            while square in board.squares:
                square = self.choose_square()
            board.put(square=square, piece=piece)
        return board

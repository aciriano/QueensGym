from io import StringIO
from typing import Type

import numpy as np
from typing_extensions import Self

from queensgym.exceptions import InvalidPieceError
from queensgym.exceptions import OccupiedSquareError
from queensgym.piece import ChessPiece
from queensgym.piece import ChessPieceRegistry
from queensgym.square import Square


class Board:
    """
    If you manually increase the Board.max value, please note that a
    value greater than 32_767 could cause an OveflowError in runtime
    when you invoke the method `heat_map`.
    """

    max: int = 32_767
    n: int
    state: dict[Square, Type[ChessPiece]]

    @classmethod
    def max_size(cls: Type[Self]) -> int:
        """Returns the maximum size of the board."""
        return cls.max

    def __init__(self, n: int) -> None:
        """
        Initializes a board with the given dimension.

        Args:
            n (int): The dimension of the board. Must be a positive integer
                and 1 <= n <= Board.max.

        Raises:
            TypeError: If n is not an integer.
            ValueError: If n is not a positive integer or if it is not in the
                range 1 <= n <= Board.max.
        """
        if not isinstance(n, int):
            raise TypeError(f"Board dimension must be a positive integer. Received: {n}.")
        elif not (1 <= n <= self.__class__.max_size()):
            raise ValueError(
                f"Board dimension must be 1<=n<={self.__class__.max_size()}. " f"Received: {n}."
            )
        else:
            self.n = n
            self.state = dict()

    @property
    def squares(self) -> list[Square]:
        return list(self.state.keys())

    def square_to_numpy(self, square: Square) -> tuple[int, int]:
        return self.n - square.rank, square.file - 1

    def numpy_to_square(self, np_square: tuple[int, int]) -> Square:
        return Square(np_square[1] + 1, self.n - np_square[0])

    def square_to_seq(self, square: Square) -> int:
        return (square.rank - 1) * self.n + square.file

    def seq_to_square(self, seq_square: int) -> Square:
        return Square(file=((seq_square - 1) % self.n) + 1, rank=((seq_square - 1) // self.n) + 1)

    def pprint(self) -> str:
        out = StringIO()
        for rank in range(self.n, 0, -1):
            for file in range(1, self.n + 1):
                sq = Square(file=file, rank=rank)
                if sq in self.state:
                    out.write(self.state[sq].get_symbol())
                else:
                    out.write(sq.symbol)
                out.write(" ")
            out.write("\n")
        return out.getvalue()

    def get(self, square: Square) -> Type[ChessPiece] | None:
        """Get the piece on a square.

        Args:
            square (Square): The square to get the piece from.

        Return:
            Type[ChessPiece]: The piece on the square, or None if there is no piece.
        """
        return self.state.get(square)

    def put(self, square: Square, piece: Type[ChessPiece]) -> None:
        """Place a piece on a square.

        Args:
            square (Square): The square to get the piece from.
            piece (Type[ChessPiece]): The piece to place on the square.
        """
        if not ChessPieceRegistry.exists(piece):
            raise InvalidPieceError(f"Piece {piece} is not registered.")
        elif not isinstance(square, Square):
            raise TypeError("Square must be a Square.")
        elif square in self.state:
            raise OccupiedSquareError(f"Square {square} is already occupied.")
        self.state[square] = piece

    def remove(self, square: Square) -> None:
        """Remove a piece from a square.

        Args:
            square (Square): The square to remove the piece from.
        """
        if square in self.state:
            del self.state[square]

    def reset(self) -> None:
        """Remove all the pieces from the board."""
        self.state = dict()

    def as_array(self) -> np.typing.NDArray[np.int8]:
        """Build and return the map of the board as an array representation.

        Returns:
            np.ndarray: array representation of the board.
        """
        board_arr = np.zeros(self.n * self.n, dtype=np.int8)
        for square, piece in self.state.items():
            idx = self.square_to_seq(square)
            board_arr[idx - 1] = piece.get_id()
        return board_arr

    def as_matrix(self) -> np.typing.NDArray[np.int8]:
        """Build and return the map of the board as a matrix representation.

        Returns:
            np.ndarray: matrix representation of the board.
        """
        board_map = np.zeros((self.n, self.n), dtype=np.int8)
        for square, piece in self.state.items():
            a, b = self.square_to_numpy(square)
            board_map[a][b] = piece.get_id()
        return board_map

    def as_heat_map(self) -> np.typing.NDArray[np.int16]:
        """Build and return the heat map of the board.

        NOTE could be an expensive operation if `n` is large.

        Returns:
            np.ndarray: heat map representation of the board as an
            `n` by `n` array where each position keeps the number of
            pieces that are attacking the correspondant square in the board.
        """
        heat_map = np.zeros((self.n, self.n), dtype=np.int16)
        for i in {(x, y) for x in range(1, self.n + 1) for y in range(1, self.n + 1)}:
            dst_square = Square(*i)
            a, b = self.square_to_numpy(square=Square(*i))
            for square, piece in self.state.items():
                if piece.attack(square, dst_square):
                    heat_map[a][b] += 1
        return heat_map

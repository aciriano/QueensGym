from io import StringIO
from typing import Any
from typing import Type

import numpy as np
from typing_extensions import override
from typing_extensions import Self

from queensgym.exceptions import InvalidDimensionError
from queensgym.exceptions import InvalidPieceError
from queensgym.exceptions import OccupiedSquareError
from queensgym.exceptions import UnsafePlacementError
from queensgym.piece import ChessPiece
from queensgym.piece import ChessPieceRegistry
from queensgym.square import Square


__all__ = ("Board", "SafeBoard")


class Board:
    """
    A chess board of dimension n x n.

    The board is represented as a dictionary where the keys are `Square`
    instances and the values are `ChessPiece` instances.

    The maximum size of the board is defined by the class variable `max`.
    The default value is 65_535, which is the maximum value for a 16-bit
    unsigned integer. You can change this value to increase the maximum
    size of the board. However, please note that increasing this value may
    lead to performance issues and memory consumption, especially when using
    methods like `heat_map`.
    """

    max: int = 65_535
    n: int
    state: dict[Square, Type[ChessPiece]]

    @classmethod
    def valid_dimension(cls, n: Any) -> bool:
        """Return True if the input value is valid as dimension `n` for a Board."""
        return (isinstance(n, int)) and 1 <= n <= cls.max_size()

    @classmethod
    def max_size(cls: Type[Self]) -> int:
        """Returns the maximum size of the board."""
        return cls.max

    def __init__(self, n: int) -> None:
        """
        Initializes a board with the given dimension.

        Args:
            n (int): Dimension of the board. Must be a positive integer and 1 <= n <= Board.max.

        Raises:
            InvalidDimensionError: If n is not an integer or if 1 <= n <= Board.max is not met.
        """
        if not self.valid_dimension(n):
            raise InvalidDimensionError(
                f"Dimension {n} is invalid for Board. "
                f"Must be a positive integer in range [1, {self.max_size()}]"
            )
        else:
            self.n = n
            self.state = dict()

    def __str__(self) -> str:
        return f"{self.tuples}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n={self.n}, state={self.tuples})"

    @property
    def squares(self) -> list[Square]:
        return list(self.state)

    @property
    def tuples(self) -> list[tuple[str, int, int]]:
        return [(v.__name__, k.file, k.rank) for k, v in self.state.items()]

    def total_placed(self) -> int:
        return len(self.squares)

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

        Raises:
            InvalidPieceError: If the piece is not registered in the ChessPieceRegistry.
            TypeError: If the square is not a Square instance.
            OccupiedSquareError: If the square is already occupied by another piece.
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

    def as_array(self) -> np.typing.NDArray[np.uint8]:
        """
        Build and return the map of the board as an array representation.

        Returns:
            np.ndarray: array representation of the board.
        """
        board_arr = np.zeros(self.n * self.n, dtype=np.uint8)
        for square, piece in self.state.items():
            idx = self.square_to_seq(square)
            board_arr[idx - 1] = piece.get_id()
        return board_arr

    def as_matrix(self) -> np.typing.NDArray[np.uint8]:
        """
        Build and return the map of the board as a matrix representation.

        Returns:
            np.ndarray: matrix representation of the board.
        """
        board_map = np.zeros((self.n, self.n), dtype=np.uint8)
        for square, piece in self.state.items():
            a, b = self.square_to_numpy(square)
            board_map[a][b] = piece.get_id()
        return board_map

    def as_heat_map(self) -> np.typing.NDArray[np.uint16]:
        """
        Build and return the heat map of the board.

        NOTE could be an expensive operation if `n` is large.

        Returns:
            np.ndarray: heat map representation of the board as an `n`
                by `n` array where each position keeps the number of pieces
                that are attacking the correspondant square in the board.
        """
        heat_map = np.zeros((self.n, self.n), dtype=np.uint16)
        for square in self.get_attacked():
            a, b = self.square_to_numpy(square)
            heat_map[a][b] += 1
        return heat_map

    def get_attacked(self) -> list[Square]:
        """
        Get all squares that are attacked by any piece on the board.

        The number of apparecences of a square in the list
        indicates how many pieces are attacking that square.

        Returns:
            list[Square]: A list of squares that are attacked by pieces on the board.
        """
        attacked = list()
        for square, piece in self.state.items():
            attacked.extend(piece.attacked(_from=square, limit=self.n))
        return attacked

    def get_non_attacked(self) -> set[Square]:
        """
        Get all squares that are not attacked by any piece on the board.

        Returns:
            set[Square]: A set of squares that are not attacked by any piece on the board.
        """
        attacked = self.get_attacked()
        return {
            Square(file=i, rank=j)
            for i in range(1, self.n + 1)
            for j in range(1, self.n + 1)
            if Square(file=i, rank=j) not in attacked
        }

    def get_empty(self) -> set[Square]:
        """
        Get all empty squares on the board.

        Returns:
            set[Square]: A set of empty squares on the board.
        """
        return {
            Square(file=i, rank=j)
            for i in range(1, self.n + 1)
            for j in range(1, self.n + 1)
            if Square(file=i, rank=j) not in self.state
        }


class SafeBoard(Board):
    """
    A chess board that ensures safe placement of pieces.

    This board extends the basic Board class and adds a safety check
    when placing pieces. If a piece is placed in such a way that it
    would be attacked by another piece already on the board, an
    `UnsafePlacementError` is raised.
    """

    @override
    def put(self, square: Square, piece: Type[ChessPiece]) -> None:
        """
        Place a piece on a square, ensuring that the board remains in a safe state.

        Args:
            square (Square): The square to place the piece on.
            piece (Type[ChessPiece]): The piece to place on the square.

        Raises:
            UnsafePlacementError: If placing the piece would break the safe state of the board.
        """
        super().put(square, piece)
        if not self.is_safe():
            self.remove(square)
            raise UnsafePlacementError(
                f"Putting a {piece.__name__} in {square} breaks the safe state of the board."
            )

    def is_safe(self) -> bool:
        """
        Check if the board is in a safe state. A board is considered safe if
        no two pieces can attack each other.

        Returns:
            bool: True if the board is safe, False otherwise.
        """
        for i, _from in enumerate(self.squares):
            for _to in self.squares[i + 1 : :]:
                if self.state[_from].attack(_from, _to) or self.state[_to].attack(_to, _from):
                    return False
        return True

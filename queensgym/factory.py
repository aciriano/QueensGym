import abc
import random
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


BoardObject = Board | SafeBoard


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
    def init_board(self) -> BoardObject:
        raise NotImplementedError

    @abc.abstractmethod
    def configure_board(self, board: BoardObject) -> BoardObject:
        raise NotImplementedError

    def new(self) -> BoardObject:
        board = self.init_board()
        return self.configure_board(board=board)


class BoardFactory(_BoardFactory):
    """
    A factory class to create new empty Board objects.
    """

    def init_board(self) -> Board:
        return Board(n=self.n)

    def configure_board(self, board: Board) -> Board:
        return board


class SafeBoardFactory(_BoardFactory):
    """
    A factory class to create new empty SafeBoard objects.
    """

    def init_board(self) -> SafeBoard:
        return SafeBoard(n=self.n)

    def configure_board(self, board: BoardObject) -> BoardObject:
        return board


class RandomBoardFactory(_BoardFactory):
    """
    A factory class to create new Board objects with a random arrangement of pieces.

    The arrangement is based on a predefined number of pieces to be placed and a list of
    piece IDs to choose from. The preplaced parameter defines the minimum and maximum number
    of pieces to be placed on the board.
    """

    preplaced: tuple[int, int]
    pieces: list[int]

    def validate(self) -> None:
        """
        Validates the parameters of the factory.

        Raises:
            TypeError: If preplaced is not a tuple of two integers.
            ValueError: If the condition 0 <= min <= max <= n**2 is not met.
            TypeError: If pieces is not a list of integers.
            InvalidPieceError: If any piece does not exist in ChessPieceRegistry.
        """
        if (not isinstance(self.preplaced, tuple)) or len(self.preplaced) != 2:
            raise TypeError(f"Preplaced must be a tuple of two int. Received: {self.preplaced}.")

        if not all(isinstance(i, int) for i in self.preplaced):
            raise TypeError(f"Preplaced must be a tuple of two int. Received: {self.preplaced}.")

        if not (0 <= self.preplaced[0] <= self.preplaced[1] <= self.n**2):
            raise ValueError("Condition 0 <= min <= max <= n**2 is not met.")

        if (not isinstance(self.pieces, list)) or not self.pieces:
            raise TypeError(f"Pieces must be a list of integers. Received: {self.pieces}.")

        if not all(isinstance(p, int) for p in self.pieces):
            raise TypeError("All pieces must be integers.")

        if not all(ChessPieceRegistry.exists(p) for p in self.pieces):
            raise InvalidPieceError("All pieces must exist in ChessPieceRegistry.")

    @override
    def __init__(self, n: int, pieces: list[int], preplaced: tuple[int, int]) -> None:
        super().__init__(n)
        self.pieces = pieces
        self.preplaced = preplaced
        self.validate()

    def _choose_preplaced(self) -> int:
        """Choose a random number of pieces to be preplaced in the board."""
        return random.randint(self.preplaced[0], self.preplaced[1])

    def init_board(self) -> Board:
        return Board(n=self.n)

    def configure_board(self, board: Board) -> Board:
        """
        Configure the board by placing a random number of pieces in random squares.

        Args:
            board (Board): The board to configure.

        Returns:
            Board: The configured board with pieces placed.
        """
        candidates = board.get_empty()
        # Raise an error if there are not enough empty squares to place
        # at least the minimum number of preplaced pieces.
        if len(candidates) < self.preplaced[0]:
            raise GenerationError(
                f"Not enough empty squares to place, at least, {self.preplaced[0]} "
                f"pieces. Only {len(candidates)} empty squares available."
            )

        n_preplaced = self._choose_preplaced()
        # If the number of candidates is lower then the chosen preplaced,
        # update the number of preplaced pieces to the number of available squares.
        if len(candidates) < n_preplaced:
            n_preplaced = len(candidates)

        # Select random squares and pieces to be preplaced.
        chosen_squares = random.sample(candidates, k=n_preplaced)
        chosen_pieces = random.choices(self.pieces, k=n_preplaced)

        # Put the chosen pieces in the chosen squares.
        for square, piece_id in zip(chosen_squares, chosen_pieces):
            piece = ChessPieceRegistry.get(piece_id)
            board.put(square=square, piece=piece)

        return board


class RandomSafeBoardFactory(RandomBoardFactory):
    """
    A factory class to create new SafeBoard objects with a random arrangement of pieces
    in safe squares.

    It inherits from RandomBoardFactory and overrides the methods to ensure that
    pieces are placed in safe squares only. The maximum number of iterations to place
    the pieces can be specified, and if the minimum number of pieces is not placed
    within the allowed iterations, a GenerationError is raised.

    Example:
    >>> from queensgym.piece import Queen, Rook
    >>> from queensgym.factory import RandomSafeBoardFactory
    >>> preplaced = (1, 3)
    >>> pieces = [Queen.get_id(), Rook.get_id()]
    >>> factory = RandomSafeBoardFactory(n=6, pieces=pieces, preplaced=preplaced, errors=100)
    >>> factory.new().as_matrix()
    """

    preplaced: tuple[int, int]
    pieces: list[int]
    errors: int

    @override
    def __init__(
        self, n: int, pieces: list[int], preplaced: tuple[int, int], errors: int | None = None
    ):
        """
        Initialize the RandomSafeBoardFactory with the board size, pieces, preplaced pieces,
        and the maximum number of iterations allowed for placing pieces.

        Args:
            n (int): The size of the board (n x n).
            pieces (list[int]): A list of piece IDs to choose from.
            preplaced (tuple[int, int]): A tuple defining the minimum and maximum number of pieces
                to be placed on the board.
            errors (int | None): The maximum number of iterations allowed to place pieces.

        Raises:
            ValueError: If errors is not a positive integer.
        """
        errors = errors if isinstance(errors, int) else n
        if errors < 1:
            raise ValueError(f"errors must be a positive integer. Value {errors} is invalid.")
        else:
            super().__init__(n, pieces, preplaced)
            self.errors = errors

    def _choose_piece(self) -> Type[ChessPiece]:
        """
        Choose a random piece from the available pieces.

        Returns:
            Type[ChessPiece]: A random piece class from the ChessPieceRegistry.
        """
        piece_id = random.choice(self.pieces)
        return ChessPieceRegistry.get(piece_id)

    def _choose_square(self, empty: set[Square], attacked: set[Square]) -> Square | None:
        """
        Choose a safe square from the board.

        To make the process of placing pieces more efficient, this method receives
        a set of attacked squares, which are the squares that are already
        attacked by the pieces already placed on the board. This could be computed
        by calling `board.get_attacked()`, but it is more efficient to pass it as
        an argument. This is because the Factory creates SafeBoard objects putting
        one piece at a time, and the attacked squares are updated after each
        placement.

        Args:
            empty (set[Square]): A set of empty squares on the board.
            attacked (set[Square]): A set of squares that are already attacked by pieces on the
                board.
        """
        candidates = empty.difference(attacked)
        return random.choice(list(candidates)) if len(candidates) else None

    def _drop_random_pieces(self, board: BoardObject, k: int) -> BoardObject:
        """
        Drop a random number of pieces from the board.

        Args:
            board (SafeBoard): The board from which to drop pieces.
            k (int): The number of pieces to drop.

        Returns:
            SafeBoard: The board with the specified number of pieces removed.
        """
        to_delete = random.sample(board.squares, k=k)
        for i in to_delete:
            board.remove(square=i)
        return board

    def init_board(self) -> SafeBoard:
        """Initialize a new SafeBoard object."""
        return SafeBoard(n=self.n)

    def configure_board(self, board: BoardObject) -> BoardObject:
        """
        Configure the board by placing a random number of pieces in safe squares.

        A maximum number of iterations is allowed to place the pieces.
        If the maximum number of iterations is reached without placing
        the minimum number of pieces, a GenerationError is raised.

        Args:
            board (Board): The board to configure.

        Returns:
            Board: The configured board with pieces placed.

        Raises:
            GenerationError: If the maximum number of iterations is reached
                without placing the minimum number of pieces.
        """
        attacked = set(board.get_attacked())
        n_preplaced = self._choose_preplaced()
        errors = 0
        while board.total_placed() < n_preplaced:
            try:
                square = self._choose_square(empty=board.get_empty(), attacked=attacked)
                if square is None:
                    raise NoMoreSafeSquaresError
                else:
                    piece = self._choose_piece()
                    board.put(square=square, piece=piece)
            except (UnsafePlacementError, NoMoreSafeSquaresError):
                # UnsafePlacementError: This can happen if the new piece has
                # not symetrical attack relation with the rest of the pieces.
                # For example, to place a queen in a safe square of a board where
                # only queens are placed is, 'a priori' and 'a posteriori', safe.
                # But, to place a knight in the same board could potentially
                # be unsafe 'a posteriori'.
                errors += 1
                if errors < self.errors:
                    # Then, drop random pieces and try again.
                    to_delete = random.randint(1, board.total_placed())
                    board = self._drop_random_pieces(board=board, k=to_delete)
                    attacked = set(board.get_attacked())
                else:
                    if board.total_placed() >= self.preplaced[0]:
                        # Then, drop random pieces and return the board at the end
                        # to avoid returning a blocked board.
                        to_delete = board.total_placed() - self.preplaced[0]
                        board = self._drop_random_pieces(board=board, k=to_delete)
                        return board
                    else:
                        raise GenerationError(
                            f"Total number of iterations have been reached ({self.errors}) without "
                            f"placing a minimum of {self.preplaced[0]} pieces in safe squares."
                        )
            else:
                # Update the attacked squares with the new piece placed.
                new_attacked = piece.attacked(_from=square, limit=board.n)
                attacked.update(new_attacked)
        return board

import abc
from pathlib import Path
from typing import Any
from typing import Type

from queensgym.exceptions import InvalidPieceError
from queensgym.exceptions import PieceDefinitionError
from queensgym.square import Square


__all__ = (
    "ChessPiece",
    "ChessPieceRegistry",
    "register_piece",
    "Queen",
    "Rook",
    "Bishop",
    "Knight",
    "King",
    "ICONS_FOLDER",
)


ICONS_FOLDER = Path(__file__).parent.parent / "icons"


def register_piece(cls: Type["ChessPiece"]) -> Type["ChessPiece"]:
    """
    Decorator to register a chess piece class in the ChessPieceRegistry.
    See `Queen` class for an example of usage.

    Args:
        cls (Type[ChessPiece]): The chess piece class to register.
    """
    ChessPieceRegistry.add(cls)
    return cls


class ChessPieceRegistry:
    """
    A static class to hold the registry of chess pieces.

    This class cannot be instantiated and is used to register chess pieces
    that inherit from the `ChessPiece` class. It provides methods to add
    pieces to the registry and check if a piece with a given ID exists.
    This class is designed to be a singleton, ensuring that only one instance
    of the registry exists throughout the application.
    """

    __registry: dict[int, Type["ChessPiece"]] = dict()

    @classmethod
    def add(cls, piece: Type["ChessPiece"]) -> None:
        """
        Add a chess piece to the registry.

        Args:
            piece (Type[ChessPiece]): The chess piece class to add.

        Raises:
            PieceDefinitionError: If the piece ID is already in use.
        """
        if not issubclass(piece, ChessPiece):
            raise InvalidPieceError(
                f"Only subclasses of ChessPiece can be registered. "
                f"{piece.__name__} is not a subclass of ChessPiece."
            )
        elif piece.get_id() in cls.__registry:
            raise InvalidPieceError(
                f"Id {piece.get_id()} is currently used by "
                f"piece {cls.__registry[piece.get_id()].__name__}."
            )
        cls.__registry[piece.get_id()] = piece

    @classmethod
    def get(cls, piece_id: int) -> Type["ChessPiece"]:
        """ """
        try:
            return cls.__registry[piece_id]
        except (KeyError, TypeError) as e:
            raise InvalidPieceError(
                f"Id {piece_id} is not a valid registered id for ChessPiece."
            ) from e

    @classmethod
    def exists(cls, piece: int | Type["ChessPiece"]) -> bool:
        """
        Check if a chess piece is registered in the registry.

        Args:
            piece (int | Type[ChessPiece]): The chess piece class or id to check.

        Returns:
            bool: True if the piece is registered, False otherwise.
        """
        if isinstance(piece, int):
            return piece in cls.__registry
        else:
            return piece in cls.__registry.values()


class ChessPiece(abc.ABC):
    """
    Abstract base class for chess pieces.

    This class defines the interface that all chess pieces must implement.
    It includes methods to get the piece's ID, symbol, value, icon, and
    to check if the piece can attack a target square.

    All chess pieces are static classes and must define the following class methods:
        - `get_id`: Returns the unique identifier for the chess piece.
        - `get_symbol`: Returns the symbol representing the chess piece.
        - `get_value`: Returns the value of the chess piece.
        - `get_icon`: Returns the icon path for the chess piece.
        - `_attack`: Checks if the piece can attack a target square from a source square.
    """

    @classmethod
    @abc.abstractmethod
    def get_id(cls: Type["ChessPiece"]) -> int:
        """
        Get the unique identifier for the chess piece.

        Returns:
            int: The unique ID of the chess piece.
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def get_symbol(cls: Type["ChessPiece"]) -> str:
        """
        Get the symbol representing the chess piece.

        Returns:
            str: The symbol of the chess piece.
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def get_value(cls: Type["ChessPiece"]) -> float:
        """
        Get the value of the chess piece.

        Returns:
            float: The value of the chess piece.
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def get_icon(cls: Type["ChessPiece"]) -> Path | None:
        """
        Get the icon path for the chess piece.

        Returns:
            Path | None: The path to the icon file, or None if not defined.
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def _attack(cls, _from: Square, _to: Square) -> bool:
        """
        Check if the piece can attack the target square from the source square.

        Args:
            _from (Square): The square from which the piece is moving.
            _to (Square): The target square to which the piece is moving.

        Returns:
            bool: True if the piece can attack the target square, False otherwise.
        """
        raise NotImplementedError

    def __init_subclass__(cls: Type["ChessPiece"], **kwargs: Any) -> None:
        """
        Checks if the subclass has the required class variables defined and if they are valid.

        Args:
            cls (Type[ChessPiece]): The subclass being defined.
            **kwargs: Additional keyword arguments.

        Raises:
            PieceDefinitionError: If the subclass does not define the required
                class variables or if they are invalid.
        """
        super().__init_subclass__(**kwargs)
        if (not isinstance(cls.get_id(), int)) or cls.get_id() <= 0:
            raise PieceDefinitionError(
                f"Every ChessPiece must defined its id as a positive "
                f"integer ({cls.__name__}). Value {cls.get_id()} is invalid."
            )
        elif (not isinstance(cls.get_value(), float)) or cls.get_value() <= 0:
            raise PieceDefinitionError(
                f"Every ChessPiece must defined its value as a "
                f"positive float ({cls.__name__}). Value {cls.get_value()} is invalid."
            )
        elif (not isinstance(cls.get_symbol(), str)) or len(cls.get_symbol()) != 1:
            raise PieceDefinitionError(
                f"Every ChessPiece must defined its symbol as a single-character "
                f"string ({cls.__name__}). Value {cls.get_symbol()} is invalid."
            )
        elif not isinstance(cls.get_icon(), (Path, type(None))):
            raise PieceDefinitionError(
                f"Every ChessPiece must defined its icon as a Path or None "
                f"({cls.__name__}). Value {cls.get_icon()} is invalid."
            )

    @classmethod
    def attack(cls, _from: Square, _to: Square) -> bool:
        if not isinstance(_from, Square) or not isinstance(_to, Square):
            raise TypeError("Both, _from and _to, must be Square objects.")
        elif _from == _to:
            return False
        else:
            return cls._attack(_from, _to)


@register_piece
class Queen(ChessPiece):
    """
    In chess, queens can move an arbitrary number of squares in diagonally,
    horizontally or vertically. Its value is 9.
    """

    @classmethod
    def get_id(cls: Type["ChessPiece"]) -> int:
        return 1

    @classmethod
    def get_symbol(cls: Type["ChessPiece"]) -> str:
        return "♕"

    @classmethod
    def get_value(cls: Type["ChessPiece"]) -> float:
        return 9.0

    @classmethod
    def get_icon(cls: Type["ChessPiece"]) -> Path | None:
        return ICONS_FOLDER / "queen.png"

    @classmethod
    def _attack(cls, _from: Square, _to: Square) -> bool:
        return _from.in_same_diagonal(_to) or _from.in_same_file(_to) or _from.in_same_rank(_to)


@register_piece
class Rook(ChessPiece):
    """
    In chess, rooks can move an arbitrary number of squares in horizontally and
    vertically. Its value is 5.
    """

    @classmethod
    def get_id(cls: Type["ChessPiece"]) -> int:
        return 2

    @classmethod
    def get_symbol(cls: Type["ChessPiece"]) -> str:
        return "♖"

    @classmethod
    def get_value(cls: Type["ChessPiece"]) -> float:
        return 5.0

    @classmethod
    def get_icon(cls: Type["ChessPiece"]) -> Path | None:
        return ICONS_FOLDER / "rook.png"

    @classmethod
    def _attack(cls, _from: Square, _to: Square) -> bool:
        return _from.in_same_file(_to) or _from.in_same_rank(_to)


@register_piece
class Bishop(ChessPiece):
    """
    In chess, bishops can move an arbitrary number of squares in diagonally.
    Its value is 3.
    """

    @classmethod
    def get_id(cls: Type["ChessPiece"]) -> int:
        return 3

    @classmethod
    def get_symbol(cls: Type["ChessPiece"]) -> str:
        return "♗"

    @classmethod
    def get_value(cls: Type["ChessPiece"]) -> float:
        return 3.0

    @classmethod
    def get_icon(cls: Type["ChessPiece"]) -> Path | None:
        return ICONS_FOLDER / "bishop.png"

    @classmethod
    def _attack(cls, _from: Square, _to: Square) -> bool:
        return _from.in_same_diagonal(_to)


@register_piece
class Knight(ChessPiece):
    """
    In chess, knights can move in L. This means that you can move one or two positions
    up, down, left, or right, and two or one position to the side from the previous
    square. Its value is 3.
    """

    @classmethod
    def get_id(cls: Type["ChessPiece"]) -> int:
        return 4

    @classmethod
    def get_symbol(cls: Type["ChessPiece"]) -> str:
        return "♘"

    @classmethod
    def get_value(cls: Type["ChessPiece"]) -> float:
        return 3.0

    @classmethod
    def get_icon(cls: Type["ChessPiece"]) -> Path | None:
        return ICONS_FOLDER / "knight.png"

    @classmethod
    def _attack(cls, _from: Square, _to: Square) -> bool:
        return (
            _from.distance(_to) == 2
            and not _from.in_same_file(_to)
            and not _from.in_same_rank(_to)
            and not _from.in_same_diagonal(_to)
        )


@register_piece
class King(ChessPiece):
    """
    In chess, kings can move diagonally, horizontally or vertically, but only
    one square per step. Its value is not clearly defined, but by convention,
    it is set to 4.
    """

    @classmethod
    def get_id(cls: Type["ChessPiece"]) -> int:
        return 5

    @classmethod
    def get_symbol(cls: Type["ChessPiece"]) -> str:
        return "♔"

    @classmethod
    def get_value(cls: Type["ChessPiece"]) -> float:
        return 4.0

    @classmethod
    def get_icon(cls: Type["ChessPiece"]) -> Path | None:
        return ICONS_FOLDER / "king.png"

    @classmethod
    def _attack(cls, _from: Square, _to: Square) -> bool:
        return _from.distance(_to) == 1

import abc
from pathlib import Path
from typing import ClassVar
from typing import Type
from typing import TYPE_CHECKING

from typing_extensions import Self

from queensgym.exceptions import PieceDefinitionError

if TYPE_CHECKING:
    from queensgym.core.board import Square


ICONS_FOLDER = Path(__file__).parent.parent / "icons"


class ChessPiece(abc.ABC):
    # Class variables shared by all subclasses. Icon can be
    # replaced by each piece if needed.
    registry: dict[int, Type[Self]] = dict()
    icon: ClassVar[Path | None] = None

    # Class variables that must be set by subclasses as they
    # defined the static properties of the piece.
    id: ClassVar[int]
    symbol: ClassVar[str]
    value: ClassVar[float]

    @classmethod
    @abc.abstractmethod
    def attack(cls, _from: "Square", _to: "Square") -> bool:
        raise NotImplementedError

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if (not hasattr(cls, "id")) or not isinstance(cls.id, int):
            raise PieceDefinitionError(
                f"Every ChessPiece must defined its id as an integer ({cls.__name__})."
            )
        elif (not hasattr(cls, "value")) or not isinstance(cls.value, float):
            raise PieceDefinitionError(
                f"Every ChessPiece must defined its value as a float ({cls.__name__})."
            )
        elif (not hasattr(cls, "symbol")) or not isinstance(cls.symbol, str):
            raise PieceDefinitionError(
                f"Every ChessPiece must defined its symbol as a string ({cls.__name__})."
            )
        elif cls.id in ChessPiece.registry:
            raise PieceDefinitionError(
                f"Id {cls.id} is currently used by "
                f"piece {ChessPiece.registry[cls.id].__name__}."
            )
        elif cls.value <= 0:
            raise PieceDefinitionError(
                f"ChessPiece value must be a positive float. "
                f"Value {cls.value} of piece {cls.__name__} is invalid."
            )
        elif len(cls.symbol) != 1:
            raise PieceDefinitionError(
                f"ChessPiece symbol must be a single-character string. "
                f"Value {cls.symbol} of piece {cls.__name__} is invalid."
            )
        else:
            ChessPiece.registry[cls.id] = cls


class Queen(ChessPiece):
    """
    In chess, queens can move an arbitrary number of squares in diagonally,
    horizontally or vertically. Its value is 9.
    """

    id = 1
    symbol = "♕"
    value = 9.0
    icon = ICONS_FOLDER / "queen.png"

    @classmethod
    def attack(cls, _from, _to) -> bool:
        return _from.in_same_diagonal(_to) or _from.in_same_file(_to) or _from.in_same_rank(_to)


class Rook(ChessPiece):
    """
    In chess, rooks can move an arbitrary number of squares in horizontally and
    vertically. Its value is 5.
    """

    id = 2
    symbol = "♖"
    value = 5.0
    icon = ICONS_FOLDER / "rook.png"

    @classmethod
    def attack(cls, _from, _to) -> bool:
        return _from.in_same_file(_to) or _from.in_same_rank(_to)


class Bishop(ChessPiece):
    """
    In chess, bishops can move an arbitrary number of squares in diagonally.
    Its value is 3.
    """

    id = 3
    symbol = "♗"
    value = 3.0
    icon = ICONS_FOLDER / "bishop.png"

    @classmethod
    def attack(cls, _from, _to) -> bool:
        return _from.in_same_diagonal(_to)


class Knight(ChessPiece):
    """
    In chess, knights can move in L. This means that you can move one or two positions
    up, down, left, or right, and two or one position to the side from the previous
    square. Its value is 3.
    """

    id = 4
    symbol = "♘"
    value = 3.0
    icon = ICONS_FOLDER / "knight.png"

    @classmethod
    def attack(cls, _from, _to) -> bool:
        return _from.in_same_diagonal(_to)


class King(ChessPiece):
    """
    In chess, kings can move diagonally, horizontally or vertically, but only
    one square per step. Its value is not clearly defined, but by convention,
    it is set to 4.
    """

    id = 5
    symbol = "♔"
    value = 4.0
    icon = ICONS_FOLDER / "king.png"

    @classmethod
    def attack(cls, _from, _to) -> bool:
        return _from.distance(_to) == 1

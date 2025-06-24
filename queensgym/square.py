from dataclasses import dataclass
from enum import auto
from enum import Enum

from typing_extensions import Self


class Color(Enum):
    """A logical distinction between the colors of a board."""

    WHITE = auto()
    BLACK = auto()


@dataclass(frozen=True, slots=True)
class Square:
    file: int
    rank: int

    def __post_init__(self) -> None:
        """
        Validates the attributes' values and types.

        Raises:
            TypeError: if file or rank are not integers.
            ValueError: if file or rank are not positive.
        """
        for attr in ("file", "rank"):
            attr_val = getattr(self, attr)
            if not isinstance(attr_val, int):
                raise TypeError(f"Attr '{attr}' must be int, not {type(attr_val).__name__}.")
            elif attr_val < 1:
                raise ValueError(f"Attr {attr} must be positive. Received: {attr_val}.")

    @property
    def column(self) -> int:
        """Alias for squares' file."""
        return self.file

    @property
    def row(self) -> int:
        """Alias for squares' rank."""
        return self.rank

    @property
    def color(self) -> Color:
        """Logical color of the square given its spatial location."""
        return Color.WHITE if (self.file + self.rank) % 2 else Color.BLACK

    @property
    def symbol(self) -> str:
        """Unicode character of the piece given its color."""
        return "☐" if self.color is Color.WHITE else "◼︎"

    def in_same_rank(self, other: Self) -> bool:
        if not isinstance(other, Square):
            raise NotImplementedError(
                f"Cannot determine if object of type {type(other).__name__} "
                "is in the same row as the square."
            )
        return self.rank == other.rank

    def in_same_file(self, other: Self) -> bool:
        if not isinstance(other, Square):
            raise NotImplementedError(
                f"Cannot determine if object of type {type(other).__name__} "
                "is in the same file as the square."
            )
        return self.file == other.file

    def in_same_diagonal(self, other: Self) -> bool:
        if not isinstance(other, Square):
            raise NotImplementedError(
                f"Cannot determine if object of type {type(other).__name__} "
                "is in the same diagonal as the square."
            )
        return abs(self.file - other.file) == abs(self.rank - other.rank)

    def distance(self, other: Self) -> int:
        if not isinstance(other, Square):
            raise NotImplementedError(
                f"Cannot determine the distance from the square to an "
                f"object of type {type(other).__name__}."
            )
        return max(abs(self.file - other.file), abs(self.rank - other.rank))

    def get_related(self, limit: int, offset: tuple[int, int]) -> list["Square"]:
        """
        Returns a list of squares related to the current square.

        Args:
            limit (int): The maximum value for file/rank in the output squares.
            offset (tuple[int, int]): A tuple of (file_offset, rank_offset) to apply to the square.

        Returns:
            list[Square]: A list of related squares.
        """
        if (not isinstance(limit, int)) or limit <= 0:
            raise ValueError(f"Limit must be a positive integer. '{limit}' is invalid.")

        if max(self.file, self.rank) > limit:
            raise ValueError(f"Limit must be lower than maximum of ({self.file}, {self.rank})")

        if not isinstance(offset, tuple) or len(offset) != 2:
            raise TypeError(f"Offset must be a tuple of two integers. '{offset}' is invalid.")

        if not all(isinstance(i, int) for i in offset):
            raise TypeError(f"Offset must be a tuple of two integers. '{offset}' is invalid.")

        if offset == (0, 0):
            raise ValueError(f"Offset cannot be the null vector ({offset}).")

        out = []
        dx, dy = offset

        # Initialize the current file and rank based on the offset.
        _file, _rank = self.file + dx, self.rank + dy

        # Ensure the current square is within the limits. Append it and update the file/rank
        # until it goes out of bounds.
        while max(_file, _rank) <= limit and min(_file, _rank) > 0:
            related_square = Square(file=_file, rank=_rank)
            out.append(related_square)

            # Move to the next square in the specified direction.
            _file += dx
            _rank += dy
        return out

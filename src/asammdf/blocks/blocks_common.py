from typing_extensions import Buffer, Protocol, TypeVar

_T_co = TypeVar("_T_co", covariant=True)


class UnpackFrom(Protocol[_T_co]):
    def __call__(self, buffer: Buffer, offset: int = 0) -> _T_co: ...

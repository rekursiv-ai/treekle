from typing import Literal, overload

from treekle.types import Classes, Dialect, Hooks, MutablePlainTree, PlainTree

def setup_encode(module: dict[str, object]) -> None: ...
def setup_decode(module: dict[str, object]) -> None: ...
@overload
def to_plain(
    obj: object,
    *,
    dialect: Dialect = ...,
    mutable: Literal[True] = ...,
    hooks: Hooks = ...,
    classes: Classes = ...,
    sort: bool = ...,
) -> MutablePlainTree: ...
@overload
def to_plain(
    obj: object,
    *,
    dialect: Dialect = ...,
    mutable: bool,
    hooks: Hooks = ...,
    classes: Classes = ...,
    sort: bool = ...,
) -> PlainTree: ...
@overload
def from_plain[T](
    data: object,
    target: type[T],
    *,
    hooks: Hooks = ...,
    classes: Classes = ...,
    allow_imports: bool = ...,
    strict: bool = ...,
) -> T: ...
@overload
def from_plain[T](
    data: object,
    target: type[T],
    *,
    default: None,
    hooks: Hooks = ...,
    classes: Classes = ...,
    allow_imports: bool = ...,
    strict: bool = ...,
) -> T | None: ...
@overload
def from_plain[T](
    data: object,
    target: type[T],
    *,
    default: T,
    hooks: Hooks = ...,
    classes: Classes = ...,
    allow_imports: bool = ...,
    strict: bool = ...,
) -> T: ...
@overload
def from_plain(
    data: object,
    target: object,
    *,
    default: object = ...,
    hooks: Hooks = ...,
    classes: Classes = ...,
    allow_imports: bool = ...,
    strict: bool = ...,
) -> object: ...
def json_dumps_plain(
    tree: object,
    *,
    ensure_ascii: bool = ...,
    separators: tuple[str, str] | None = ...,
    indent: int | str | None = ...,
    sort_keys: bool = ...,
) -> str: ...
def json_loads_plain(text: str | bytes) -> MutablePlainTree: ...
def json_loads_typed(
    text: str | bytes,
    target: object,
    *,
    hooks: Hooks = ...,
    classes: Classes = ...,
    allow_imports: bool = ...,
    strict: bool = ...,
) -> object: ...
def json_dumps(
    obj: object,
    *,
    dialect: Dialect = ...,
    hooks: Hooks = ...,
    classes: Classes = ...,
    sort: bool = ...,
    ensure_ascii: bool = ...,
    separators: tuple[str, str] | None = ...,
    indent: int | str | None = ...,
    sort_keys: bool = ...,
) -> str: ...

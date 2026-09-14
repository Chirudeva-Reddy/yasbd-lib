"""Partial runtime type validation for public entry points.

This module intentionally covers only what yasbd uses: plain types,
``None``, unions (``X | Y``), and subscripted generics checked via their
origin (e.g. ``Collection[str]`` validates as ``collections.abc.Collection``
without checking element types). Exotic forms (``Literal``, ``Annotated``,
``TypeVar``, nested parameters) are accepted unchecked rather than rejected.

It is a lightweight beartype replacement, not a full type checker: when in
doubt it passes the value through instead of raising.
"""

import reprlib
import typing
from collections.abc import Callable
from functools import wraps
from types import UnionType

from yasbd.exceptions import InvalidInputError

F = typing.TypeVar("F", bound=Callable)


def _trunc_repr(value):  # pragma: no cover
    """Truncate values for readable error messages."""
    # Use reprlib for auto-truncation on non-strings (faster for lists/dicts/nested)
    if isinstance(value, str) and len(value) >= 120:
        return value[:500] + "..."
    return reprlib.repr(value)


def _compile_validator(expected_type):
    """Compile a type hint into a fast, specialized validation function once at definition time."""
    if expected_type is None or expected_type is type(None):
        return lambda v: v is None

    origin = typing.get_origin(expected_type)

    # 1. Handle Unions
    if origin is UnionType or origin is typing.Union:
        sub_validators = tuple(_compile_validator(arg) for arg in typing.get_args(expected_type))
        return lambda v: any(fn(v) for fn in sub_validators)

    # 2. Handle type[...] generics
    if origin is type:
        args = typing.get_args(expected_type)
        return lambda v: isinstance(v, type) and (not args or issubclass(v, args[0]))

    # 3. Handle other generic origins (e.g., list[int])
    if origin is not None:
        return lambda v: isinstance(v, origin)

    # 4. Standard types and safe fallbacks for exotic types (Literal, etc.)
    try:
        isinstance(None, expected_type)
    except TypeError:
        # If it's an exotic type that isinstance rejects, gracefully pass through
        return lambda v: True

    return lambda v: isinstance(v, expected_type)


def _raise_validation_error(value, expected_type, name):
    """Raise the formatted invalid input error."""
    raise InvalidInputError(
        f"Invalid type for {name!r}.\n"
        f"Expected {expected_type}.\n"
        f"Found: (input={_trunc_repr(value)}, type={type(value).__name__!r})"
    )


def validate_input(fx: F) -> F:
    """Validate function arguments based on pre-compiled type hints."""
    hints = typing.get_type_hints(fx)
    ret_type = hints.pop("return", None)

    if not hints and ret_type is None:
        return fx

    # Pre-compute positional param indices (skip self)
    code = fx.__code__
    varnames = code.co_varnames
    argcount = code.co_argcount
    kwonlycount = code.co_kwonlyargcount

    pos_names = varnames[:argcount]
    kwonly_names = varnames[argcount : argcount + kwonlycount]

    is_method = pos_names and pos_names[0] in ("self", "cls")

    # Pre-compile validators for positional arguments
    pos_checks = []
    for i, name in enumerate(pos_names):
        if name not in hints or (i == 0 and is_method):
            continue
        expected = hints[name]
        pos_checks.append((i, name, _compile_validator(expected), expected))

    # Pre-compile validators for keyword-only arguments
    kw_checks = []
    for name in kwonly_names:
        if name in hints:
            expected = hints[name]
            kw_checks.append((name, _compile_validator(expected), expected))

    # Pre-compile return type validator
    ret_validator = _compile_validator(ret_type) if ret_type is not None else None

    @wraps(fx)
    def wrapper(*args, **kwargs):
        # Validate positional-or-keyword arguments
        for idx, name, validator, expected in pos_checks:
            if idx < len(args):
                if not validator(args[idx]):
                    _raise_validation_error(args[idx], expected, name)
            elif name in kwargs:
                if not validator(kwargs[name]):
                    _raise_validation_error(kwargs[name], expected, name)

        # Validate keyword-only arguments
        for name, validator, expected in kw_checks:
            if name in kwargs:
                if not validator(kwargs[name]):
                    _raise_validation_error(kwargs[name], expected, name)

        result = fx(*args, **kwargs)

        # Validate return type
        if ret_validator is not None:
            if not ret_validator(result):
                _raise_validation_error(result, ret_type, f"Return of {fx.__name__!r}")

        return result

    return wrapper

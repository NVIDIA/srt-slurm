# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Wrapper classes for formattable strings and paths.

These classes ensure that configuration values with placeholders are always
explicitly formatted before use, preventing accidental use of unformatted templates.
"""

import dataclasses
import logging
import os
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from marshmallow import fields

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FormattableString:
    """A string that may contain placeholders requiring formatting.

    Supported placeholders:
        - Environment variables: $HOME, $USER, etc.
        - Runtime placeholders: {job_id}, {run_name}, {head_node_ip}, etc.
        - Custom placeholders: any additional kwargs passed to get_string()
          (e.g., {nginx_url}, {frontend_url}, {index}, {host}, {port})

    Example:
        >>> s = FormattableString("output_{job_id}")
        >>> s.get_string(runtime)  # Must be explicitly formatted
        "output_12345"
    """

    template: str

    def get_string(
        self,
        runtime: "RuntimeContext",
        **extra_kwargs,
    ) -> str:
        """Format the template string with runtime values.

        Args:
            runtime: RuntimeContext with all runtime values
            **extra_kwargs: Additional placeholders to include in formatting

        Returns:
            Formatted string with all placeholders replaced
        """
        return runtime.format_string(self.template, **extra_kwargs)

    def raw_string(self, format_kwargs: dict[str, str] | None = None) -> str:
        """Format without runtime context (for early expansion).

        Args:
            format_kwargs: Optional format kwargs for the template

        Returns:
            Formatted string with env vars expanded
        """
        formatted = self.template.format(**format_kwargs) if format_kwargs else self.template
        return os.path.expandvars(formatted)

    def __str__(self) -> str:
        logger.error("FormattableString cannot be directly converted to a string")
        traceback.print_exc()
        return repr(self)

    def __repr__(self) -> str:
        return f"FormattableString(template={self.template!r})"


@dataclass(frozen=True)
class FormattablePath:
    """A path that may contain placeholders requiring formatting.

    Supports the same placeholders as FormattableString (including custom
    placeholders like {nginx_url}, {frontend_url}, etc.), but returns Path objects
    and handles path-specific operations like making absolute paths.

    The ensure_exists parameter will only create a directory if the path doesn't
    already exist, avoiding errors when the path points to an existing file.

    Example:
        >>> p = FormattablePath("$HOME/logs/{job_id}")
        >>> p.get_path(runtime)  # Creates directory if it doesn't exist
        Path('/home/user/logs/12345')
        >>> p = FormattablePath("$HOME/containers/model.sqsh")
        >>> p.get_path(runtime)  # Won't try to mkdir on existing file
        Path('/home/user/containers/model.sqsh')
    """

    # The path text, with `{placeholders}` and `$VARS` expanded at runtime.
    template: str

    def raw_path_no_context(
        self,
        make_absolute: bool = True,
        ensure_exists: bool = False,
        format_kwargs: dict[str, str] | None = None,
    ) -> Path:
        """Return the raw path without runtime context formatting.

        This is useful for paths that don't depend on the runtime context,
        or for early-stage path expansion before RuntimeContext exists.

        Args:
            make_absolute: If True, convert to absolute path
            ensure_exists: If True, create as directory if it doesn't exist
            format_kwargs: Additional format kwargs for the template

        Returns:
            Path object
        """
        formatted = self.template.format(**format_kwargs) if format_kwargs else self.template
        path = Path(os.path.expandvars(formatted)).expanduser()

        if make_absolute:
            path = path.resolve()
        if ensure_exists and not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        return path

    def get_path(
        self,
        runtime: "RuntimeContext",
        make_absolute: bool = True,
        ensure_exists: bool = False,
        **extra_kwargs,
    ) -> Path:
        """Format the template path and return a Path object.

        Args:
            runtime: RuntimeContext with all runtime values
            make_absolute: If True, convert to absolute path
            ensure_exists: If True, create as directory if it doesn't exist
            **extra_kwargs: Additional placeholders to include in formatting

        Returns:
            Formatted Path object
        """
        # Format using runtime context
        formatted = runtime.format_string(self.template, **extra_kwargs)
        replaced = dataclasses.replace(self, template=formatted)
        return replaced.raw_path_no_context(make_absolute=make_absolute, ensure_exists=ensure_exists)

    def __str__(self) -> str:
        logger.error("FormattablePath cannot be directly converted to a string")
        traceback.print_exc()
        return repr(self)

    def __repr__(self) -> str:
        return f"FormattablePath(template={self.template!r})"


def formattable_path_from_str(value: str) -> FormattablePath:
    """Helper to create FormattablePath from string during config parsing."""
    return FormattablePath(template=value)


def formattable_string_from_str(value: str) -> FormattableString:
    """Helper to create FormattableString from string during config parsing."""
    return FormattableString(template=value)


# Marshmallow custom fields for serialization/deserialization
class FormattablePathField(fields.Field):
    """Marshmallow field for FormattablePath that converts strings to FormattablePath."""

    def __init__(self, allow_none: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.allow_none = allow_none

    def _serialize(self, value: Any | None, attr: str | None, obj: Any, **kwargs) -> Any:
        if value is None:
            return None
        if isinstance(value, FormattablePath):
            return value.template
        return str(value)

    def _deserialize(
        self,
        value: Any,
        attr: str | None,
        data: Mapping[str, Any] | None,
        **kwargs,
    ) -> Any:
        if value is None:
            if self.allow_none:
                return None
            raise self.make_error("required")
        if isinstance(value, FormattablePath):
            return value
        if isinstance(value, str):
            return FormattablePath(template=value)
        raise self.make_error("invalid")


class FormattableStringField(fields.Field):
    """Marshmallow field for FormattableString that converts strings to FormattableString."""

    def __init__(self, allow_none: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.allow_none = allow_none

    def _serialize(self, value: Any | None, attr: str | None, obj: Any, **kwargs) -> Any:
        if value is None:
            return None
        if isinstance(value, FormattableString):
            return value.template
        return str(value)

    def _deserialize(
        self,
        value: Any,
        attr: str | None,
        data: Mapping[str, Any] | None,
        **kwargs,
    ) -> Any:
        if value is None:
            if self.allow_none:
                return None
            raise self.make_error("required")
        if isinstance(value, FormattableString):
            return value
        if isinstance(value, str):
            return FormattableString(template=value)
        raise self.make_error("invalid")

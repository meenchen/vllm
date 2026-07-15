#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Balanced final-answer extraction for cache-only AIME recovery."""

from __future__ import annotations


class BalancedBoxedExtractor:
    r"""Extract the final balanced ``\boxed{}`` or ``\fbox{}`` value."""

    _MARKERS = (r"\boxed", r"\fbox")

    def extract(self, response: str, **_: object) -> str | None:
        lowered = response.lower()
        marker_index = max(lowered.rfind(marker) for marker in self._MARKERS)
        if marker_index < 0:
            return None

        left_brace = response.find("{", marker_index)
        if left_brace < 0:
            return None

        depth = 0
        for index in range(left_brace, len(response)):
            if response[index] == "{":
                depth += 1
            elif response[index] == "}":
                depth -= 1
                if depth == 0:
                    return response[left_brace + 1 : index].strip()
        return None

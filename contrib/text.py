#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
from typing import Set

from nltk import wordpunct_tokenize


def clean_text(text: str, stop_words: Set[str] = set(), trash_symbols: str = None) -> str:
    """
    Removes `trash_symbols` from `text`, tokenize `text`, clean `stop_words`
    :return: cleaned string
    """
    if trash_symbols:
        text = text.lower().translate(dict.fromkeys(list(map(ord, trash_symbols)), None))
    return ' '.join([w for w in wordpunct_tokenize(text) if w.lower() not in stop_words])

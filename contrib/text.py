#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
from typing import Set

from nltk import wordpunct_tokenize


def clean_text(text: str, stop_words: Set[str]) -> str:
    return ' '.join([w for w in wordpunct_tokenize(text.lower())
                     if w.lower() not in stop_words])

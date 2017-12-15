#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import json
from pprint import pprint
from typing import List

import nltk
from nltk.corpus import stopwords

from contrib.text import clean_text


def get_posts(filename: str) -> List:
    with open(filename, encoding='utf-16') as data_file:
        data = json.load(data_file)
    return data


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words.update([c for c in '.,"\'?!:;()[]{}«»|/-'])

posts = get_posts('posts_1.json')
for post in posts:
    post["text"] = clean_text(post["text"], stop_words)

pprint(posts[:1])

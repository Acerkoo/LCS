from langdetect import detect
import langdetect

import sys

filename = sys.argv[1]
src = sys.argv[2]
tgt = sys.argv[3]

with open(filename, 'r') as fin:
    linelist = []
    for idx, line in enumerate(fin.readlines()):
        line = line.strip().split()
        l = len(line)
        linelist.append((line, idx))

import random
random.shuffle(linelist)

acc, srcs, en, all = 0, 0, 0, 0
for line, idx in linelist:
    all += 1
    try:
        lang = detect(' '.join(line))
        if "zh" in lang:
            lang = "zh"
        if lang == tgt:
            acc += 1
        elif lang == src:
            srcs += 1
        elif lang == 'en':
            en += 1
    except:
        pass

acc_score = acc / all * 100
src_score = srcs / all * 100
en_score = en / all * 100
print("sentence num: {}, acc: {}; src: {}; en: {}; ".format(all, acc_score, src_score, en_score))
    
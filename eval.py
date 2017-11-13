# python evalm.py --guess sample-output/task1/persian-medium-out --gold ../all/task1/persian-dev --task 1
#!/usr/bin/env python
"""
Official Evaluation Script for the CoNLL-SIGMORPHON 2017 Shared Task.
Returns accuracy and mean Levenhstein distance.
Author: Ryan Cotterell
Last Update: 05/09/2017
"""

import numpy as np
import codecs

def distance(str1, str2):
	"""Simple Levenshtein implementation for evalm."""
	m = np.zeros([len(str2)+1, len(str1)+1])
	for x in xrange(1, len(str2) + 1):
		m[x][0] = m[x-1][0] + 1
	for y in xrange(1, len(str1) + 1):
		m[0][y] = m[0][y-1] + 1
	for x in xrange(1, len(str2) + 1):
		for y in xrange(1, len(str1) + 1):
			if str1[y-1] == str2[x-1]:
				dg = 0
			else:
				dg = 1
			m[x][y] = min(m[x-1][y] + 1, m[x][y-1] + 1, m[x-1][y-1] + dg)
	return int(m[len(str2)][len(str1)])

def read(fname):
	""" read file name """
	D = {}
	with codecs.open(fname, 'rb', encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			lemma, word, tag = line.split("\t")
			if lemma not in D:
				D[lemma] = {}
			D[lemma][tag] = word
	return D


def eval_form(gold, guess, ignore=set()):
	""" compute average accuracy and edit distance for task 1 """
	correct, dist, total = 0., 0., 0.
	for lemma, D in gold.items():
		for tag, str1 in D.items():
			if (lemma, tag) in ignore:
				continue
			
			str2 = u"" # empty string if no guess
			if lemma in guess and tag in guess[lemma]:
				str2 = guess[lemma][tag]
			if str1 == str2:
				correct += 1
			dist += distance(str1, str2)
			total += 1
	return (round(correct/total*100, 2), round(dist/total, 2))

def evaluate(gold, guess, log_file, language, train_size, name):
	D_gold = read(gold)
	D_guess = read(guess)
	f= codecs.open(log_file, 'a', encoding='utf-8')
	f.write('%s \n' % (name))
	f.write("acccuracy:\t{0:.2f}\nlevenshtein:\t{1:.2f}".format(*eval_form(D_gold, D_guess)) + '\n')


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description='CoNLL-SIGMORPHON 2017 Shared Task Evaluation')
	parser.add_argument("--gold", help="Gold standard (uncovered)", required=True, type=str)
	parser.add_argument("--guess", help="Model output", required=True, type=str)
	args = parser.parse_args()    

	D_gold = read(args.gold)
	D_guess = read(args.guess)

	print "acccuracy:\t{0:.2f}\nlevenshtein:\t{1:.2f}".format(*eval_form(D_gold, D_guess))
	
# python evalm.py --guess hindi-medium-output --gold ../all/task1/persian-dev 

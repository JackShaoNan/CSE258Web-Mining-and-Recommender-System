{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Reading data...\n",
      "done\n"
     ]
    }
   ],
   "source": [
    "import numpy\n",
    "import urllib\n",
    "import scipy.optimize\n",
    "import random\n",
    "\n",
    "def parseData(fname):\n",
    "  for l in urllib.urlopen(fname):\n",
    "    yield eval(l)\n",
    "\n",
    "print \"Reading data...\"\n",
    "data = list(parseData(\"http://jmcauley.ucsd.edu/cse190/data/beer/beer_50000.json\"))\n",
    "print \"done\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [],
   "source": [
    "def feature(datum):\n",
    "  feat = [1]\n",
    "  feat.append(datum['user/ageInSeconds'])\n",
    "  return feat"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/anaconda2/lib/python2.7/site-packages/ipykernel_launcher.py:3: FutureWarning: `rcond` parameter will change to the default of machine precision times ``max(M, N)`` where M and N are the input matrix dimensions.\n",
      "To use the future default and silence this warning we advise to pass `rcond=None`, to keep using the old, explicitly pass `rcond=-1`.\n",
      "  This is separate from the ipykernel package so we can avoid doing imports until\n"
     ]
    }
   ],
   "source": [
    "X = [feature(d) for d in data]\n",
    "y = [d['review/overall'] for d in data]\n",
    "theta,residuals,rank,s = numpy.linalg.lstsq(X, y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 4.09478473e+00, -1.58610537e-10])"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "theta"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "matrix([[3.90290031e+00],\n",
       "        [7.61015628e-12]])"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X = numpy.matrix(X)\n",
    "y = numpy.matrix(y)\n",
    "numpy.linalg.inv(X.T * X) * X.T * y.T"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/anaconda2/lib/python2.7/site-packages/ipykernel_launcher.py:10: FutureWarning: `rcond` parameter will change to the default of machine precision times ``max(M, N)`` where M and N are the input matrix dimensions.\n",
      "To use the future default and silence this warning we advise to pass `rcond=None`, to keep using the old, explicitly pass `rcond=-1`.\n",
      "  # Remove the CWD from sys.path while we load stuff.\n"
     ]
    }
   ],
   "source": [
    "data2 = [d for d in data if d.has_key('user/ageInSeconds')]\n",
    "\n",
    "def feature(datum):\n",
    "  feat = [1]\n",
    "  feat.append(datum['user/ageInSeconds'])\n",
    "  return feat\n",
    "\n",
    "X = [feature(d) for d in data2]\n",
    "y = [d['review/overall'] for d in data2]\n",
    "theta,residuals,rank,s = numpy.linalg.lstsq(X, y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 4.09478473e+00, -1.58610537e-10])"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "theta"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "10389"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'beer/ABV': 8.2,\n",
       " 'beer/beerId': '37518',\n",
       " 'beer/brewerId': '14879',\n",
       " 'beer/name': 'Mean Manalishi Double I.P.A.',\n",
       " 'beer/style': 'American Double / Imperial IPA',\n",
       " 'review/appearance': 3.0,\n",
       " 'review/aroma': 4.0,\n",
       " 'review/overall': 2.5,\n",
       " 'review/palate': 2.5,\n",
       " 'review/taste': 3.0,\n",
       " 'review/text': 'poured a copper color that was ok. excellent head that stayed throughout. good carbonation and good lacing through the drink. the smell is a very intense hop aroma. the taste is an overwhelimg hop experience. i love hops but this is too much even for me. it will take a couple of hours for me palate to recover. \\t\\tmaybe too much of a good thing.\\t\\tsuckem up and movem out.\\t\\tgiblet',\n",
       " 'review/timeStruct': {'hour': 10,\n",
       "  'isdst': 0,\n",
       "  'mday': 16,\n",
       "  'min': 11,\n",
       "  'mon': 5,\n",
       "  'sec': 14,\n",
       "  'wday': 4,\n",
       "  'yday': 137,\n",
       "  'year': 2008},\n",
       " 'review/timeUnix': 1210932674,\n",
       " 'user/ageInSeconds': 1505225047,\n",
       " 'user/birthdayRaw': 'Apr 1, 1967',\n",
       " 'user/birthdayUnix': -86889600,\n",
       " 'user/gender': 'Male',\n",
       " 'user/profileName': 'giblet'}"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data[645]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[1, 2366546647]"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "max(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[1, 703436648]"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "min(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1.0"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "min([d['review/taste'] for d in data if d.has_key('review/taste')])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [],
   "source": [
    "def dis_rate(datam):\n",
    "    res = {1.0:0, 1.5:0, 2.0:0, 2.5:0, 3.0:0, 3.5:0, 4.0:0, 4.5:0, 5.0:0}\n",
    "    for d in data: \n",
    "      if d.has_key('review/taste'):\n",
    "        res[d['review/taste']] = res[d['review/taste']] + 1\n",
    "    return res"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [],
   "source": [
    "res = dis_rate(data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{1.0: 28,\n",
       " 1.5: 76,\n",
       " 2.0: 218,\n",
       " 2.5: 324,\n",
       " 3.0: 851,\n",
       " 3.5: 1761,\n",
       " 4.0: 3367,\n",
       " 4.5: 2722,\n",
       " 5.0: 1042}"
      ]
     },
     "execution_count": 70,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "res\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "10389"
      ]
     },
     "execution_count": 72,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sum(res.values())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "sum([d for d in res])"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 2",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.15"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

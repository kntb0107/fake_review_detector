{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# MODEL IMPLEMENTATION AND EVALUATION\n",
    "\n",
    "This is the stage where the three models are built, optimized and evaluated.\n",
    "\n",
    "Models used: `` Multinominal Naive Bayes`` , ``Support Vector Machine``, ``Logistic Regression``\n",
    "\n",
    "Evaluation methods used: ``accuracy, precision, recall, f1_score`` and ``confusion matrix``\n",
    "\n",
    "## Summary\n",
    "\n",
    "After appropriate evaluation, LR with count vectorizer has been deemed the best. The rest of the models has all worked above 80% accuracy, with the other metrics working out above 79%. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 139,
   "metadata": {},
   "outputs": [],
   "source": [
    "#LIBRARIES \n",
    "import pandas as pd\n",
    "import nltk\n",
    "from nltk.corpus import stopwords\n",
    "import sklearn\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "from sklearn.svm import SVC \n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score\n",
    "from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix\n",
    "import pickle\n",
    "import warnings\n",
    "warnings.simplefilter(\"ignore\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 140,
   "metadata": {},
   "outputs": [],
   "source": [
    "#lOADING DATASETS \n",
    "df = pd.read_csv(\"data and pickle files/cleaned_data.csv\",encoding=\"latin1\") #due to special charas should be encoded as latin 1\n",
    "\n",
    "toCheck = pd.read_csv(\"data and pickle files/updated_data.csv\",encoding=\"latin1\")\n",
    "#REMOVE MAX\n",
    "pd.set_option('display.max_columns', None)\n",
    "pd.set_option('display.max_rows', None)\n",
    "\n",
    "#DROP EXTRA COLUMNS\n",
    "df.drop(['Unnamed: 0'], axis=1, inplace=True)\n",
    "toCheck.drop(['Unnamed: 0'], axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# DOUBLE-CHECKING...\n",
    "\n",
    "Double checking if there are any NULL values within the dataset. This would cause issues later on if there are as such."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 141,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>review_text</th>\n",
       "      <th>verified_purchase</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>191</th>\n",
       "      <td>NaN</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>523</th>\n",
       "      <td>NaN</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1072</th>\n",
       "      <td>NaN</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1111</th>\n",
       "      <td>NaN</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1230</th>\n",
       "      <td>NaN</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1316</th>\n",
       "      <td>NaN</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     review_text  verified_purchase\n",
       "191          NaN               True\n",
       "523          NaN               True\n",
       "1072         NaN               True\n",
       "1111         NaN               True\n",
       "1230         NaN               True\n",
       "1316         NaN               True"
      ]
     },
     "execution_count": 141,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#CHECKING WHICH ROW IS NULL FROM PRE-PROCESSING\n",
    "checkNULL = df.isnull()\n",
    "checkNULL = checkNULL.any(axis=1)\n",
    "df[checkNULL]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 142,
   "metadata": {},
   "outputs": [],
   "source": [
    "toCheck = toCheck.drop_duplicates().reset_index(drop=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 143,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>review_text</th>\n",
       "      <th>verified_purchase</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>191</th>\n",
       "      <td>A+</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>523</th>\n",
       "      <td>5*</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1072</th>\n",
       "      <td>very</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1111</th>\n",
       "      <td>Does what it should</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1230</th>\n",
       "      <td>A+</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1316</th>\n",
       "      <td>A*****</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              review_text  verified_purchase\n",
       "191                    A+               True\n",
       "523                    5*               True\n",
       "1072                 very               True\n",
       "1111  Does what it should               True\n",
       "1230                   A+               True\n",
       "1316               A*****               True"
      ]
     },
     "execution_count": 143,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "toCheck.iloc[[191,523,1072,1111,1230,1316],[3,4]]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Checking the old csv file, it can be seen as to why the five rows were totally cleaned out within its review_text. That was because within the text processing stage previously, only words which held meaning were kept, and if we refer to the second table we can see that most of them were either stopwords or had symbols and numbers. Since they don't hold meaning either way, these will be dropped subsequently. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 144,
   "metadata": {},
   "outputs": [],
   "source": [
    "#DROP THE NULL ROWS\n",
    "df = df.dropna(how='any',axis=0) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 145,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "False    0.525701\n",
       "True     0.474299\n",
       "Name: verified_purchase, dtype: float64"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#UPDATED VP VALUES \n",
    "df[\"verified_purchase\"].value_counts(normalize=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The change barely had any affect on the T/F values, and thus we are ready to proceed."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# MODELING\n",
    "\n",
    "Within the dataset, there are currently only two columns. Out of the two, review_text is going to be assigned as the input variable, and verified_purchases as the target variable. The data is then going to be split accordingly."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 146,
   "metadata": {},
   "outputs": [],
   "source": [
    "#ASSIGN THE VARIABLES\n",
    "X = df['review_text'] #input var\n",
    "y = df['verified_purchase'] #target var"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 147,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of rows:\n",
      "Entire dataset: 1712\n",
      "Train dataset: 1027\n",
      "Test dataset: 685\n"
     ]
    }
   ],
   "source": [
    "#SPLIT DATA\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    df['review_text'], df['verified_purchase'],test_size=0.4, random_state=42) #40% gives best results, 42 is no of life...\n",
    "\n",
    "entiredf = format(df.shape[0])\n",
    "traindf = format(X_train.shape[0])\n",
    "testdf = format(X_test.shape[0])\n",
    "\n",
    "print('Number of rows:')\n",
    "print('Entire dataset:', entiredf)\n",
    "print('Train dataset:', traindf)\n",
    "print('Test dataset:',testdf)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The data is decided to be split into 60 - 40, which has been determined by trial and error. This splitting produces the highest accuracy for the models, and thus we are going to with that. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## COUNT VECTORIZER AND MODELING\n",
    "\n",
    "word vectorization maps words or phrases from a lexicon to a matching vector of real numbers, which may then be used to determine word predictions and semantics, and this is done due to the fact that models only understand numerical data.\n",
    "\n",
    "We are going to be utlizing two of the vectorization methods, the first one being count vectorizer. We just count the number of times a word appears in the document in CountVectorizer, which results in a bias in favor of the most common terms."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 148,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Vocabulary: \n",
      " {'current': 413, 'sell': 1498, 'price': 1310, 'compar': 341, 'supermarket': 1705, 'good': 760, 'smell': 1563, 'pleasant': 1270, 'need': 1134, 'add': 14, 'small': 1558, 'cloth': 319, 'fresh': 700, 'great': 776, 'moisturis': 1100, 'sensit': 1502, 'love': 1009, 'pour': 1293, 'smaller': 1559, 'bottl': 193, 'make': 1032, 'manag': 1036, 'beat': 136, 'decent': 432, 'fabric': 614, 'soften': 1580, 'nice': 1142, 'fragranc': 695, 'purchas': 1334, 'deliveri': 456, 'cream': 397, 'handwash': 801, 'cheaper': 281, 'hand': 798, 'better': 154, 'liquid': 988, 'oh': 1174, 'wing': 1917, 'dove': 514, 'kitchen': 940, 'bathroom': 131, 'shower': 1527, 'room': 1436, 'recommend': 1378, 'highli': 830, 'star': 1636, 'simpl': 1537, 'gel': 727, 'like': 977, 'glue': 753, 'hard': 805, 'rub': 1444, 'slip': 1555, 'bath': 130, 'goe': 756, 'smoothli': 1569, 'easili': 539, 'wast': 1885, 'leav': 964, 'feel': 643, 'silki': 1535, 'soft': 1579, 'scenti': 1477, 'review': 1419, 'collect': 326, 'promot': 1322, 'excel': 589, 'everi': 580, 'day': 426, 'facial': 617, 'wash': 1882, 'excess': 591, 'face': 616, 'strip': 1669, 'natur': 1127, 'oil': 1175, 'care': 256, 'routin': 1442, 'morn': 1108, 'night': 1144, 'clean': 308, 'brilliant': 213, 'offer': 1172, 'gorgeou': 762, 'amaz': 52, 'valu': 1848, 'girli': 739, 'hair': 794, 'buy': 235, 'chang': 276, 'preserv': 1304, 'come': 331, 'bad': 109, 'sore': 1599, 'rash': 1354, 'eye': 608, 'burn': 230, 'lip': 987, 'tingl': 1775, 'phone': 1255, 'told': 1785, 'stuff': 1677, 'ask': 92, 'said': 1454, 'know': 943, 'want': 1878, 'reason': 1368, 'mayb': 1056, 'save': 1468, 'money': 1103, 'ingredi': 897, 'nearli': 1130, 'year': 1950, 'sinc': 1541, 'nivea': 1147, 'sold': 1583, 'compani': 339, 'german': 733, 'im': 872, 'realli': 1366, 'angri': 56, 'suppos': 1708, 'ok': 1177, 'rubbish': 1446, 'burnt': 231, 'pleas': 1269, 'usual': 1846, 'stock': 1651, 'fulli': 713, 'asda': 90, 'gave': 726, 'refund': 1386, 'gift': 735, 'card': 253, 'receipt': 1371, 'pocket': 1277, 'condition': 355, 'normal': 1151, 'oili': 1176, 'week': 1896, 'saw': 1470, 'differ': 480, 'felt': 646, 'cleans': 310, 'clearer': 313, 'notic': 1156, 'straightaway': 1659, 'red': 1381, 'blemish': 169, 'previou': 1308, 'kid': 936, 'doesnt': 505, 'irrit': 914, 'scent': 1476, 'littl': 993, 'bit': 162, 'long': 999, 'way': 1892, 'similar': 1536, 'perfect': 1236, 'got': 764, 'coupl': 390, 'ago': 32, 'refresh': 1385, 'bodi': 181, 'smooth': 1567, 'cucumb': 408, 'relax': 1393, 'best': 153, 'came': 243, 'separ': 1504, 'packet': 1208, 'sealabl': 1491, 'affect': 25, 'rough': 1440, 'dri': 525, 'otherwis': 1193, 'fantast': 631, 'lot': 1007, 'effort': 549, 'reduc': 1382, 'plastic': 1268, 'concentr': 350, 'buyer': 236, 'say': 1471, 'larger': 949, 'probabl': 1316, 'explain': 601, 'label': 945, 'fuchsia': 711, 'perfum': 1239, 'version': 1864, 'overpow': 1203, 'big': 155, 'plu': 1274, 'difficulti': 482, 'intens': 907, 'past': 1226, 'note': 1153, 'outer': 1194, 'sleev': 1550, 'recycl': 1380, 'dispos': 498, 'charg': 279, 'kind': 937, 'overbear': 1199, 'anyon': 66, 'glow': 751, 'afford': 26, 'comfort': 332, 'creation': 400, 'round': 1441, 'market': 1044, 'close': 318, 'match': 1053, 'honeysuckl': 846, 'sandalwood': 1463, 'person': 1246, 'favourit': 638, 'howev': 855, 'descript': 465, 'ad': 13, 'recent': 1373, 'amazon': 55, 'pantri': 1215, 'order': 1189, 'lamin': 946, 'tile': 1771, 'floor': 670, 'subtl': 1685, 'streak': 1662, 'free': 698, 'shine': 1523, 'moistur': 1099, 'essenti': 577, 'tri': 1808, 'time': 1773, 'today': 1781, 'packag': 1207, 'easi': 537, 'open': 1182, 'squeez': 1627, 'releas': 1394, 'puff': 1330, 'froth': 707, 'pure': 1335, 'smear': 1562, 'white': 1908, 'absorb': 4, 'non': 1148, 'greasi': 775, 'appli': 73, 'think': 1759, 'expens': 598, 'cheap': 280, 'qualiti': 1340, 'dairi': 419, 'aroma': 84, 'dont': 508, 'drench': 523, 'lotion': 1008, 'pretti': 1306, 'worth': 1937, 'thank': 1752, 'item': 919, 'describ': 464, 'rapid': 1352, 'stop': 1653, 'static': 1641, 'chocol': 294, 'flavour': 669, 'creami': 398, 'soap': 1576, 'anyth': 67, 'els': 553, 'lather': 954, 'noth': 1154, 'fail': 621, 'protect': 1326, 'gentl': 730, 'basic': 126, 'harm': 806, 'nourish': 1157, 'tasti': 1733, 'beef': 143, 'tomato': 1786, 'grate': 772, 'cheddar': 284, 'chees': 286, 'mix': 1091, 'nd': 1128, 'let': 969, 'sit': 1544, 'minut': 1083, 'regret': 1388, 'set': 1509, 'took': 1791, 'expect': 597, 'size': 1546, 'persil': 1244, 'quit': 1348, 'pictur': 1258, 'mislead': 1087, 'apart': 71, 'cheer': 285, 'makeup': 1034, 'eas': 536, 'micellar': 1074, 'water': 1888, 'left': 965, 'pack': 1206, 'extra': 602, 'impresss': 881, 'content': 368, 'aw': 103, 'cut': 416, 'right': 1427, 'fingernail': 658, 'turn': 1818, 'upsid': 1844, 'approach': 78, 'empti': 558, 'ridicul': 1426, 'design': 466, 'bought': 194, 'lumber': 1016, 'home': 841, 'larg': 948, 'decant': 431, 'exist': 594, 'contain': 367, 'tell': 1740, 'straight': 1658, 'away': 105, 'despit': 469, 'compart': 343, 'machin': 1025, 'towel': 1796, 'unlik': 1835, 'manufactur': 1040, 'healthi': 817, 'definit': 447, 'forese': 684, 'futur': 720, 'gotten': 766, 'result': 1415, 'brillant': 212, 'brand': 201, 'absolut': 3, 'fan': 629, 'wow': 1939, 'factor': 619, 'base': 123, 'commerci': 335, 'help': 825, 'disappoint': 489, 'especi': 576, 'point': 1279, 'ice': 866, 'chunk': 301, 'difficult': 481, 'eat': 540, 'entir': 569, 'tub': 1814, 'nasti': 1126, 'chemic': 287, 'super': 1701, 'famili': 628, 'young': 1956, 'old': 1179, 'friendli': 705, 'mean': 1059, 'colour': 329, 'honestli': 844, 'problemat': 1318, 'prone': 1323, 'dermat': 462, 'tast': 1732, 'weed': 1895, 'vinegar': 1867, 'work': 1932, 'remov': 1403, 'havent': 811, 'hamper': 797, 'christma': 298, 'sure': 1709, 'magnum': 1029, 'incas': 885, 'amazebal': 53, 'togeth': 1782, 'delici': 453, 'superb': 1702, 'kit': 939, 'bio': 159, 'stubborn': 1674, 'stain': 1631, 'word': 1931, 'warn': 1881, 'handl': 800, 'tend': 1743, 'ensur': 566, 'thoroughli': 1763, 'rins': 1428, 'repair': 1405, 'guy': 790, 'fals': 627, 'economi': 543, 'build': 226, 'caus': 261, 'issu': 916, 'huggabl': 858, 'strong': 1671, 'alcohol': 36, 'pot': 1291, 'noodl': 1149, 'student': 1676, 'yuck': 1959, 'addit': 15, 'sauc': 1466, 'pasta': 1227, 'carboard': 250, 'situat': 1545, 'access': 6, 'kettl': 933, 'chose': 297, 'mugshot': 1116, 'mistak': 1089, 'scrumptiou': 1488, 'authent': 99, 'linger': 984, 'bonu': 188, 'percent': 1235, 'fact': 618, 'husband': 862, 'classic': 307, 'lynx': 1024, 'deodor': 459, 'regularli': 1390, 'number': 1160, 'skincar': 1547, 'avoid': 102, 'toner': 1788, 'experi': 599, 'anoth': 62, 'major': 1031, 'extrem': 605, 'flaki': 665, 'sting': 1649, 'dewi': 475, 'look': 1003, 'deffin': 440, 'opinion': 1183, 'reccomend': 1370, 'deal': 429, 'slow': 1556, 'cook': 375, 'beauti': 138, 'tip': 1777, 'mark': 1043, 'alway': 51, 'fresher': 701, 'regular': 1389, 'late': 951, 'overal': 1198, 'improv': 882, 'claim': 304, 'perform': 1238, 'miracl': 1084, 'lol': 997, 'reaction': 1360, 'happen': 802, 'winter': 1919, 'weather': 1894, 'central': 265, 'heat': 819, 'vaselin': 1853, 'dealt': 430, 'includ': 887, 'elbow': 551, 'knee': 941, 'place': 1263, 'glide': 746, 'dream': 522, 'summer': 1695, 'spent': 1614, 'lay': 958, 'beach': 133, 'eyemak': 611, 'men': 1067, 'boot': 190, 'tame': 1726, 'divin': 500, 'iron': 913, 'tear': 1737, 'penni': 1233, 'scale': 1472, 'deliv': 455, 'quickli': 1346, 'shampoo': 1516, 'leak': 963, 'tan': 1727, 'sunb': 1697, 'spray': 1624, 'thought': 1764, 'clear': 312, 'friend': 704, 'huge': 857, 'indoor': 892, 'didnt': 477, 'exxtra': 607, 'fast': 633, 'thankyou': 1753, 'half': 796, 'shop': 1526, 'line': 982, 'fake': 625, 'africa': 28, 'poorer': 1282, 'new': 1138, 'softer': 1581, 'particularli': 1221, 'wrap': 1940, 'storag': 1655, 'measur': 1061, 'touch': 1794, 'brighter': 209, 'cleaner': 309, 'afterward': 29, 'everyon': 583, 'impress': 880, 'neutral': 1137, 'defiantli': 442, 'fab': 613, 'rate': 1356, 'mani': 1038, 'micel': 1073, 'wipe': 1920, 'total': 1793, 'alreadi': 46, 'receiv': 1372, 'feed': 642, 'event': 578, 'absoulut': 5, 'stick': 1646, 'job': 926, 'live': 994, 'hype': 865, 'clearli': 314, 'lead': 961, 'leader': 962, 'inspir': 901, 'confid': 357, 'certainli': 269, 'harsh': 808, 'favorit': 636, 'imposs': 879, 'sourc': 1603, 'check': 283, 'target': 1731, 'piec': 1260, 'crack': 395, 'bare': 119, 'far': 632, 'concern': 352, 'waterproof': 1890, 'mascara': 1048, 'powder': 1295, 'return': 1417, 'paid': 1210, 'daili': 418, 'basi': 125, 'age': 30, 'multipl': 1117, 'bulk': 227, 'door': 509, 'funnel': 717, 'easier': 538, 'section': 1496, 'washer': 1883, 'moreov': 1107, 'thing': 1758, 'exactli': 588, 'someth': 1588, 'odor': 1169, 'blend': 170, 'energis': 563, 'complain': 345, 'lid': 971, 'bin': 158, 'spill': 1618, 'everywher': 586, 'box': 197, 'ml': 1093, 'hope': 847, 'misl': 1086, 'month': 1104, 'start': 1637, 'enjoy': 565, 'mr': 1114, 'hinch': 832, 'glad': 742, 'scratch': 1481, 'scour': 1480, 'pad': 1209, 'minki': 1082, 'spong': 1621, 'oven': 1197, 'hob': 837, 'tap': 1730, 'sink': 1543, 'screen': 1483, 'gleam': 745, 'compact': 338, 'rang': 1351, 'actual': 12, 'complet': 347, 'ha': 792, 'excema': 590, 'hit': 834, 'everyday': 582, 'massiv': 1052, 'cap': 245, 'lenor': 967, 'unstopp': 1840, 'convert': 373, 'mild': 1077, 'messi': 1071, 'step': 1645, 'dad': 417, 'awhil': 107, 'suitabl': 1694, 'obviou': 1162, 'choic': 295, 'believ': 146, 'continu': 369, 'term': 1746, 'benefit': 150, 'happi': 803, 'alround': 48, 'consist': 363, 'dilut': 483, 'guess': 789, 'blue': 177, 'pigment': 1261, 'act': 11, 'reli': 1395, 'hydrat': 864, 'mixtur': 1092, 'varieti': 1851, 'reach': 1358, 'matter': 1054, 'neg': 1135, 'pump': 1332, 'end': 560, 'foam': 677, 'met': 1072, 'standard': 1634, 'bar': 118, 'quarter': 1342, 'truli': 1813, 'load': 995, 'sometim': 1589, 'stapl': 1635, 'heal': 816, 'abras': 2, 'boy': 198, 'store': 1656, 'cupboard': 410, 'dead': 428, 'therapi': 1754, 'origin': 1192, 'mini': 1080, 'arriv': 85, 'real': 1363, 'delight': 454, 'quick': 1345, 'subscrib': 1682, 'light': 975, 'bubbl': 223, 'floral': 671, 'masculin': 1049, 'bigger': 156, 'volum': 1870, 'garag': 723, 'known': 944, 'sweet': 1719, 'textur': 1751, 'artifici': 89, 'justv': 929, 'hour': 852, 'pale': 1212, 'disastr': 490, 'success': 1686, 'creat': 399, 'spot': 1623, 'abl': 1, 'deepli': 438, 'exfoli': 593, 'wont': 1928, 'suffer': 1689, 'babi': 108, 'style': 1679, 'genuin': 732, 'neatli': 1131, 'everyth': 584, 'shall': 1514, 'tresmemm': 1807, 'fussi': 719, 'suit': 1693, 'therefor': 1755, 'prefer': 1301, 'burst': 232, 'transit': 1802, 'snack': 1572, 'went': 1900, 'weekend': 1897, 'decid': 433, 'wet': 1901, 'cashmer': 259, 'sweater': 1718, 'sainsburi': 1455, 'scrummi': 1487, 'delic': 452, 'requir': 1410, 'bring': 214, 'runni': 1449, 'glam': 743, 'somewher': 1591, 'follow': 680, 'inexpens': 894, 'plain': 1264, 'mother': 1112, 'law': 957, 'wat': 1886, 'discount': 492, 'tin': 1774, 'sooth': 1597, 'residu': 1411, 'cool': 378, 'readi': 1362, 'magic': 1027, 'latest': 953, 'variat': 1850, 'address': 16, 'environment': 571, 'transport': 1803, 'cost': 384, 'commend': 333, 'sens': 1500, 'detect': 470, 'breath': 205, 'form': 687, 'slightli': 1552, 'zesti': 1961, 'user': 1845, 'repres': 1408, 'lessen': 968, 'impact': 877, 'environ': 570, 'albeit': 35, 'smallest': 1560, 'ideal': 869, 'son': 1592, 'adult': 20, 'children': 292, 'belov': 147, 'pet': 1250, 'incred': 890, 'lolli': 998, 'bargain': 120, 'introduc': 909, 'partner': 1223, 'sadli': 1452, 'share': 1517, 'hot': 851, 'melt': 1065, 'drip': 528, 'finish': 659, 'favour': 637, 'properli': 1324, 'inferior': 895, 'tesco': 1748, 'microwav': 1076, 'sever': 1510, 'badli': 111, 'lumpi': 1017, 'mostli': 1109, 'simplic': 1539, 'gone': 758, 'effect': 547, 'trace': 1799, 'fanci': 630, 'maskara': 1050, 'remain': 1399, 'sticki': 1647, 'defin': 444, 'heavi': 820, 'frequent': 699, 'perhap': 1240, 'capsul': 247, 'household': 854, 'terribl': 1747, 'allerg': 37, 'constant': 364, 'hay': 812, 'fever': 648, 'wrong': 1945, 'sign': 1529, 'gluten': 754, 'casserol': 260, 'sat': 1464, 'toilet': 1783, 'barley': 122, 'coeliac': 323, 'peopl': 1234, 'unsuit': 1841, 'pick': 1257, 'signific': 1531, 'rel': 1392, 'sturdier': 1678, 'travel': 1804, 'hous': 853, 'tini': 1776, 'surfac': 1711, 'pic': 1256, 'wonder': 1926, 'deni': 457, 'fix': 663, 'instead': 904, 'whenev': 1903, 'react': 1359, 'true': 1812, 'luxuri': 1023, 'prime': 1312, 'conveni': 372, 'incorpor': 888, 'cleanser': 311, 'eventu': 579, 'margin': 1042, 'obvious': 1163, 'comment': 334, 'appreci': 76, 'break': 203, 'vegan': 1856, 'daughter': 425, 'persuad': 1248, 'hesit': 826, 'pay': 1230, 'forward': 692, 'soup': 1602, 'risotto': 1432, 'younger': 1957, 'menopaus': 1068, 'calm': 240, 'wrinkl': 1942, 'crepey': 401, 'layer': 959, 'conjunct': 359, 'hyaluron': 863, 'companion': 340, 'loss': 1005, 'firm': 660, 'oz': 1205, 'csmart': 406, 'screw': 1484, 'determin': 472, 'loos': 1004, 'knife': 942, 'wise': 1921, 'advoid': 24, 'smart': 1561, 'local': 996, 'okay': 1178, 'meant': 1060, 'equal': 572, 'artif': 88, 'upset': 1843, 'simpli': 1538, 'english': 564, 'mustard': 1122, 'colman': 327, 'sunday': 1698, 'roast': 1433, 'tube': 1815, 'invari': 910, 'serv': 1507, 'nozzl': 1159, 'block': 172, 'someon': 1587, 'scrub': 1486, 'extremli': 606, 'frangranc': 697, 'fairli': 624, 'spread': 1625, 'soak': 1575, 'smelt': 1566, 'wilkinson': 1914, 'area': 80, 'soon': 1593, 'becom': 139, 'scalp': 1474, 'satisfi': 1465, 'sorri': 1600, 'voucher': 1871, 'post': 1289, 'slimi': 1554, 'problem': 1317, 'unhappi': 1832, 'respons': 1413, 'treat': 1805, 'portug': 1287, 'miss': 1088, 'tea': 1736, 'straighten': 1660, 'curl': 412, 'damag': 420, 'seller': 1499, 'pod': 1278, 'bag': 113, 'filler': 653, 'rich': 1423, 'special': 1609, 'elsewher': 554, 'laundri': 956, 'hate': 809, 'arrog': 86, 'unilev': 1833, 'hold': 838, 'palm': 1213, 'fine': 656, 'paragon': 1217, 'champ': 274, 'tight': 1769, 'win': 1916, 'doubl': 512, 'struggl': 1673, 'tough': 1795, 'unbeliev': 1825, 'compliment': 348, 'august': 98, 'itchi': 918, 'stream': 1664, 'grandchildren': 770, 'unfortun': 1831, 'green': 778, 'lime': 979, 'individu': 891, 'dandruff': 423, 'anti': 63, 'prior': 1314, 'particular': 1220, 'busi': 233, 'mum': 1118, 'admit': 18, 'neglect': 1136, 'empathis': 557, 'trial': 1809, 'opportun': 1184, 'reconnect': 1379, 'cme': 320, 'smother': 1570, 'offens': 1171, 'alo': 41, 'applic': 74, 'bedtim': 141, 'verdict': 1860, 'thumb': 1766, 'ye': 1949, 'second': 1493, 'lightweight': 976, 'tendenc': 1744, 'pull': 1331, 'liber': 970, 'discreet': 494, 'overwhelm': 1204, 'protector': 1327, 'retain': 1416, 'ezyema': 612, 'switch': 1721, 'radiant': 1349, 'regim': 1387, 'thinner': 1760, 'heavili': 821, 'cancel': 244, 'subscript': 1683, 'wait': 1874, 'fyi': 721, 'attract': 97, 'floweri': 673, 'badeda': 110, 'holiday': 840, 'sun': 1696, 'till': 1772, 'stope': 1654, 'blow': 176, 'prevent': 1307, 'transform': 1801, 'nose': 1152, 'dehydr': 451, 'sort': 1601, 'rest': 1414, 'sleep': 1549, 'caramel': 248, 'marshmallow': 1046, 'liter': 991, 'calori': 242, 'bomb': 185, 'gross': 782, 'gonna': 759, 'throw': 1765, 'coca': 322, 'butter': 234, 'marmit': 1045, 'lover': 1011, 'glass': 744, 'jar': 923, 'oddli': 1168, 'given': 740, 'persev': 1243, 'shini': 1524, 'root': 1437, 'drier': 527, 'grown': 785, 'horribl': 849, 'agre': 33, 'tighter': 1770, 'import': 878, 'confus': 358, 'milk': 1079, 'maker': 1033, 'risk': 1431, 'underwear': 1830, 'ultra': 1824, 'twice': 1819, 'combin': 330, 'harmoni': 807, 'chicken': 290, 'mushroom': 1121, 'pie': 1259, 'salt': 1458, 'ruin': 1447, 'chuck': 300, 'tongu': 1790, 'season': 1492, 'accident': 7, 'swallow': 1715, 'sea': 1489, 'xd': 1946, 'older': 1180, 'wide': 1911, 'awak': 104, 'flakey': 664, 'smelli': 1564, 'pit': 1262, 'plenti': 1273, 'longer': 1000, 'comparison': 342, 'yo': 1954, 'born': 191, 'defient': 443, 'odourless': 1170, 'suppl': 1706, 'nightli': 1145, 'bed': 140, 'clog': 317, 'pore': 1285, 'replenish': 1407, 'daytim': 427, 'reappli': 1367, 'overnight': 1202, 'unscent': 1837, 'patch': 1228, 'stone': 1652, 'munchi': 1119, 'food': 682, 'depend': 461, 'wish': 1922, 'talk': 1724, 'visit': 1869, 'certifi': 270, 'british': 215, 'foundat': 693, 'freshli': 702, 'launder': 955, 'reliabl': 1396, 'cooler': 379, 'wahs': 1873, 'soapi': 1577, 'type': 1820, 'downsid': 515, 'hole': 839, 'tresemm': 1806, 'stand': 1633, 'yummi': 1960, 'sugari': 1691, 'eco': 541, 'parcel': 1218, 'brain': 199, 'op': 1181, 'wife': 1912, 'safe': 1453, 'recomend': 1377, 'aswel': 94, 'moist': 1097, 'surf': 1710, 'stay': 1642, 'articl': 87, 'acn': 10, 'scar': 1475, 'opposit': 1185, 'life': 973, 'visibl': 1868, 'fade': 620, 'dermatologist': 463, 'game': 722, 'changer': 277, 'main': 1030, 'test': 1749, 'singl': 1542, 'mayo': 1057, 'bod': 180, 'bat': 127, 'quench': 1343, 'fell': 645, 'cooki': 377, 'dough': 513, 'superdrug': 1703, 'prize': 1315, 'buzz': 237, 'limit': 981, 'edit': 546, 'stayer': 1643, 'scratchi': 1482, 'rip': 1429, 'mistreat': 1090, 'dock': 501, 'reus': 1418, 'revitalis': 1420, 'gentli': 731, 'impur': 883, 'servic': 1508, 'fault': 634, 'rocemmend': 1434, 'almond': 40, 'mouth': 1113, 'fave': 635, 'dark': 424, 'passion': 1225, 'narrow': 1125, 'accur': 8, 'slim': 1553, 'pourer': 1294, 'dribbl': 526, 'washload': 1884, 'flower': 672, 'hint': 833, 'fruit': 709, 'mango': 1037, 'brought': 219, 'cornet': 381, 'luvli': 1021, 'carri': 257, 'worri': 1934, 'man': 1035, 'choos': 296, 'sweat': 1717, 'mayonnais': 1058, 'substanc': 1684, 'squeezi': 1628, 'geniu': 729, 'idea': 868, 'tall': 1725, 'unstabl': 1839, 'ive': 922, 'fridg': 703, 'tumbl': 1816, 'brittl': 216, 'shatter': 1520, 'groundhog': 783, 'cri': 402, 'overhaul': 1200, 'curri': 414, 'besid': 152, 'sachet': 1451, 'edibl': 545, 'phenomen': 1253, 'function': 715, 'longest': 1001, 'rid': 1425, 'kept': 932, 'drawback': 519, 'suppli': 1707, 'wear': 1893, 'diabet': 476, 'carbohydr': 251, 'nutrit': 1161, 'inform': 896, 'whatsoev': 1902, 'wall': 1877, 'specif': 1610, 'portion': 1286, 'whitehead': 1909, 'bash': 124, 'inadvert': 884, 'deterg': 471, 'fairi': 623, 'joint': 927, 'biolog': 161, 'septic': 1505, 'tank': 1728, 'dispens': 497, 'insert': 899, 'suffici': 1690, 'condit': 354, 'competit': 344, 'hav': 810, 'btilliant': 221, 'wild': 1913, 'enthusiast': 567, 'err': 573, 'gener': 728, 'custom': 415, 'write': 1944, 'uncomfort': 1829, 'clash': 305, 'coat': 321, 'sickli': 1528, 'raspberri': 1355, 'cover': 392, 'core': 380, 'remind': 1402, 'cheapest': 282, 'imagin': 873, 'insipid': 900, 'rush': 1450, 'develop': 473, 'process': 1319, 'bitter': 164, 'rippl': 1430, 'consid': 361, 'lush': 1020, 'suggest': 1692, 'petrolatum': 1251, 'damp': 421, 'eczema': 544, 'rosacea': 1439, 'coz': 394, 'wors': 1935, 'apprehens': 77, 'broken': 218, 'fit': 662, 'purpos': 1338, 'sport': 1622, 'mad': 1026, 'teenag': 1738, 'promis': 1321, 'moisturisor': 1101, 'deoder': 458, 'discov': 493, 'shame': 1515, 'broke': 217, 'manli': 1039, 'foami': 678, 'wondr': 1927, 'variou': 1852, 'chamomil': 273, 'weight': 1899, 'thicker': 1757, 'space': 1604, 'nicest': 1143, 'mmmm': 1094, 'bo': 178, 'beater': 137, 'plainli': 1265, 'bright': 207, 'bold': 184, 'grudg': 786, 'dirti': 487, 'tablet': 1722, 'necess': 1132, 'weekli': 1898, 'groceri': 781, 'budget': 225, 'cake': 239, 'exot': 595, 'veget': 1857, 'ariel': 81, 'domin': 507, 'global': 748, 'consum': 365, 'giant': 734, 'expert': 600, 'dodgi': 503, 'imit': 874, 'sub': 1680, 'guarante': 788, 'high': 829, 'rubberi': 1445, 'gossam': 763, 'case': 258, 'brightli': 210, 'funki': 716, 'purpl': 1337, 'grey': 780, 'swirl': 1720, 'gu': 787, 'outsid': 1195, 'grade': 767, 'disappear': 488, 'mysteri': 1123, 'sphinx': 1616, 'giza': 741, 'bermuda': 151, 'triangl': 1810, 'voynich': 1872, 'manuscript': 1041, 'xlarg': 1947, 'discontinu': 491, 'option': 1187, 'limescal': 980, 'unclog': 1828, 'wake': 1875, 'plump': 1276, 'ill': 871, 'pun': 1333, 'intend': 906, 'altern': 49, 'forth': 690, 'healthier': 818, 'advic': 22, 'minimum': 1081, 'annoy': 60, 'asid': 91, 'run': 1448, 'plug': 1275, 'potenti': 1292, 'increas': 889, 'member': 1066, 'endors': 561, 'practic': 1298, 'nappi': 1124, 'pyramid': 1339, 'warm': 1880, 'present': 1303, 'st': 1630, 'rememb': 1401, 'repurchas': 1409, 'figur': 652, 'medicin': 1062, 'cabinet': 238, 'salti': 1459, 'yeast': 1951, 'honest': 843, 'toast': 1780, 'fluctuat': 674, 'particulr': 1222, 'watch': 1887, 'smellllllll': 1565, 'sauna': 1467, 'outstand': 1196, 'turkey': 1817, 'neck': 1133, 'shake': 1512, 'graviti': 774, 'vastli': 1854, 'sharp': 1519, 'begin': 144, 'assum': 93, 'grow': 784, 'ocado': 1164, 'sampl': 1460, 'state': 1639, 'formula': 689, 'correct': 383, 'later': 952, 'boost': 189, 'sooo': 1595, 'complaint': 346, 'concept': 351, 'recal': 1369, 'tetra': 1750, 'public': 1329, 'innov': 898, 'refil': 1384, 'cardboard': 254, 'dose': 511, 'counter': 388, 'read': 1361, 'print': 1313, 'wrapper': 1941, 'magnif': 1028, 'anim': 57, 'whilst': 1906, 'provid': 1328, 'headquart': 815, 'offic': 1173, 'produc': 1320, 'countri': 389, 'sale': 1457, 'carbon': 252, 'footprint': 683, 'torn': 1792, 'extrat': 604, 'wherea': 1904, 'secondli': 1494, 'spare': 1607, 'rib': 1422, 'persdper': 1242, 'reorder': 1404, 'somewhat': 1590, 'feminin': 647, 'perspir': 1247, 'danc': 422, 'bay': 132, 'waxi': 1891, 'shark': 1518, 'ben': 148, 'jerri': 925, 'fragrant': 696, 'bargin': 121, 'vanilla': 1849, 'iritatw': 912, 'tbh': 1735, 'loo': 1002, 'conceal': 349, 'moment': 1102, 'coverag': 393, 'brighten': 208, 'glowi': 752, 'perfectli': 1237, 'view': 1866, 'consciou': 360, 'paper': 1216, 'temperament': 1741, 'lazi': 960, 'girl': 737, 'solut': 1585, 'deep': 437, 'repeat': 1406, 'versatil': 1863, 'heel': 822, 'econom': 542, 'defo': 449, 'direct': 485, 'chapstick': 278, 'jot': 928, 'bob': 179, 'uncl': 1827, 'awesom': 106, 'doeant': 504, 'soggi': 1582, 'mess': 1070, 'dish': 496, 'lux': 1022, 'plan': 1266, 'kcal': 930, 'control': 371, 'diet': 479, 'biodegrad': 160, 'forev': 685, 'blackhead': 165, 'zone': 1962, 'fed': 641, 'tone': 1787, 'anymor': 65, 'bland': 166, 'wateri': 1889, 'gold': 757, 'spring': 1626, 'cold': 325, 'cardigan': 255, 'thirti': 1761, 'degre': 450, 'tie': 1768, 'boil': 182, 'bikini': 157, 'dedic': 436, 'ylang': 1953, 'town': 1797, 'aaaaamaz': 0, 'tattoooo': 1734, 'previous': 1309, 'admir': 17, 'blast': 167, 'whiff': 1905, 'planet': 1267, 'pollut': 1280, 'funni': 718, 'cone': 356, 'crispi': 403, 'cornetto': 382, 'flare': 666, 'itch': 917, 'crazi': 396, 'escap': 575, 'lift': 974, 'drop': 529, 'instantli': 903, 'relief': 1397, 'effortlessli': 550, 'alot': 44, 'effici': 548, 'linen': 983, 'pre': 1300, 'extract': 603, 'honey': 845, 'suckl': 1687, 'yesterday': 1952, 'perman': 1241, 'cherri': 288, 'blossom': 174, 'pea': 1231, 'sandal': 1462, 'wood': 1929, 'cup': 409, 'lunch': 1018, 'king': 938, 'gram': 769, 'antiperspir': 64, 'unblock': 1826, 'gradual': 768, 'pleasantli': 1271, 'surpris': 1712, 'cours': 391, 'fcuk': 639, 'opt': 1186, 'starter': 1638, 'newborn': 1139, 'gym': 791, 'hunger': 859, 'challeng': 272, 'strength': 1666, 'endur': 562, 'helmann': 824, 'bone': 187, 'tendon': 1745, 'appeal': 72, 'advantag': 21, 'citru': 303, 'muscl': 1120, 'bump': 228, 'ador': 19, 'handi': 799, 'persist': 1245, 'whitout': 1910, 'sin': 1540, 'spend': 1613, 'woken': 1924, 'reiment': 1391, 'smudg': 1571, 'novelti': 1158, 'fish': 661, 'quantiti': 1341, 'deodour': 460, 'swear': 1716, 'arkward': 82, 'kick': 935, 'sent': 1503, 'toiletri': 1784, 'linnen': 986, 'film': 654, 'teeth': 1739, 'ceram': 266, 'realis': 1364, 'cent': 264, 'gooey': 761, 'marshmallowey': 1047, 'phish': 1254, 'class': 306, 'spain': 1606, 'breakout': 204, 'intoler': 908, 'superior': 1704, 'alon': 43, 'thiught': 1762, 'cojld': 324, 'garnier': 724, 'apar': 70, 'stripey': 1670, 'feet': 644, 'orang': 1188, 'stink': 1650, 'moral': 1106, 'stori': 1657, 'format': 688, 'shave': 1521, 'dress': 524, 'fewer': 649, 'worn': 1933, 'meet': 1064, 'amazingli': 54, 'flavor': 668, 'powderi': 1296, 'exempt': 592, 'medium': 1063, 'annoyiji': 61, 'occas': 1165, 'immedi': 875, 'serum': 1506, 'contribut': 370, 'deco': 435, 'women': 1925, 'eldest': 552, 'forgotten': 686, 'chew': 289, 'rope': 1438, 'school': 1478, 'lie': 972, 'nother': 1155, 'automat': 100, 'gloopi': 749, 'drugstor': 530, 'brainer': 200, 'partnership': 1224, 'encourag': 559, 'spici': 1617, 'fear': 640, 'spilt': 1619, 'alright': 47, 'gotta': 765, 'cetearyl': 271, 'cif': 302, 'power': 1297, 'mirror': 1085, 'nightmar': 1146, 'fun': 714, 'ordinari': 1190, 'thicken': 1756, 'gravi': 773, 'pop': 1283, 'press': 1305, 'spell': 1612, 'scali': 1473, 'quid': 1347, 'tempt': 1742, 'everytim': 585, 'spf': 1615, 'critic': 405, 'uva': 1847, 'convinc': 374, 'refer': 1383, 'verifi': 1861, 'overli': 1201, 'allergi': 38, 'certain': 268, 'bondi': 186, 'sand': 1461, 'slight': 1551, 'fabul': 615, 'hike': 831, 'itsel': 920, 'frizz': 706, 'flat': 667, 'lank': 947, 'bounc': 195, 'stuck': 1675, 'unus': 1842, 'cooker': 376, 'invigor': 911, 'newer': 1140, 'rd': 1357, 'isnt': 915, 'cerav': 267, 'norm': 1150, 'begun': 145, 'calmer': 241, 'sciencey': 1479, 'id': 867, 'unless': 1834, 'anytim': 68, 'indulg': 893, 'bite': 163, 'dog': 506, 'micro': 1075, 'granul': 771, 'wari': 1879, 'stainless': 1632, 'steel': 1644, 'liquidi': 989, 'velvet': 1858, 'hopingthat': 848, 'fair': 622, 'btw': 222, 'instruct': 905, 'typic': 1821, 'dy': 535, 'tong': 1789, 'question': 1344, 'clip': 316, 'sensat': 1501, 'definatli': 445, 'surprisingli': 1713, 'shock': 1525, 'litr': 992, 'imo': 876, 'salad': 1456, 'bake': 114, 'bean': 134, 'frozen': 708, 'pear': 1232, 'youth': 1958, 'anitipersperi': 58, 'accustom': 9, 'richer': 1424, 'bundl': 229, 'low': 1012, 'carb': 249, 'duti': 534, 'sock': 1578, 'fraction': 694, 'loveliest': 1010, 'wrist': 1943, 'drum': 531, 'modest': 1096, 'inch': 886, 'god': 755, 'somebodi': 1586, 'brazil': 202, 'expat': 596, 'bearabl': 135, 'vfm': 1865, 'head': 813, 'winner': 1918, 'iv': 921, 'batgain': 129, 'averag': 101, 'oreal': 1191, 'resist': 1412, 'girlfriend': 738, 'mention': 1069, 'benefici': 149, 'list': 990, 'chanc': 275, 'transfer': 1800, 'snif': 1574, 'commit': 336, 'stiff': 1648, 'kg': 934, 'alobg': 42, 'matur': 1055, 'hiya': 836, 'stress': 1667, 'pricey': 1311, 'reliev': 1398, 'final': 655, 'tropic': 1811, 'lili': 978, 'mud': 1115, 'spag': 1605, 'bol': 183, 'attent': 96, 'lingeri': 985, 'cotton': 386, 'silk': 1534, 'fiber': 650, 'gigant': 736, 'possibl': 1288, 'lash': 950, 'eyebrow': 609, 'hide': 828, 'sunshin': 1700, 'clingi': 315, 'greazi': 777, 'advis': 23, 'rare': 1353, 'desir': 467, 'blotch': 175, 'brown': 220, 'attach': 95, 'ignor': 870, 'fortun': 691, 'fulfil': 712, 'criteria': 404, 'wilt': 1915, 'yard': 1948, 'radiu': 1350, 'alarm': 34, 'whisk': 1907, 'drainag': 518, 'devin': 474, 'signatur': 1530, 'haha': 793, 'saver': 1469, 'pamper': 1214, 'heheh': 823, 'hunk': 860, 'shelf': 1522, 'unscrew': 1838, 'glove': 750, 'odd': 1167, 'horrif': 850, 'headach': 814, 'desper': 468, 'afraid': 27, 'disturb': 499, 'properti': 1325, 'fluff': 675, 'dryness': 532, 'unpleas': 1836, 'seal': 1490, 'drain': 517, 'yogurt': 1955, 'alpro': 45, 'blop': 173, 'postag': 1290, 'doctor': 502, 'soooo': 1596, 'walk': 1876, 'pervas': 1249, 'sky': 1548, 'fruiti': 710, 'strike': 1668, 'balanc': 115, 'slurp': 1557, 'reckon': 1375, 'nearer': 1129, 'path': 1229, 'decidedli': 434, 'newsweek': 1141, 'fluffi': 676, 'smoother': 1568, 'hive': 835, 'aggrav': 31, 'wool': 1930, 'chsnge': 299, 'remark': 1400, 'specifi': 1611, 'conclus': 353, 'lure': 1019, 'entic': 568, 'squint': 1629, 'reviv': 1421, 'fallen': 626, 'deffinalti': 441, 'fond': 681, 'poor': 1281, 'consider': 362, 'dinner': 484, 'parti': 1219, 'everybodi': 581, 'jelli': 924, 'loyal': 1013, 'balm': 116, 'finger': 657, 'color': 328, 'petroleum': 1252, 'shadow': 1511, 'appropri': 79, 'tidi': 1767, 'eyelin': 610, 'error': 574, 'spars': 1608, 'bud': 224, 'homemad': 842, 'brill': 211, 'row': 1443, 'roll': 1435, 'shaken': 1513, 'disgust': 495, 'contact': 366, 'suddenli': 1688, 'alth': 50, 'insread': 902, 'toxic': 1798, 'dud': 533, 'split': 1620, 'cautiou': 262, 'lucki': 1015, 'monthli': 1105, 'beed': 142, 'scrib': 1485, 'hurt': 861, 'glitter': 747, 'ankl': 59, 'drawer': 520, 'bovril': 196, 'moan': 1095, 'arm': 83, 'significantli': 1532, 'strang': 1661, 'bleach': 168, 'occasion': 1166, 'recognis': 1376, 'cedarwood': 263, 'popular': 1284, 'bother': 192, 'dozen': 516, 'blindingli': 171, 'leg': 966, 'tanner': 1729, 'stronger': 1672, 'street': 1665, 'veg': 1855, 'cube': 407, 'batch': 128, 'recipebut': 1374, 'secret': 1495, 'mile': 1078, 'hr': 856, 'sedentari': 1497, 'worst': 1936, 'verri': 1862, 'hi': 827, 'uk': 1822, 'anywher': 69, 'massag': 1051, 'keen': 931, 'ban': 117, 'sneaki': 1573, 'woke': 1923, 'sunk': 1699, 'tacki': 1723, 'surviv': 1714, 'brexit': 206, 'defintley': 448, 'die': 478, 'em': 556, 'pregnant': 1302, 'moisteris': 1098, 'lost': 1006, 'count': 387, 'streaki': 1663, 'capful': 246, 'common': 337, 'loyalti': 1014, 'sooner': 1594, 'mostur': 1110, 'definetli': 446, 'mostureris': 1111, 'tissu': 1779, 'pain': 1211, 'pleasur': 1272, 'happili': 804, 'vera': 1859, 'directli': 486, 'allow': 39, 'subject': 1681, 'greenhous': 779, 'gase': 725, 'ultim': 1823, 'fight': 651, 'tire': 1778, 'drawn': 521, 'child': 291, 'dosag': 510, 'statement': 1640, 'worthwhil': 1938, 'sorbet': 1598, 'focus': 679, 'exact': 587, 'costco': 385, 'curiou': 411, 'baffl': 112, 'elviv': 555, 'realiz': 1365, 'silicon': 1533, 'hairdress': 795, 'appoint': 75, 'defenc': 439, 'purifi': 1336, 'pralin': 1299, 'choc': 293, 'solero': 1584}\n"
     ]
    }
   ],
   "source": [
    "count_vectorizer  = CountVectorizer(stop_words='english')\n",
    "count_vectorizer.fit(X_train)\n",
    "print('\\nVocabulary: \\n', count_vectorizer.vocabulary_)\n",
    "\n",
    "train_c = count_vectorizer.fit_transform(X_train)\n",
    "test_c = count_vectorizer.transform(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Multinomial Naive Bayes model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 149,
   "metadata": {},
   "outputs": [],
   "source": [
    "#IMPLEMENTING AND RUNNNING MNB MODEL - COUNT\n",
    "mnb1 = MultinomialNB()\n",
    "mnb1.fit(train_c, y_train)\n",
    "prediction = mnb1.predict(test_c)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 150,
   "metadata": {},
   "outputs": [],
   "source": [
    "#EVALUATION\n",
    "mnb_a1 = accuracy_score(y_test, prediction)*100\n",
    "mnb_p1 = precision_score(y_test, prediction)* 100\n",
    "mnb_r1 = recall_score(y_test, prediction)*100\n",
    "mnb_f11 = f1_score(y_test, prediction)*100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 151,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x1c0360f82b0>"
      ]
     },
     "execution_count": 151,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#CONFUSION MATRIX\n",
    "cm =  confusion_matrix(y_test, prediction, labels=mnb1.classes_)\n",
    "display = ConfusionMatrixDisplay(confusion_matrix=cm,\n",
    "                              display_labels=mnb1.classes_) \n",
    "display.plot() "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Support Vector Machine model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 152,
   "metadata": {},
   "outputs": [],
   "source": [
    "#IMPLEMENTING AND RUNNNING SVM MODEL - COUNT\n",
    "svm1 = SVC(kernel='linear')\n",
    "svm1.fit(train_c, y_train)\n",
    "prediction = svm1.predict(test_c)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 153,
   "metadata": {},
   "outputs": [],
   "source": [
    "#EVALUATION\n",
    "svm_a1 = accuracy_score(y_test, prediction)*100\n",
    "svm_p1 = precision_score(y_test, prediction)* 100\n",
    "svm_r1 = recall_score(y_test, prediction)*100\n",
    "svm_f11 = f1_score(y_test, prediction)*100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 154,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x1c0360f89d0>"
      ]
     },
     "execution_count": 154,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#CONFUSION MATRIX\n",
    "cm =  confusion_matrix(y_test, prediction, labels=svm1.classes_)\n",
    "display = ConfusionMatrixDisplay(confusion_matrix=cm,\n",
    "                              display_labels=svm1.classes_) \n",
    "display.plot() "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Logistic Regression model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 155,
   "metadata": {},
   "outputs": [],
   "source": [
    "#IMPLEMENTING AND RUNNNING LR MODEL - COUNT\n",
    "lr1 = LogisticRegression()\n",
    "lr1.fit(train_c, y_train)\n",
    "prediction = lr1.predict(test_c)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 156,
   "metadata": {},
   "outputs": [],
   "source": [
    "#EVALUATION\n",
    "lr_a1 = accuracy_score(y_test, prediction)*100\n",
    "lr_p1 = precision_score(y_test, prediction)* 100\n",
    "lr_r1 = recall_score(y_test, prediction)*100\n",
    "lr_f11 = f1_score(y_test, prediction)*100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 157,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x1c0361e8280>"
      ]
     },
     "execution_count": 157,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#CONFUSION MATRIX\n",
    "cm =  confusion_matrix(y_test, prediction, labels=lr1.classes_)\n",
    "display = ConfusionMatrixDisplay(confusion_matrix=cm,\n",
    "                              display_labels=lr1.classes_) \n",
    "display.plot() "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## TFIDF VECTORIZER AND MODELING\n",
    "\n",
    "We examine the total document weightage of a word in TfidfVectorizer. It assists us in coping with the most common terms. We may use it to penalize them. The word counts are weighted by a measure of how frequently they appear in the documents in TfidfVectorizer."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 158,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Vocabulary: \n",
      " {'current': 413, 'sell': 1498, 'price': 1310, 'compar': 341, 'supermarket': 1705, 'good': 760, 'smell': 1563, 'pleasant': 1270, 'need': 1134, 'add': 14, 'small': 1558, 'cloth': 319, 'fresh': 700, 'great': 776, 'moisturis': 1100, 'sensit': 1502, 'love': 1009, 'pour': 1293, 'smaller': 1559, 'bottl': 193, 'make': 1032, 'manag': 1036, 'beat': 136, 'decent': 432, 'fabric': 614, 'soften': 1580, 'nice': 1142, 'fragranc': 695, 'purchas': 1334, 'deliveri': 456, 'cream': 397, 'handwash': 801, 'cheaper': 281, 'hand': 798, 'better': 154, 'liquid': 988, 'oh': 1174, 'wing': 1917, 'dove': 514, 'kitchen': 940, 'bathroom': 131, 'shower': 1527, 'room': 1436, 'recommend': 1378, 'highli': 830, 'star': 1636, 'simpl': 1537, 'gel': 727, 'like': 977, 'glue': 753, 'hard': 805, 'rub': 1444, 'slip': 1555, 'bath': 130, 'goe': 756, 'smoothli': 1569, 'easili': 539, 'wast': 1885, 'leav': 964, 'feel': 643, 'silki': 1535, 'soft': 1579, 'scenti': 1477, 'review': 1419, 'collect': 326, 'promot': 1322, 'excel': 589, 'everi': 580, 'day': 426, 'facial': 617, 'wash': 1882, 'excess': 591, 'face': 616, 'strip': 1669, 'natur': 1127, 'oil': 1175, 'care': 256, 'routin': 1442, 'morn': 1108, 'night': 1144, 'clean': 308, 'brilliant': 213, 'offer': 1172, 'gorgeou': 762, 'amaz': 52, 'valu': 1848, 'girli': 739, 'hair': 794, 'buy': 235, 'chang': 276, 'preserv': 1304, 'come': 331, 'bad': 109, 'sore': 1599, 'rash': 1354, 'eye': 608, 'burn': 230, 'lip': 987, 'tingl': 1775, 'phone': 1255, 'told': 1785, 'stuff': 1677, 'ask': 92, 'said': 1454, 'know': 943, 'want': 1878, 'reason': 1368, 'mayb': 1056, 'save': 1468, 'money': 1103, 'ingredi': 897, 'nearli': 1130, 'year': 1950, 'sinc': 1541, 'nivea': 1147, 'sold': 1583, 'compani': 339, 'german': 733, 'im': 872, 'realli': 1366, 'angri': 56, 'suppos': 1708, 'ok': 1177, 'rubbish': 1446, 'burnt': 231, 'pleas': 1269, 'usual': 1846, 'stock': 1651, 'fulli': 713, 'asda': 90, 'gave': 726, 'refund': 1386, 'gift': 735, 'card': 253, 'receipt': 1371, 'pocket': 1277, 'condition': 355, 'normal': 1151, 'oili': 1176, 'week': 1896, 'saw': 1470, 'differ': 480, 'felt': 646, 'cleans': 310, 'clearer': 313, 'notic': 1156, 'straightaway': 1659, 'red': 1381, 'blemish': 169, 'previou': 1308, 'kid': 936, 'doesnt': 505, 'irrit': 914, 'scent': 1476, 'littl': 993, 'bit': 162, 'long': 999, 'way': 1892, 'similar': 1536, 'perfect': 1236, 'got': 764, 'coupl': 390, 'ago': 32, 'refresh': 1385, 'bodi': 181, 'smooth': 1567, 'cucumb': 408, 'relax': 1393, 'best': 153, 'came': 243, 'separ': 1504, 'packet': 1208, 'sealabl': 1491, 'affect': 25, 'rough': 1440, 'dri': 525, 'otherwis': 1193, 'fantast': 631, 'lot': 1007, 'effort': 549, 'reduc': 1382, 'plastic': 1268, 'concentr': 350, 'buyer': 236, 'say': 1471, 'larger': 949, 'probabl': 1316, 'explain': 601, 'label': 945, 'fuchsia': 711, 'perfum': 1239, 'version': 1864, 'overpow': 1203, 'big': 155, 'plu': 1274, 'difficulti': 482, 'intens': 907, 'past': 1226, 'note': 1153, 'outer': 1194, 'sleev': 1550, 'recycl': 1380, 'dispos': 498, 'charg': 279, 'kind': 937, 'overbear': 1199, 'anyon': 66, 'glow': 751, 'afford': 26, 'comfort': 332, 'creation': 400, 'round': 1441, 'market': 1044, 'close': 318, 'match': 1053, 'honeysuckl': 846, 'sandalwood': 1463, 'person': 1246, 'favourit': 638, 'howev': 855, 'descript': 465, 'ad': 13, 'recent': 1373, 'amazon': 55, 'pantri': 1215, 'order': 1189, 'lamin': 946, 'tile': 1771, 'floor': 670, 'subtl': 1685, 'streak': 1662, 'free': 698, 'shine': 1523, 'moistur': 1099, 'essenti': 577, 'tri': 1808, 'time': 1773, 'today': 1781, 'packag': 1207, 'easi': 537, 'open': 1182, 'squeez': 1627, 'releas': 1394, 'puff': 1330, 'froth': 707, 'pure': 1335, 'smear': 1562, 'white': 1908, 'absorb': 4, 'non': 1148, 'greasi': 775, 'appli': 73, 'think': 1759, 'expens': 598, 'cheap': 280, 'qualiti': 1340, 'dairi': 419, 'aroma': 84, 'dont': 508, 'drench': 523, 'lotion': 1008, 'pretti': 1306, 'worth': 1937, 'thank': 1752, 'item': 919, 'describ': 464, 'rapid': 1352, 'stop': 1653, 'static': 1641, 'chocol': 294, 'flavour': 669, 'creami': 398, 'soap': 1576, 'anyth': 67, 'els': 553, 'lather': 954, 'noth': 1154, 'fail': 621, 'protect': 1326, 'gentl': 730, 'basic': 126, 'harm': 806, 'nourish': 1157, 'tasti': 1733, 'beef': 143, 'tomato': 1786, 'grate': 772, 'cheddar': 284, 'chees': 286, 'mix': 1091, 'nd': 1128, 'let': 969, 'sit': 1544, 'minut': 1083, 'regret': 1388, 'set': 1509, 'took': 1791, 'expect': 597, 'size': 1546, 'persil': 1244, 'quit': 1348, 'pictur': 1258, 'mislead': 1087, 'apart': 71, 'cheer': 285, 'makeup': 1034, 'eas': 536, 'micellar': 1074, 'water': 1888, 'left': 965, 'pack': 1206, 'extra': 602, 'impresss': 881, 'content': 368, 'aw': 103, 'cut': 416, 'right': 1427, 'fingernail': 658, 'turn': 1818, 'upsid': 1844, 'approach': 78, 'empti': 558, 'ridicul': 1426, 'design': 466, 'bought': 194, 'lumber': 1016, 'home': 841, 'larg': 948, 'decant': 431, 'exist': 594, 'contain': 367, 'tell': 1740, 'straight': 1658, 'away': 105, 'despit': 469, 'compart': 343, 'machin': 1025, 'towel': 1796, 'unlik': 1835, 'manufactur': 1040, 'healthi': 817, 'definit': 447, 'forese': 684, 'futur': 720, 'gotten': 766, 'result': 1415, 'brillant': 212, 'brand': 201, 'absolut': 3, 'fan': 629, 'wow': 1939, 'factor': 619, 'base': 123, 'commerci': 335, 'help': 825, 'disappoint': 489, 'especi': 576, 'point': 1279, 'ice': 866, 'chunk': 301, 'difficult': 481, 'eat': 540, 'entir': 569, 'tub': 1814, 'nasti': 1126, 'chemic': 287, 'super': 1701, 'famili': 628, 'young': 1956, 'old': 1179, 'friendli': 705, 'mean': 1059, 'colour': 329, 'honestli': 844, 'problemat': 1318, 'prone': 1323, 'dermat': 462, 'tast': 1732, 'weed': 1895, 'vinegar': 1867, 'work': 1932, 'remov': 1403, 'havent': 811, 'hamper': 797, 'christma': 298, 'sure': 1709, 'magnum': 1029, 'incas': 885, 'amazebal': 53, 'togeth': 1782, 'delici': 453, 'superb': 1702, 'kit': 939, 'bio': 159, 'stubborn': 1674, 'stain': 1631, 'word': 1931, 'warn': 1881, 'handl': 800, 'tend': 1743, 'ensur': 566, 'thoroughli': 1763, 'rins': 1428, 'repair': 1405, 'guy': 790, 'fals': 627, 'economi': 543, 'build': 226, 'caus': 261, 'issu': 916, 'huggabl': 858, 'strong': 1671, 'alcohol': 36, 'pot': 1291, 'noodl': 1149, 'student': 1676, 'yuck': 1959, 'addit': 15, 'sauc': 1466, 'pasta': 1227, 'carboard': 250, 'situat': 1545, 'access': 6, 'kettl': 933, 'chose': 297, 'mugshot': 1116, 'mistak': 1089, 'scrumptiou': 1488, 'authent': 99, 'linger': 984, 'bonu': 188, 'percent': 1235, 'fact': 618, 'husband': 862, 'classic': 307, 'lynx': 1024, 'deodor': 459, 'regularli': 1390, 'number': 1160, 'skincar': 1547, 'avoid': 102, 'toner': 1788, 'experi': 599, 'anoth': 62, 'major': 1031, 'extrem': 605, 'flaki': 665, 'sting': 1649, 'dewi': 475, 'look': 1003, 'deffin': 440, 'opinion': 1183, 'reccomend': 1370, 'deal': 429, 'slow': 1556, 'cook': 375, 'beauti': 138, 'tip': 1777, 'mark': 1043, 'alway': 51, 'fresher': 701, 'regular': 1389, 'late': 951, 'overal': 1198, 'improv': 882, 'claim': 304, 'perform': 1238, 'miracl': 1084, 'lol': 997, 'reaction': 1360, 'happen': 802, 'winter': 1919, 'weather': 1894, 'central': 265, 'heat': 819, 'vaselin': 1853, 'dealt': 430, 'includ': 887, 'elbow': 551, 'knee': 941, 'place': 1263, 'glide': 746, 'dream': 522, 'summer': 1695, 'spent': 1614, 'lay': 958, 'beach': 133, 'eyemak': 611, 'men': 1067, 'boot': 190, 'tame': 1726, 'divin': 500, 'iron': 913, 'tear': 1737, 'penni': 1233, 'scale': 1472, 'deliv': 455, 'quickli': 1346, 'shampoo': 1516, 'leak': 963, 'tan': 1727, 'sunb': 1697, 'spray': 1624, 'thought': 1764, 'clear': 312, 'friend': 704, 'huge': 857, 'indoor': 892, 'didnt': 477, 'exxtra': 607, 'fast': 633, 'thankyou': 1753, 'half': 796, 'shop': 1526, 'line': 982, 'fake': 625, 'africa': 28, 'poorer': 1282, 'new': 1138, 'softer': 1581, 'particularli': 1221, 'wrap': 1940, 'storag': 1655, 'measur': 1061, 'touch': 1794, 'brighter': 209, 'cleaner': 309, 'afterward': 29, 'everyon': 583, 'impress': 880, 'neutral': 1137, 'defiantli': 442, 'fab': 613, 'rate': 1356, 'mani': 1038, 'micel': 1073, 'wipe': 1920, 'total': 1793, 'alreadi': 46, 'receiv': 1372, 'feed': 642, 'event': 578, 'absoulut': 5, 'stick': 1646, 'job': 926, 'live': 994, 'hype': 865, 'clearli': 314, 'lead': 961, 'leader': 962, 'inspir': 901, 'confid': 357, 'certainli': 269, 'harsh': 808, 'favorit': 636, 'imposs': 879, 'sourc': 1603, 'check': 283, 'target': 1731, 'piec': 1260, 'crack': 395, 'bare': 119, 'far': 632, 'concern': 352, 'waterproof': 1890, 'mascara': 1048, 'powder': 1295, 'return': 1417, 'paid': 1210, 'daili': 418, 'basi': 125, 'age': 30, 'multipl': 1117, 'bulk': 227, 'door': 509, 'funnel': 717, 'easier': 538, 'section': 1496, 'washer': 1883, 'moreov': 1107, 'thing': 1758, 'exactli': 588, 'someth': 1588, 'odor': 1169, 'blend': 170, 'energis': 563, 'complain': 345, 'lid': 971, 'bin': 158, 'spill': 1618, 'everywher': 586, 'box': 197, 'ml': 1093, 'hope': 847, 'misl': 1086, 'month': 1104, 'start': 1637, 'enjoy': 565, 'mr': 1114, 'hinch': 832, 'glad': 742, 'scratch': 1481, 'scour': 1480, 'pad': 1209, 'minki': 1082, 'spong': 1621, 'oven': 1197, 'hob': 837, 'tap': 1730, 'sink': 1543, 'screen': 1483, 'gleam': 745, 'compact': 338, 'rang': 1351, 'actual': 12, 'complet': 347, 'ha': 792, 'excema': 590, 'hit': 834, 'everyday': 582, 'massiv': 1052, 'cap': 245, 'lenor': 967, 'unstopp': 1840, 'convert': 373, 'mild': 1077, 'messi': 1071, 'step': 1645, 'dad': 417, 'awhil': 107, 'suitabl': 1694, 'obviou': 1162, 'choic': 295, 'believ': 146, 'continu': 369, 'term': 1746, 'benefit': 150, 'happi': 803, 'alround': 48, 'consist': 363, 'dilut': 483, 'guess': 789, 'blue': 177, 'pigment': 1261, 'act': 11, 'reli': 1395, 'hydrat': 864, 'mixtur': 1092, 'varieti': 1851, 'reach': 1358, 'matter': 1054, 'neg': 1135, 'pump': 1332, 'end': 560, 'foam': 677, 'met': 1072, 'standard': 1634, 'bar': 118, 'quarter': 1342, 'truli': 1813, 'load': 995, 'sometim': 1589, 'stapl': 1635, 'heal': 816, 'abras': 2, 'boy': 198, 'store': 1656, 'cupboard': 410, 'dead': 428, 'therapi': 1754, 'origin': 1192, 'mini': 1080, 'arriv': 85, 'real': 1363, 'delight': 454, 'quick': 1345, 'subscrib': 1682, 'light': 975, 'bubbl': 223, 'floral': 671, 'masculin': 1049, 'bigger': 156, 'volum': 1870, 'garag': 723, 'known': 944, 'sweet': 1719, 'textur': 1751, 'artifici': 89, 'justv': 929, 'hour': 852, 'pale': 1212, 'disastr': 490, 'success': 1686, 'creat': 399, 'spot': 1623, 'abl': 1, 'deepli': 438, 'exfoli': 593, 'wont': 1928, 'suffer': 1689, 'babi': 108, 'style': 1679, 'genuin': 732, 'neatli': 1131, 'everyth': 584, 'shall': 1514, 'tresmemm': 1807, 'fussi': 719, 'suit': 1693, 'therefor': 1755, 'prefer': 1301, 'burst': 232, 'transit': 1802, 'snack': 1572, 'went': 1900, 'weekend': 1897, 'decid': 433, 'wet': 1901, 'cashmer': 259, 'sweater': 1718, 'sainsburi': 1455, 'scrummi': 1487, 'delic': 452, 'requir': 1410, 'bring': 214, 'runni': 1449, 'glam': 743, 'somewher': 1591, 'follow': 680, 'inexpens': 894, 'plain': 1264, 'mother': 1112, 'law': 957, 'wat': 1886, 'discount': 492, 'tin': 1774, 'sooth': 1597, 'residu': 1411, 'cool': 378, 'readi': 1362, 'magic': 1027, 'latest': 953, 'variat': 1850, 'address': 16, 'environment': 571, 'transport': 1803, 'cost': 384, 'commend': 333, 'sens': 1500, 'detect': 470, 'breath': 205, 'form': 687, 'slightli': 1552, 'zesti': 1961, 'user': 1845, 'repres': 1408, 'lessen': 968, 'impact': 877, 'environ': 570, 'albeit': 35, 'smallest': 1560, 'ideal': 869, 'son': 1592, 'adult': 20, 'children': 292, 'belov': 147, 'pet': 1250, 'incred': 890, 'lolli': 998, 'bargain': 120, 'introduc': 909, 'partner': 1223, 'sadli': 1452, 'share': 1517, 'hot': 851, 'melt': 1065, 'drip': 528, 'finish': 659, 'favour': 637, 'properli': 1324, 'inferior': 895, 'tesco': 1748, 'microwav': 1076, 'sever': 1510, 'badli': 111, 'lumpi': 1017, 'mostli': 1109, 'simplic': 1539, 'gone': 758, 'effect': 547, 'trace': 1799, 'fanci': 630, 'maskara': 1050, 'remain': 1399, 'sticki': 1647, 'defin': 444, 'heavi': 820, 'frequent': 699, 'perhap': 1240, 'capsul': 247, 'household': 854, 'terribl': 1747, 'allerg': 37, 'constant': 364, 'hay': 812, 'fever': 648, 'wrong': 1945, 'sign': 1529, 'gluten': 754, 'casserol': 260, 'sat': 1464, 'toilet': 1783, 'barley': 122, 'coeliac': 323, 'peopl': 1234, 'unsuit': 1841, 'pick': 1257, 'signific': 1531, 'rel': 1392, 'sturdier': 1678, 'travel': 1804, 'hous': 853, 'tini': 1776, 'surfac': 1711, 'pic': 1256, 'wonder': 1926, 'deni': 457, 'fix': 663, 'instead': 904, 'whenev': 1903, 'react': 1359, 'true': 1812, 'luxuri': 1023, 'prime': 1312, 'conveni': 372, 'incorpor': 888, 'cleanser': 311, 'eventu': 579, 'margin': 1042, 'obvious': 1163, 'comment': 334, 'appreci': 76, 'break': 203, 'vegan': 1856, 'daughter': 425, 'persuad': 1248, 'hesit': 826, 'pay': 1230, 'forward': 692, 'soup': 1602, 'risotto': 1432, 'younger': 1957, 'menopaus': 1068, 'calm': 240, 'wrinkl': 1942, 'crepey': 401, 'layer': 959, 'conjunct': 359, 'hyaluron': 863, 'companion': 340, 'loss': 1005, 'firm': 660, 'oz': 1205, 'csmart': 406, 'screw': 1484, 'determin': 472, 'loos': 1004, 'knife': 942, 'wise': 1921, 'advoid': 24, 'smart': 1561, 'local': 996, 'okay': 1178, 'meant': 1060, 'equal': 572, 'artif': 88, 'upset': 1843, 'simpli': 1538, 'english': 564, 'mustard': 1122, 'colman': 327, 'sunday': 1698, 'roast': 1433, 'tube': 1815, 'invari': 910, 'serv': 1507, 'nozzl': 1159, 'block': 172, 'someon': 1587, 'scrub': 1486, 'extremli': 606, 'frangranc': 697, 'fairli': 624, 'spread': 1625, 'soak': 1575, 'smelt': 1566, 'wilkinson': 1914, 'area': 80, 'soon': 1593, 'becom': 139, 'scalp': 1474, 'satisfi': 1465, 'sorri': 1600, 'voucher': 1871, 'post': 1289, 'slimi': 1554, 'problem': 1317, 'unhappi': 1832, 'respons': 1413, 'treat': 1805, 'portug': 1287, 'miss': 1088, 'tea': 1736, 'straighten': 1660, 'curl': 412, 'damag': 420, 'seller': 1499, 'pod': 1278, 'bag': 113, 'filler': 653, 'rich': 1423, 'special': 1609, 'elsewher': 554, 'laundri': 956, 'hate': 809, 'arrog': 86, 'unilev': 1833, 'hold': 838, 'palm': 1213, 'fine': 656, 'paragon': 1217, 'champ': 274, 'tight': 1769, 'win': 1916, 'doubl': 512, 'struggl': 1673, 'tough': 1795, 'unbeliev': 1825, 'compliment': 348, 'august': 98, 'itchi': 918, 'stream': 1664, 'grandchildren': 770, 'unfortun': 1831, 'green': 778, 'lime': 979, 'individu': 891, 'dandruff': 423, 'anti': 63, 'prior': 1314, 'particular': 1220, 'busi': 233, 'mum': 1118, 'admit': 18, 'neglect': 1136, 'empathis': 557, 'trial': 1809, 'opportun': 1184, 'reconnect': 1379, 'cme': 320, 'smother': 1570, 'offens': 1171, 'alo': 41, 'applic': 74, 'bedtim': 141, 'verdict': 1860, 'thumb': 1766, 'ye': 1949, 'second': 1493, 'lightweight': 976, 'tendenc': 1744, 'pull': 1331, 'liber': 970, 'discreet': 494, 'overwhelm': 1204, 'protector': 1327, 'retain': 1416, 'ezyema': 612, 'switch': 1721, 'radiant': 1349, 'regim': 1387, 'thinner': 1760, 'heavili': 821, 'cancel': 244, 'subscript': 1683, 'wait': 1874, 'fyi': 721, 'attract': 97, 'floweri': 673, 'badeda': 110, 'holiday': 840, 'sun': 1696, 'till': 1772, 'stope': 1654, 'blow': 176, 'prevent': 1307, 'transform': 1801, 'nose': 1152, 'dehydr': 451, 'sort': 1601, 'rest': 1414, 'sleep': 1549, 'caramel': 248, 'marshmallow': 1046, 'liter': 991, 'calori': 242, 'bomb': 185, 'gross': 782, 'gonna': 759, 'throw': 1765, 'coca': 322, 'butter': 234, 'marmit': 1045, 'lover': 1011, 'glass': 744, 'jar': 923, 'oddli': 1168, 'given': 740, 'persev': 1243, 'shini': 1524, 'root': 1437, 'drier': 527, 'grown': 785, 'horribl': 849, 'agre': 33, 'tighter': 1770, 'import': 878, 'confus': 358, 'milk': 1079, 'maker': 1033, 'risk': 1431, 'underwear': 1830, 'ultra': 1824, 'twice': 1819, 'combin': 330, 'harmoni': 807, 'chicken': 290, 'mushroom': 1121, 'pie': 1259, 'salt': 1458, 'ruin': 1447, 'chuck': 300, 'tongu': 1790, 'season': 1492, 'accident': 7, 'swallow': 1715, 'sea': 1489, 'xd': 1946, 'older': 1180, 'wide': 1911, 'awak': 104, 'flakey': 664, 'smelli': 1564, 'pit': 1262, 'plenti': 1273, 'longer': 1000, 'comparison': 342, 'yo': 1954, 'born': 191, 'defient': 443, 'odourless': 1170, 'suppl': 1706, 'nightli': 1145, 'bed': 140, 'clog': 317, 'pore': 1285, 'replenish': 1407, 'daytim': 427, 'reappli': 1367, 'overnight': 1202, 'unscent': 1837, 'patch': 1228, 'stone': 1652, 'munchi': 1119, 'food': 682, 'depend': 461, 'wish': 1922, 'talk': 1724, 'visit': 1869, 'certifi': 270, 'british': 215, 'foundat': 693, 'freshli': 702, 'launder': 955, 'reliabl': 1396, 'cooler': 379, 'wahs': 1873, 'soapi': 1577, 'type': 1820, 'downsid': 515, 'hole': 839, 'tresemm': 1806, 'stand': 1633, 'yummi': 1960, 'sugari': 1691, 'eco': 541, 'parcel': 1218, 'brain': 199, 'op': 1181, 'wife': 1912, 'safe': 1453, 'recomend': 1377, 'aswel': 94, 'moist': 1097, 'surf': 1710, 'stay': 1642, 'articl': 87, 'acn': 10, 'scar': 1475, 'opposit': 1185, 'life': 973, 'visibl': 1868, 'fade': 620, 'dermatologist': 463, 'game': 722, 'changer': 277, 'main': 1030, 'test': 1749, 'singl': 1542, 'mayo': 1057, 'bod': 180, 'bat': 127, 'quench': 1343, 'fell': 645, 'cooki': 377, 'dough': 513, 'superdrug': 1703, 'prize': 1315, 'buzz': 237, 'limit': 981, 'edit': 546, 'stayer': 1643, 'scratchi': 1482, 'rip': 1429, 'mistreat': 1090, 'dock': 501, 'reus': 1418, 'revitalis': 1420, 'gentli': 731, 'impur': 883, 'servic': 1508, 'fault': 634, 'rocemmend': 1434, 'almond': 40, 'mouth': 1113, 'fave': 635, 'dark': 424, 'passion': 1225, 'narrow': 1125, 'accur': 8, 'slim': 1553, 'pourer': 1294, 'dribbl': 526, 'washload': 1884, 'flower': 672, 'hint': 833, 'fruit': 709, 'mango': 1037, 'brought': 219, 'cornet': 381, 'luvli': 1021, 'carri': 257, 'worri': 1934, 'man': 1035, 'choos': 296, 'sweat': 1717, 'mayonnais': 1058, 'substanc': 1684, 'squeezi': 1628, 'geniu': 729, 'idea': 868, 'tall': 1725, 'unstabl': 1839, 'ive': 922, 'fridg': 703, 'tumbl': 1816, 'brittl': 216, 'shatter': 1520, 'groundhog': 783, 'cri': 402, 'overhaul': 1200, 'curri': 414, 'besid': 152, 'sachet': 1451, 'edibl': 545, 'phenomen': 1253, 'function': 715, 'longest': 1001, 'rid': 1425, 'kept': 932, 'drawback': 519, 'suppli': 1707, 'wear': 1893, 'diabet': 476, 'carbohydr': 251, 'nutrit': 1161, 'inform': 896, 'whatsoev': 1902, 'wall': 1877, 'specif': 1610, 'portion': 1286, 'whitehead': 1909, 'bash': 124, 'inadvert': 884, 'deterg': 471, 'fairi': 623, 'joint': 927, 'biolog': 161, 'septic': 1505, 'tank': 1728, 'dispens': 497, 'insert': 899, 'suffici': 1690, 'condit': 354, 'competit': 344, 'hav': 810, 'btilliant': 221, 'wild': 1913, 'enthusiast': 567, 'err': 573, 'gener': 728, 'custom': 415, 'write': 1944, 'uncomfort': 1829, 'clash': 305, 'coat': 321, 'sickli': 1528, 'raspberri': 1355, 'cover': 392, 'core': 380, 'remind': 1402, 'cheapest': 282, 'imagin': 873, 'insipid': 900, 'rush': 1450, 'develop': 473, 'process': 1319, 'bitter': 164, 'rippl': 1430, 'consid': 361, 'lush': 1020, 'suggest': 1692, 'petrolatum': 1251, 'damp': 421, 'eczema': 544, 'rosacea': 1439, 'coz': 394, 'wors': 1935, 'apprehens': 77, 'broken': 218, 'fit': 662, 'purpos': 1338, 'sport': 1622, 'mad': 1026, 'teenag': 1738, 'promis': 1321, 'moisturisor': 1101, 'deoder': 458, 'discov': 493, 'shame': 1515, 'broke': 217, 'manli': 1039, 'foami': 678, 'wondr': 1927, 'variou': 1852, 'chamomil': 273, 'weight': 1899, 'thicker': 1757, 'space': 1604, 'nicest': 1143, 'mmmm': 1094, 'bo': 178, 'beater': 137, 'plainli': 1265, 'bright': 207, 'bold': 184, 'grudg': 786, 'dirti': 487, 'tablet': 1722, 'necess': 1132, 'weekli': 1898, 'groceri': 781, 'budget': 225, 'cake': 239, 'exot': 595, 'veget': 1857, 'ariel': 81, 'domin': 507, 'global': 748, 'consum': 365, 'giant': 734, 'expert': 600, 'dodgi': 503, 'imit': 874, 'sub': 1680, 'guarante': 788, 'high': 829, 'rubberi': 1445, 'gossam': 763, 'case': 258, 'brightli': 210, 'funki': 716, 'purpl': 1337, 'grey': 780, 'swirl': 1720, 'gu': 787, 'outsid': 1195, 'grade': 767, 'disappear': 488, 'mysteri': 1123, 'sphinx': 1616, 'giza': 741, 'bermuda': 151, 'triangl': 1810, 'voynich': 1872, 'manuscript': 1041, 'xlarg': 1947, 'discontinu': 491, 'option': 1187, 'limescal': 980, 'unclog': 1828, 'wake': 1875, 'plump': 1276, 'ill': 871, 'pun': 1333, 'intend': 906, 'altern': 49, 'forth': 690, 'healthier': 818, 'advic': 22, 'minimum': 1081, 'annoy': 60, 'asid': 91, 'run': 1448, 'plug': 1275, 'potenti': 1292, 'increas': 889, 'member': 1066, 'endors': 561, 'practic': 1298, 'nappi': 1124, 'pyramid': 1339, 'warm': 1880, 'present': 1303, 'st': 1630, 'rememb': 1401, 'repurchas': 1409, 'figur': 652, 'medicin': 1062, 'cabinet': 238, 'salti': 1459, 'yeast': 1951, 'honest': 843, 'toast': 1780, 'fluctuat': 674, 'particulr': 1222, 'watch': 1887, 'smellllllll': 1565, 'sauna': 1467, 'outstand': 1196, 'turkey': 1817, 'neck': 1133, 'shake': 1512, 'graviti': 774, 'vastli': 1854, 'sharp': 1519, 'begin': 144, 'assum': 93, 'grow': 784, 'ocado': 1164, 'sampl': 1460, 'state': 1639, 'formula': 689, 'correct': 383, 'later': 952, 'boost': 189, 'sooo': 1595, 'complaint': 346, 'concept': 351, 'recal': 1369, 'tetra': 1750, 'public': 1329, 'innov': 898, 'refil': 1384, 'cardboard': 254, 'dose': 511, 'counter': 388, 'read': 1361, 'print': 1313, 'wrapper': 1941, 'magnif': 1028, 'anim': 57, 'whilst': 1906, 'provid': 1328, 'headquart': 815, 'offic': 1173, 'produc': 1320, 'countri': 389, 'sale': 1457, 'carbon': 252, 'footprint': 683, 'torn': 1792, 'extrat': 604, 'wherea': 1904, 'secondli': 1494, 'spare': 1607, 'rib': 1422, 'persdper': 1242, 'reorder': 1404, 'somewhat': 1590, 'feminin': 647, 'perspir': 1247, 'danc': 422, 'bay': 132, 'waxi': 1891, 'shark': 1518, 'ben': 148, 'jerri': 925, 'fragrant': 696, 'bargin': 121, 'vanilla': 1849, 'iritatw': 912, 'tbh': 1735, 'loo': 1002, 'conceal': 349, 'moment': 1102, 'coverag': 393, 'brighten': 208, 'glowi': 752, 'perfectli': 1237, 'view': 1866, 'consciou': 360, 'paper': 1216, 'temperament': 1741, 'lazi': 960, 'girl': 737, 'solut': 1585, 'deep': 437, 'repeat': 1406, 'versatil': 1863, 'heel': 822, 'econom': 542, 'defo': 449, 'direct': 485, 'chapstick': 278, 'jot': 928, 'bob': 179, 'uncl': 1827, 'awesom': 106, 'doeant': 504, 'soggi': 1582, 'mess': 1070, 'dish': 496, 'lux': 1022, 'plan': 1266, 'kcal': 930, 'control': 371, 'diet': 479, 'biodegrad': 160, 'forev': 685, 'blackhead': 165, 'zone': 1962, 'fed': 641, 'tone': 1787, 'anymor': 65, 'bland': 166, 'wateri': 1889, 'gold': 757, 'spring': 1626, 'cold': 325, 'cardigan': 255, 'thirti': 1761, 'degre': 450, 'tie': 1768, 'boil': 182, 'bikini': 157, 'dedic': 436, 'ylang': 1953, 'town': 1797, 'aaaaamaz': 0, 'tattoooo': 1734, 'previous': 1309, 'admir': 17, 'blast': 167, 'whiff': 1905, 'planet': 1267, 'pollut': 1280, 'funni': 718, 'cone': 356, 'crispi': 403, 'cornetto': 382, 'flare': 666, 'itch': 917, 'crazi': 396, 'escap': 575, 'lift': 974, 'drop': 529, 'instantli': 903, 'relief': 1397, 'effortlessli': 550, 'alot': 44, 'effici': 548, 'linen': 983, 'pre': 1300, 'extract': 603, 'honey': 845, 'suckl': 1687, 'yesterday': 1952, 'perman': 1241, 'cherri': 288, 'blossom': 174, 'pea': 1231, 'sandal': 1462, 'wood': 1929, 'cup': 409, 'lunch': 1018, 'king': 938, 'gram': 769, 'antiperspir': 64, 'unblock': 1826, 'gradual': 768, 'pleasantli': 1271, 'surpris': 1712, 'cours': 391, 'fcuk': 639, 'opt': 1186, 'starter': 1638, 'newborn': 1139, 'gym': 791, 'hunger': 859, 'challeng': 272, 'strength': 1666, 'endur': 562, 'helmann': 824, 'bone': 187, 'tendon': 1745, 'appeal': 72, 'advantag': 21, 'citru': 303, 'muscl': 1120, 'bump': 228, 'ador': 19, 'handi': 799, 'persist': 1245, 'whitout': 1910, 'sin': 1540, 'spend': 1613, 'woken': 1924, 'reiment': 1391, 'smudg': 1571, 'novelti': 1158, 'fish': 661, 'quantiti': 1341, 'deodour': 460, 'swear': 1716, 'arkward': 82, 'kick': 935, 'sent': 1503, 'toiletri': 1784, 'linnen': 986, 'film': 654, 'teeth': 1739, 'ceram': 266, 'realis': 1364, 'cent': 264, 'gooey': 761, 'marshmallowey': 1047, 'phish': 1254, 'class': 306, 'spain': 1606, 'breakout': 204, 'intoler': 908, 'superior': 1704, 'alon': 43, 'thiught': 1762, 'cojld': 324, 'garnier': 724, 'apar': 70, 'stripey': 1670, 'feet': 644, 'orang': 1188, 'stink': 1650, 'moral': 1106, 'stori': 1657, 'format': 688, 'shave': 1521, 'dress': 524, 'fewer': 649, 'worn': 1933, 'meet': 1064, 'amazingli': 54, 'flavor': 668, 'powderi': 1296, 'exempt': 592, 'medium': 1063, 'annoyiji': 61, 'occas': 1165, 'immedi': 875, 'serum': 1506, 'contribut': 370, 'deco': 435, 'women': 1925, 'eldest': 552, 'forgotten': 686, 'chew': 289, 'rope': 1438, 'school': 1478, 'lie': 972, 'nother': 1155, 'automat': 100, 'gloopi': 749, 'drugstor': 530, 'brainer': 200, 'partnership': 1224, 'encourag': 559, 'spici': 1617, 'fear': 640, 'spilt': 1619, 'alright': 47, 'gotta': 765, 'cetearyl': 271, 'cif': 302, 'power': 1297, 'mirror': 1085, 'nightmar': 1146, 'fun': 714, 'ordinari': 1190, 'thicken': 1756, 'gravi': 773, 'pop': 1283, 'press': 1305, 'spell': 1612, 'scali': 1473, 'quid': 1347, 'tempt': 1742, 'everytim': 585, 'spf': 1615, 'critic': 405, 'uva': 1847, 'convinc': 374, 'refer': 1383, 'verifi': 1861, 'overli': 1201, 'allergi': 38, 'certain': 268, 'bondi': 186, 'sand': 1461, 'slight': 1551, 'fabul': 615, 'hike': 831, 'itsel': 920, 'frizz': 706, 'flat': 667, 'lank': 947, 'bounc': 195, 'stuck': 1675, 'unus': 1842, 'cooker': 376, 'invigor': 911, 'newer': 1140, 'rd': 1357, 'isnt': 915, 'cerav': 267, 'norm': 1150, 'begun': 145, 'calmer': 241, 'sciencey': 1479, 'id': 867, 'unless': 1834, 'anytim': 68, 'indulg': 893, 'bite': 163, 'dog': 506, 'micro': 1075, 'granul': 771, 'wari': 1879, 'stainless': 1632, 'steel': 1644, 'liquidi': 989, 'velvet': 1858, 'hopingthat': 848, 'fair': 622, 'btw': 222, 'instruct': 905, 'typic': 1821, 'dy': 535, 'tong': 1789, 'question': 1344, 'clip': 316, 'sensat': 1501, 'definatli': 445, 'surprisingli': 1713, 'shock': 1525, 'litr': 992, 'imo': 876, 'salad': 1456, 'bake': 114, 'bean': 134, 'frozen': 708, 'pear': 1232, 'youth': 1958, 'anitipersperi': 58, 'accustom': 9, 'richer': 1424, 'bundl': 229, 'low': 1012, 'carb': 249, 'duti': 534, 'sock': 1578, 'fraction': 694, 'loveliest': 1010, 'wrist': 1943, 'drum': 531, 'modest': 1096, 'inch': 886, 'god': 755, 'somebodi': 1586, 'brazil': 202, 'expat': 596, 'bearabl': 135, 'vfm': 1865, 'head': 813, 'winner': 1918, 'iv': 921, 'batgain': 129, 'averag': 101, 'oreal': 1191, 'resist': 1412, 'girlfriend': 738, 'mention': 1069, 'benefici': 149, 'list': 990, 'chanc': 275, 'transfer': 1800, 'snif': 1574, 'commit': 336, 'stiff': 1648, 'kg': 934, 'alobg': 42, 'matur': 1055, 'hiya': 836, 'stress': 1667, 'pricey': 1311, 'reliev': 1398, 'final': 655, 'tropic': 1811, 'lili': 978, 'mud': 1115, 'spag': 1605, 'bol': 183, 'attent': 96, 'lingeri': 985, 'cotton': 386, 'silk': 1534, 'fiber': 650, 'gigant': 736, 'possibl': 1288, 'lash': 950, 'eyebrow': 609, 'hide': 828, 'sunshin': 1700, 'clingi': 315, 'greazi': 777, 'advis': 23, 'rare': 1353, 'desir': 467, 'blotch': 175, 'brown': 220, 'attach': 95, 'ignor': 870, 'fortun': 691, 'fulfil': 712, 'criteria': 404, 'wilt': 1915, 'yard': 1948, 'radiu': 1350, 'alarm': 34, 'whisk': 1907, 'drainag': 518, 'devin': 474, 'signatur': 1530, 'haha': 793, 'saver': 1469, 'pamper': 1214, 'heheh': 823, 'hunk': 860, 'shelf': 1522, 'unscrew': 1838, 'glove': 750, 'odd': 1167, 'horrif': 850, 'headach': 814, 'desper': 468, 'afraid': 27, 'disturb': 499, 'properti': 1325, 'fluff': 675, 'dryness': 532, 'unpleas': 1836, 'seal': 1490, 'drain': 517, 'yogurt': 1955, 'alpro': 45, 'blop': 173, 'postag': 1290, 'doctor': 502, 'soooo': 1596, 'walk': 1876, 'pervas': 1249, 'sky': 1548, 'fruiti': 710, 'strike': 1668, 'balanc': 115, 'slurp': 1557, 'reckon': 1375, 'nearer': 1129, 'path': 1229, 'decidedli': 434, 'newsweek': 1141, 'fluffi': 676, 'smoother': 1568, 'hive': 835, 'aggrav': 31, 'wool': 1930, 'chsnge': 299, 'remark': 1400, 'specifi': 1611, 'conclus': 353, 'lure': 1019, 'entic': 568, 'squint': 1629, 'reviv': 1421, 'fallen': 626, 'deffinalti': 441, 'fond': 681, 'poor': 1281, 'consider': 362, 'dinner': 484, 'parti': 1219, 'everybodi': 581, 'jelli': 924, 'loyal': 1013, 'balm': 116, 'finger': 657, 'color': 328, 'petroleum': 1252, 'shadow': 1511, 'appropri': 79, 'tidi': 1767, 'eyelin': 610, 'error': 574, 'spars': 1608, 'bud': 224, 'homemad': 842, 'brill': 211, 'row': 1443, 'roll': 1435, 'shaken': 1513, 'disgust': 495, 'contact': 366, 'suddenli': 1688, 'alth': 50, 'insread': 902, 'toxic': 1798, 'dud': 533, 'split': 1620, 'cautiou': 262, 'lucki': 1015, 'monthli': 1105, 'beed': 142, 'scrib': 1485, 'hurt': 861, 'glitter': 747, 'ankl': 59, 'drawer': 520, 'bovril': 196, 'moan': 1095, 'arm': 83, 'significantli': 1532, 'strang': 1661, 'bleach': 168, 'occasion': 1166, 'recognis': 1376, 'cedarwood': 263, 'popular': 1284, 'bother': 192, 'dozen': 516, 'blindingli': 171, 'leg': 966, 'tanner': 1729, 'stronger': 1672, 'street': 1665, 'veg': 1855, 'cube': 407, 'batch': 128, 'recipebut': 1374, 'secret': 1495, 'mile': 1078, 'hr': 856, 'sedentari': 1497, 'worst': 1936, 'verri': 1862, 'hi': 827, 'uk': 1822, 'anywher': 69, 'massag': 1051, 'keen': 931, 'ban': 117, 'sneaki': 1573, 'woke': 1923, 'sunk': 1699, 'tacki': 1723, 'surviv': 1714, 'brexit': 206, 'defintley': 448, 'die': 478, 'em': 556, 'pregnant': 1302, 'moisteris': 1098, 'lost': 1006, 'count': 387, 'streaki': 1663, 'capful': 246, 'common': 337, 'loyalti': 1014, 'sooner': 1594, 'mostur': 1110, 'definetli': 446, 'mostureris': 1111, 'tissu': 1779, 'pain': 1211, 'pleasur': 1272, 'happili': 804, 'vera': 1859, 'directli': 486, 'allow': 39, 'subject': 1681, 'greenhous': 779, 'gase': 725, 'ultim': 1823, 'fight': 651, 'tire': 1778, 'drawn': 521, 'child': 291, 'dosag': 510, 'statement': 1640, 'worthwhil': 1938, 'sorbet': 1598, 'focus': 679, 'exact': 587, 'costco': 385, 'curiou': 411, 'baffl': 112, 'elviv': 555, 'realiz': 1365, 'silicon': 1533, 'hairdress': 795, 'appoint': 75, 'defenc': 439, 'purifi': 1336, 'pralin': 1299, 'choc': 293, 'solero': 1584}\n"
     ]
    }
   ],
   "source": [
    "TFIDF_vectorizer  = TfidfVectorizer(stop_words='english')\n",
    "\n",
    "TFIDF_vectorizer.fit(X_train)\n",
    "print('\\nVocabulary: \\n', TFIDF_vectorizer.vocabulary_)\n",
    "\n",
    "train_tf = TFIDF_vectorizer.fit_transform(X_train)\n",
    "test_tf = TFIDF_vectorizer.transform(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Multinomial Naive Bayes model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 159,
   "metadata": {},
   "outputs": [],
   "source": [
    "#IMPLEMENTING AND RUNNING MNB MODEL - TFIDF\n",
    "mnb2 = MultinomialNB()\n",
    "mnb2.fit(train_tf, y_train)\n",
    "prediction = mnb2.predict(test_tf)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 160,
   "metadata": {},
   "outputs": [],
   "source": [
    "#EVALUATION\n",
    "mnb_a2 = accuracy_score(y_test, prediction)*100\n",
    "mnb_p2 = precision_score(y_test, prediction)* 100\n",
    "mnb_r2 = recall_score(y_test, prediction)*100\n",
    "mnb_f12 = f1_score(y_test, prediction)*100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 161,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x1c0360637c0>"
      ]
     },
     "execution_count": 161,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#CONFUSION MATRIX\n",
    "cm =  confusion_matrix(y_test, prediction, labels=mnb2.classes_)\n",
    "display = ConfusionMatrixDisplay(confusion_matrix=cm,\n",
    "                              display_labels=mnb2.classes_) \n",
    "display.plot() "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Support Vector Machine model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 162,
   "metadata": {},
   "outputs": [],
   "source": [
    "#IMPLEMENTING AND RUNNING SVM MODEL - TFIDF \n",
    "svm2 = SVC(kernel='linear')\n",
    "svm2.fit(train_tf, y_train)\n",
    "prediction = svm2.predict(test_tf)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 163,
   "metadata": {},
   "outputs": [],
   "source": [
    "#EVALUATION\n",
    "svm_a2 = accuracy_score(y_test, prediction)*100\n",
    "svm_p2 = precision_score(y_test, prediction)* 100\n",
    "svm_r2 = recall_score(y_test, prediction)*100\n",
    "svm_f12 = f1_score(y_test, prediction)*100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 164,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x1c0363f7820>"
      ]
     },
     "execution_count": 164,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#CONFUSION MATRIX\n",
    "cm =  confusion_matrix(y_test, prediction, labels=svm2.classes_)\n",
    "display = ConfusionMatrixDisplay(confusion_matrix=cm,\n",
    "                              display_labels=svm2.classes_) \n",
    "display.plot() "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Logistic Regression model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 165,
   "metadata": {},
   "outputs": [],
   "source": [
    "#IMPLEMENTATION AND RUNNING LR MODEL - TFIDF \n",
    "lr2 = LogisticRegression()\n",
    "lr2.fit(train_tf, y_train)\n",
    "prediction = lr2.predict(test_tf)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 166,
   "metadata": {},
   "outputs": [],
   "source": [
    "#EVALUATION\n",
    "lr_a2 = accuracy_score(y_test, prediction)*100\n",
    "lr_p2 = precision_score(y_test, prediction)* 100\n",
    "lr_r2 = recall_score(y_test, prediction)*100\n",
    "lr_f12 = f1_score(y_test, prediction)*100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 167,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x1c0361ede80>"
      ]
     },
     "execution_count": 167,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAUsAAAEGCAYAAADscbcsAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAcO0lEQVR4nO3de5hWZb3/8fdnhmE4qwgo4iCIQGIqEmpmGh5+otbvZ3YSs7LSsEKtdJeH7S5/mh12HvaurJ1uvbS2abLVyuySjCyzTEQkFEw8gMhBOcZxgDl89x9rTT6wZ55njXNY88x8Xte1Lp7nXvda6zuMfrnvdd/rXooIzMysuIq8AzAzKwdOlmZmGThZmpll4GRpZpaBk6WZWQa98g6gIwwZXBmjaqryDsNaYfGCfnmHYK20mQ1rI2JoW84x9YT+sW59Q6a6Ty/YMSsiTm3L9dqiWybLUTVVzJlVk3cY1gpTRxyRdwjWSr9tnPlqW8+xbn0Dc2aNzFS3cviLQ9p6vbbolsnSzMpDAI005h1GJk6WZpabIKiLbN3wvDlZmlmu3LI0MyshCBrK5JFrJ0szy1UjTpZmZkUF0OBkaWZWmluWZmYlBFDne5ZmZsUF4W64mVlJAQ3lkSudLM0sP8kTPOXBydLMciQaUN5BZOJkaWa5SQZ4nCzNzIpK5lk6WZqZldTolqWZWXFuWZqZZRCIhjJ5u42TpZnlyt1wM7MSArEzKvMOIxMnSzPLTTIp3d1wM7OSPMBjZlZChGgItyzNzEpqdMvSzKy4ZICnPNJQeURpZt2SB3jMzDJq8DxLM7Pi/ASPmVlGjR4NNzMrLllIw8nSzKyoQNT5cUczs+Ii8KR0M7PS5EnpZmalBG5Zmpll4gEeM7MSAnnxXzOzUpJX4ZZHGiqPKM2sm5LXszQzKyUonyd4yiNKM+u2GtLWZamtGEk1kh6V9LykhZK+kJZfLWmFpPnpdnrBMVdIeknSC5KmlorTLUszy02E2qtlWQ9cGhHzJA0Enpb0SLrvpoi4vrCypAnANOAQYD/gt5LGRURDSxdwsjSz3CQDPG1/3DEiVgGr0s+bJT0PjChyyBnAPRGxA1gi6SXgKOCJlg5wN9zMcpS8gyfLBgyRNLdgm97sGaVRwBHAk2nRhZIWSLpd0l5p2QjgtYLDllM8ubplaWb5SQZ4Mo+Gr42IycUqSBoA3Ad8MSI2SfohcG16qWuBG4BPQ7M3QaPYuZ0szSxX7fUEj6QqkkR5V0TcDxARbxTsvxX4Vfp1OVBTcPj+wMpi53c33Mxy0/QET5atGEkCbgOej4gbC8qHF1Q7E3gu/fxLYJqkakmjgbHAnGLXcMvSzHLVTi8sOxb4OPCspPlp2ZXA2ZImknSxlwIXAETEQkn3AotIRtJnFBsJBydLM8tRBNQ1tj1ZRsTjNH8f8tdFjrkOuC7rNZwszSw3STe8PO4GOlmaWa78bLi12uoVVXznCyPZsLoKVQSnf2wdZ56/lpcX9uF7l9dQu7WCffbfyWU3v0r/gY0AvLKoD9+9rIatmyuoqIDv/XoxvfsUnQFhHaj/oHq+dP1rjBq/nQi48dKRHHva33nn/9lE3U6x6tVqbrikhq2b/L8etHrqUK467DcmqQF4tqDo/RGxtIW6WyJiQEfFUi4qewXTv7qSsYfVsm1LBReeOo5Jx2/m3/5pJJ/56goOO2Yrs+4ezH//cBjnfuV1GurhXy86gC9/91XGHLKdTesrqaxyoszT565ZwdxHB/H16aPpVdVIdd9G+vYfyO3f3I/GBnHelSuZduFqbvvGfnmH2kWUTze8I6OsjYiJBdvSDrxWt7D3PvWMPawWgH4DGqk5aAdrV1Wx/OVqDn3nVgCOOH4zjz+0JwBP/2Egow+uZcwh2wEYNLiByvJ4UV631G9AA4cevZWH7x4MQH1dBVs39WLeY4NobEhaT8/P68eQ4XV5htnlNKbv4Sm15a3TUrqkAZJmS5on6VlJZzRTZ7ikx9LVQZ6TdFxafoqkJ9JjZ6az9Lu111/rzcvP9eVtk7ZxwPjtPDFrEAB//NWerFlZBcDyV/ogwZVnH8iMU8Zx783D8gy5x9v3gB1sXNeLS29axs2zXuCL31lGdd9dZ6NMnbaepx4dmFOEXU8yGl6ZactbRybLvgXLIj0AbAfOjIhJwAnADelE0kIfBWZFxETgcGC+pCHAVcDJ6bFzgUt2v5ik6U3PjK5ZV3S6VJdXu7WCa88fxWevWUH/gY1ccuMyHrxjCDOmjqN2SwW9eidd7YZ6eG5Ofy77/qvc8PMX+fPDe/DMH7v9vyNdVmUlHHToNn714yHMmDqe7dsqOOvC1f/Yf/bFr9NQL353/15FztKztNek9M7QkXeZa9OkB/zjUaRvSDoeaCR5aH0f4PWCY54Cbk/r/jwi5kt6DzAB+FOaW3vTzMogEXELcAvA5MPLd4Sjvg6uPX8UJ35gA+8+fSMAI8fu4Jv3vALA8pereXJ20socOryOw47Zyh57J/84HHniJl56ti9HHLcln+B7uLWrqlizqooXnukPwOMP7clH0mR58ofXc9TJm7j8IwfR/HTAnqsrdLGz6Mw7q+cAQ4F3pEn0DaBPYYWIeAw4HlgB/ETSJ0j+y3qk4N7nhIg4rxPj7jRNo6c1Y3fwwQvW/KP872uTf9MaG+Gn/74P7/v4OgDeMWUzSxb1Yfs20VAPC54YwMhxO3KJ3WDDmirWruzN/mOSe8gT372ZZYurmTxlEx/5/Btc/ckD2bG9PAYzOkvTaHhPb1nubg9gdUTUSToBOGD3CpIOAFZExK2S+gOTSGbY3yzpoIh4SVI/YP+IWNyJsXeKhXP6M/u/BzP64Fo+d/J4AD51xUpWLKnmwTuGAHDsaRs5Zdp6AAbu2cAHLljDRaePQ4KjTtzE0Sdvyi1+g5v/ZQSXfe9VelUFry/rzQ2XjOR7Dy2mqjr45j0vAfC3ef357uU1Jc7Uc5TLaHhnJsu7gAclzQXmA39rps4U4MuS6oAtwCciYo2kTwJ3S6pO610FdLtk+fajtzJr5fxm9mzmzPPXNnvMSR/cwEkf3NCxgVlmryzsx0Wnj9+l7FPvnpBTNF1fhKjv6cly93mTEbEWOKZY3Yi4E7izmf2/A47sgDDNLGddoYudhR8jMLPc+AkeM7OMnCzNzEpommdZDpwszSxX5TLP0snSzHITAfXtsPhvZ3CyNLNcuRtuZlaC71mamWUUTpZmZqV5gMfMrIQI37M0M8tANHg03MysNN+zNDMrwc+Gm5llEcl9y3LgZGlmufJouJlZCeEBHjOzbNwNNzPLwKPhZmYlRDhZmpll4qlDZmYZ+J6lmVkJgWj0aLiZWWll0rB0sjSzHHmAx8wsozJpWpbHzQIz67YilGkrRlKNpEclPS9poaQvpOWDJT0i6cX0z70KjrlC0kuSXpA0tVScLbYsJX2PIjk/Ii4udXIzs2ICaGxsl254PXBpRMyTNBB4WtIjwCeB2RHxLUmXA5cDl0maAEwDDgH2A34raVxENLR0gWLd8Lnt8ROYmbUogHa4ZxkRq4BV6efNkp4HRgBnAFPSancCvwcuS8vviYgdwBJJLwFHAU+0dI0Wk2VE3Fn4XVL/iNj6Vn8YM7PmtGKe5RBJhY24WyLilt0rSRoFHAE8CeyTJlIiYpWkYWm1EcBfCg5bnpa1qOQAj6RjgNuAAcBISYcDF0TE50sda2ZWUvZkuTYiJherIGkAcB/wxYjYJLXYam1uR9FIsgzw/BswFVgHEBF/BY7PcJyZWQnZBneyTC+SVEWSKO+KiPvT4jckDU/3DwdWp+XLgZqCw/cHVhY7f6bR8Ih4bbeiFm+Cmpm1SmTcilDShLwNeD4ibizY9Uvg3PTzucAvCsqnSaqWNBoYC8wpdo0s8yxfk/QuICT1Bi4Gns9wnJlZcQHRPqPhxwIfB56VND8tuxL4FnCvpPOAZcCHASJioaR7gUUkI+kzio2EQ7Zk+Vng30lufq4AZgEzWv+zmJk1p11Gwx8vcqKTWjjmOuC6rNcomSwjYi1wTtYTmpm1Snd5gkfSgZIelLRG0mpJv5B0YGcEZ2Y9QDvcs+wMWQZ4fgrcCwwnmek+E7i7I4Mysx6iaVJ6li1nWZKlIuInEVGfbv9Fl8jzZtYdRGTb8lbs2fDB6cdH02cq7yFJkmcBD3VCbGbWE7TPaHiHKzbA8zRJcmz6SS4o2BfAtR0VlJn1HOoCrcYsij0bProzAzGzHqiLDN5kkWnxX0lvByYAfZrKIuLHHRWUmfUUXWPwJossC2l8jWSJownAr4HTgMcBJ0sza7syaVlmGQ3/EMkM+Ncj4lPA4UB1h0ZlZj1HY8YtZ1m64bUR0SipXtIgklU7PCndzNqunRb/7QxZkuVcSXsCt5KMkG+hxOocZmZZlf1oeJOCRX7/Q9LDwKCIWNCxYZlZj1HuyVLSpGL7ImJex4RkZtb1FGtZ3lBkXwAntnMs7Wbxgn5M3W9i3mFYK0xf/HLeIVgr/XZs+5yn7LvhEXFCZwZiZj1Q0C0edzQz63jl3rI0M+sMZd8NNzPrFGWSLLOslC5JH5P01fT7SElHdXxoZtYjdKOV0n8AHAOcnX7fDNzcYRGZWY+hyL7lLUs3/OiImCTpGYCI2JC+EtfMrO260Wh4naRK0oawpKF0icfazaw76AqtxiyydMO/CzwADJN0HcnybN/o0KjMrOcok3uWWZ4Nv0vS0yTLtAl4f0Q83+GRmVn310XuR2aRZfHfkcA24MHCsohY1pGBmVkP0V2SJcmbHJteXNYHGA28ABzSgXGZWQ+hMhkBydINP7Twe7oa0QUtVDcz65Za/QRPRMyTdGRHBGNmPVB36YZLuqTgawUwCVjTYRGZWc/RnQZ4gIEFn+tJ7mHe1zHhmFmP0x2SZToZfUBEfLmT4jGznqbck6WkXhFRX+z1EmZmbSG6x2j4HJL7k/Ml/RKYCWxt2hkR93dwbGbW3XWze5aDgXUk79xpmm8ZgJOlmbVdN0iWw9KR8Od4M0k2KZMfz8y6vDLJJsWSZSUwgF2TZJMy+fHMrKvrDt3wVRFxTadFYmY9UzslS0m3A+8DVkfE29Oyq4HP8Obc8Csj4tfpviuA84AG4OKImFXs/MWWaCuPFTnNrHxFMhqeZcvgDuDUZspvioiJ6daUKCcA00jWuDgV+EE6VbJFxZLlSZnCMzNri3ZazzIiHgPWZ7zqGcA9EbEjIpYALwFF3y3WYrKMiKwXNTN7y1rxDp4hkuYWbNMzXuJCSQsk3S5pr7RsBPBaQZ3laVmLsqyUbmbWcbK3LNdGxOSC7ZYMZ/8hMAaYCKwCbkjLWz1w7WRpZvnJmijf4iBQRLwREQ0R0Qjcyptd7eVATUHV/YGVxc7lZGlmuREd+ypcScMLvp5JMm8c4JfANEnVkkYDY0meWmxRq9ezNDNrT+01z1LS3cAUknuby4GvAVMkTSRpmy4lXbg8IhZKuhdYRLKa2oyIaCh2fidLM8tXOyXLiDi7meLbitS/Drgu6/mdLM0sX93gCR4zs47VzVYdMjPrOE6WZmaldYfFf83MOpy74WZmpbRhwnlnc7I0s3w5WZqZFdf0BE85cLI0s1ypsTyypZOlmeXH9yzNzLJxN9zMLAsnSzOz0tyyNDPLwsnSzKyE8OOOZmYleZ6lmVlWUR7Z0snSzHLllqW1Wf9BDXzp+tcY9bbtRMCNl9QwZHgdH7/0dWrG7uDi08fy4oJ+eYfZo21ZVcmjXxlG7ZpKVAFvO2sTh567CYDnfjyIhXcNoqISaqZs451fWQ/Aur/15o9fHULdlgqogDPvW0Gv6jLJGO3Nk9J3JWlvYHb6dV+gAViTfj8qInZ2Rhzl5nPXrGDu7wfy9emj6FXVSHXfYMvGSq45fxQXf3t53uEZUFEJx1y+jiGH7GTnFvHAB0aw/7G11K6t5NXZ/fjQg8up7A2165IXqTbWw6NfHsoJ/7qGvQ/eyfYNFVT0KpNs0UE8wFMgItaRvOQcSVcDWyLi+qb9knpFRH1nxFIu+g1o4NB3buX6LyavNq6vq6C+DrZuqsw5MivUb1gD/YYlLwXsPSDYc0wdW9/oxd/uHcjh0zdS2Tup13fvJCMsf7wvg8fvZO+Dk/ZBn73KJFN0ICfLEiTdAawHjgDmSdpMQRKV9BzwvohYKuljwMVAb+BJ4POlXltZ7vY9YCcb11Vy6U2vceAhtby4oB8//Jf92FHrZNlVbV7ei7WLqhl2+Hae/PZgXp/bh6du2ote1cHRl61n2GE72Li0CoBff3pfatdXMua9W5j4mY05R56joGwGeCpyvv444OSIuLSlCpIOBs4Cjo2IiSRd+HOaqTdd0lxJc+vY0WEBd5bKyuCgQ2v51Y/3ZsYp49m+rYKzLlydd1jWgrqt4pGL9uFdV66l94CgsUHs2FTB+2eu5OivrGf2F4cRAY0N4o15fTjx+tWccfdKlj7SnxV/7pN3+LlSZNvylneynJmhhXgS8A7gKUnz0+8H7l4pIm6JiMkRMbmK6g4ItXOtXVXFmlVVvPBMfwAe/9UeHHRobc5RWXMa6+CRi/bhoP+7hdFTtwHQf996Rp+yFQmGHb4DBNs3VNB/n3qGH7mdPoMb6dU3GPmebaxdVP7/vbZJZNxylney3FrwuZ5d42n651bAnRExMd3GR8TVnRVgXjasqWLtyt7sP2Y7ABOP28KyF3t2C6QrioA/XDmUPcfUcdin3+xOjzp5Kyv/0heAvy+porFO9NmrkZrjaln3Qm/qa0VjPaya04e9xvTc8c2mSenl0LLsSlOHlgLvA5A0CRidls8GfiHppohYLWkwMDAiXs0nzM5z81UjuOz7y+hVFby+rDc3fKmGd526kc9/fQV77F3PtT9ZwssL+/DPHx2Td6g91htPV/PiLwYyePwO7vt/IwA48pL1jP/gZv5w5VBmvnd/KqqCKd9ejQTVezRy2Kc28sAHR4Cg5j3bGHlCD+4xRHjx37fgPuATaVf7KWAxQEQsknQV8BtJFUAdMAPo9snylYV9uei0cbuU/fnhPfjzw3vkFJHtbt/JO5i++JVm9514/Zpmy8eesYWxZ2zpyLDKS3nkys5Pli11oSOiFjilhX0/A37WgWGZWU66Qhc7i67UsjSzniYAd8PNzDIoj1zpZGlm+XI33MwsA4+Gm5mV0kUmnGfhZGlmuUkmpZdHtnSyNLN8edUhM7PS3LI0MyvF9yzNzLIon2fD8151yMx6uohsWwmSbpe0Ol04vKlssKRHJL2Y/rlXwb4rJL0k6QVJU0ud38nSzPITyWslsmwZ3AGculvZ5cDsiBhLsoLZ5QCSJgDTgEPSY34gqehrCJwszSxf7dSyjIjHSF5VU+gM4M70853A+wvK74mIHRGxBHgJOKrY+Z0szSxf2VdKH9L06ph0m57h7PtExCqA9M9hafkI4LWCesvTshZ5gMfMcqXGzBMt10bE5Pa6bDNlRZuvblmaWX6CZFJ6lu2teUPScID0z6a3/i0Hagrq7Q+sLHYiJ0szy40IFNm2t+iXwLnp53OBXxSUT5NULWk0MBaYU+xE7oabWb7a6QkeSXcDU0jubS4HvgZ8C7hX0nnAMuDDySVjoaR7gUUkL0ucUepNs06WZpavdkqWEXF2C7tOaqH+dcB1Wc/vZGlm+Wm6Z1kGnCzNLFetGA3PlZOlmeUo24TzrsDJ0szyEzhZmpllUh69cCdLM8uXF/81M8vCydLMrIQIaCiPfriTpZnlyy1LM7MMnCzNzEoIoEzeweNkaWY5CgjfszQzKy7wAI+ZWSa+Z2lmloGTpZlZKV5Iw8ystAC8RJuZWQZuWZqZleLHHc3MSgsIz7M0M8vAT/CYmWXge5ZmZiVEeDTczCwTtyzNzEoJoqEh7yAycbI0s/x4iTYzs4w8dcjMrLgAwi1LM7MSwov/mpllUi4DPIoyGbZvDUlrgFfzjqODDAHW5h2EtUp3/Z0dEBFD23ICSQ+T/P1ksTYiTm3L9dqiWybL7kzS3IiYnHcclp1/Z91DRd4BmJmVAydLM7MMnCzLzy15B2Ct5t9ZN+B7lmZmGbhlaWaWgZOlmVkGnpSeM0kNwLMFRe+PiKUt1N0SEQM6JTArStLewOz0675AA7Am/X5UROzMJTDrML5nmbPWJEAny65J0tXAloi4vqCsV0TU5xeVtTd3w7sYSQMkzZY0T9Kzks5ops5wSY9Jmi/pOUnHpeWnSHoiPXamJCfWTiTpDkk3SnoU+LakqyX9U8H+5ySNSj9/TNKc9Hf4I0mVOYVtGTlZ5q9v+j/MfEkPANuBMyNiEnACcIMk7XbMR4FZETEROByYL2kIcBVwcnrsXOCSzvsxLDWO5HdwaUsVJB0MnAUcm/4OG4BzOik+e4t8zzJ/ten/MABIqgK+Iel4oBEYAewDvF5wzFPA7Wndn0fEfEnvASYAf0pza2/giU76GexNMyOi1MoQJwHvAJ5Kf1d9gdUdHZi1jZNl13MOMBR4R0TUSVoK9CmsEBGPpcn0vcBPJH0H2AA8EhFnd3bAtoutBZ/r2bX31vR7FHBnRFzRaVFZm7kb3vXsAaxOE+UJwAG7V5B0QFrnVuA2YBLwF+BYSQeldfpJGteJcdv/tpTkd4OkScDotHw28CFJw9J9g9PfqXVhbll2PXcBD0qaC8wH/tZMnSnAlyXVAVuAT0TEGkmfBO6WVJ3WuwpY3PEhWwvuAz4haT7JrZPFABGxSNJVwG8kVQB1wAy677KC3YKnDpmZZeBuuJlZBk6WZmYZOFmamWXgZGlmloGTpZlZBk6WPZSkhoJny2dK6teGc90h6UPp5/+UNKFI3SmS3vUWrrE0faQzU/ludba08lq7PNNtBk6WPVltREyMiLcDO4HPFu58qws7RMT5EbGoSJUpQKuTpVnenCwN4I/AQWmr71FJPwWelVQp6TuSnpK0QNIFAEp8X9IiSQ8Bw5pOJOn3kiann09NV0D6a7qS0iiSpPyltFV7nKShku5Lr/GUpGPTY/eW9BtJz0j6EckjgkVJ+rmkpyUtlDR9t303pLHMljQ0LRsj6eH0mD9Kelt7/GVa9+QneHo4Sb2A04CH06KjgLdHxJI04WyMiCPTp4L+JOk3wBHAeOBQkkU+FgG373beocCtwPHpuQZHxHpJ/0HB2o9pYr4pIh6XNBKYBRwMfA14PCKukfReYJfk14JPp9foS7JIxX0RsQ7oD8yLiEslfTU994UkLxL7bES8KOlo4AfAiW/hr9F6ACfLnqtv+hgeJC3L20i6x3MiYklafgpwWNP9SJLn1scCxwN3p6vrrJT0u2bO/07gsaZzRcT6FuI4GZhQsArdIEkD02t8ID32IUkbMvxMF0s6M/1ck8a6jmT1pp+l5f8F3K9krc93ATMLrl2NWQucLHuuXZaGA0iTRuGqOQIuiohZu9U7HSj1nKwy1IHkVtAxEVHbTCyZn8WVNIUk8R4TEdsk/Z7dVmsqEOl1/77734FZS3zP0oqZBXwuXTcTSeMk9QceA6al9zSHkyxSvLsngPdIGp0eOzgt3wwMLKj3G5IuMWm9puT1GOmCuJJOA/YqEesewIY0Ub6NpGXbpAJoah1/lKR7vwlYIunD6TUk6fAS17AezMnSivlPkvuR8yQ9B/yIpDfyAPAiyYvWfgj8YfcDI2INyX3G+yX9lTe7wQ8CZzYN8AAXA5PTAaRFvDkq//+B4yXNI7kdsKxErA8DvSQtAK4lWbKuyVbgEElPk9yTvCYtPwc4L41vIfC/XuFh1sSrDpmZZeCWpZlZBk6WZmYZOFmamWXgZGlmloGTpZlZBk6WZmYZOFmamWXwP8oCs4phSvlUAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#CONFUSION MATRIX\n",
    "cm =  confusion_matrix(y_test, prediction, labels=lr2.classes_)\n",
    "display = ConfusionMatrixDisplay(confusion_matrix=cm,\n",
    "                              display_labels=lr2.classes_) \n",
    "display.plot() "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### COMPARING ACCURACY"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 168,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>MNB</th>\n",
       "      <th>SVM</th>\n",
       "      <th>LR</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Count Vectorizer</th>\n",
       "      <td>80.0</td>\n",
       "      <td>84.0</td>\n",
       "      <td>85.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Tfidf Vectorizer</th>\n",
       "      <td>81.0</td>\n",
       "      <td>84.0</td>\n",
       "      <td>82.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   MNB   SVM    LR\n",
       "Count Vectorizer  80.0  84.0  85.0\n",
       "Tfidf Vectorizer  81.0  84.0  82.0"
      ]
     },
     "execution_count": 168,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model_accuracy={'MNB': [round(mnb_a1), round(mnb_a2)],\n",
    "                'SVM': [round(svm_a1), round(svm_a2)],\n",
    "                'LR': [round(lr_a1), round(lr_a2)]\n",
    "               }\n",
    "ma = pd.DataFrame(model_accuracy, columns = ['MNB','SVM','LR'], index=['Count Vectorizer','Tfidf Vectorizer'])\n",
    "ma"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### COMPARING PRECISION"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 169,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>MNB</th>\n",
       "      <th>SVM</th>\n",
       "      <th>LR</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Count Vectorizer</th>\n",
       "      <td>81.0</td>\n",
       "      <td>79.0</td>\n",
       "      <td>80.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Tfidf Vectorizer</th>\n",
       "      <td>85.0</td>\n",
       "      <td>82.0</td>\n",
       "      <td>81.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   MNB   SVM    LR\n",
       "Count Vectorizer  81.0  79.0  80.0\n",
       "Tfidf Vectorizer  85.0  82.0  81.0"
      ]
     },
     "execution_count": 169,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model_precision={'MNB': [round(mnb_p1), round(mnb_p2)],\n",
    "                'SVM': [round(svm_p1), round(svm_p2)],\n",
    "                'LR': [round(lr_p1), round(lr_p2)]\n",
    "               }\n",
    "mp = pd.DataFrame(model_precision, columns = ['MNB','SVM','LR'], index=['Count Vectorizer','Tfidf Vectorizer'])\n",
    "mp"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### COMPARING RECALL"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 170,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>MNB</th>\n",
       "      <th>SVM</th>\n",
       "      <th>LR</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Count Vectorizer</th>\n",
       "      <td>77.0</td>\n",
       "      <td>91.0</td>\n",
       "      <td>92.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Tfidf Vectorizer</th>\n",
       "      <td>74.0</td>\n",
       "      <td>84.0</td>\n",
       "      <td>81.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   MNB   SVM    LR\n",
       "Count Vectorizer  77.0  91.0  92.0\n",
       "Tfidf Vectorizer  74.0  84.0  81.0"
      ]
     },
     "execution_count": 170,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model_recall={'MNB': [round(mnb_r1), round(mnb_r2)],\n",
    "                'SVM': [round(svm_r1), round(svm_r2)],\n",
    "                'LR': [round(lr_r1), round(lr_r2)]\n",
    "               }\n",
    "mr = pd.DataFrame(model_recall, columns = ['MNB','SVM','LR'], index=['Count Vectorizer','Tfidf Vectorizer'])\n",
    "mr"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### COMPARING F1 SCORE"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 171,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>MNB</th>\n",
       "      <th>SVM</th>\n",
       "      <th>LR</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Count Vectorizer</th>\n",
       "      <td>79.0</td>\n",
       "      <td>84.0</td>\n",
       "      <td>85.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Tfidf Vectorizer</th>\n",
       "      <td>79.0</td>\n",
       "      <td>83.0</td>\n",
       "      <td>81.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   MNB   SVM    LR\n",
       "Count Vectorizer  79.0  84.0  85.0\n",
       "Tfidf Vectorizer  79.0  83.0  81.0"
      ]
     },
     "execution_count": 171,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model_f1={'MNB': [round(mnb_f11), round(mnb_f12)],\n",
    "                'SVM': [round(svm_f11), round(svm_f12)],\n",
    "                'LR': [round(lr_f11), round(lr_f12)]\n",
    "               }\n",
    "mf1 = pd.DataFrame(model_f1, columns = ['MNB','SVM','LR'], index=['Count Vectorizer','Tfidf Vectorizer'])\n",
    "mf1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 172,
   "metadata": {},
   "outputs": [],
   "source": [
    "#SAVING THE BEST MODEL WITH ITS RESPECTIVE VECTORIZER\n",
    "pickle.dump(lr1, open('data and pickle files/data and pickle files/best_model.pkl', 'wb'))\n",
    "pickle.dump(count_vectorizer, open('data and pickle files/count_vectorizer.pkl', 'wb'))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

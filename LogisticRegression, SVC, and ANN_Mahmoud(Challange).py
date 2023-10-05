{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "56d3420c",
   "metadata": {},
   "source": [
    "# Data undrstanding and data preperation according to crisp model."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "89ff1279",
   "metadata": {},
   "source": [
    "### <ins> Importing modules and data </ins>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "a7f6a23d",
   "metadata": {},
   "outputs": [],
   "source": [
    "import warnings\n",
    "\n",
    "# To filter out all warnings\n",
    "warnings.filterwarnings(\"ignore\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "4d5a7533",
   "metadata": {},
   "outputs": [],
   "source": [
    "# import libraries\n",
    "import pandas as pd \n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt \n",
    "from IPython import get_ipython\n",
    "get_ipython().run_line_magic('matplotlib', 'qt')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "da2fe997",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(32950, 16)"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# import dataset\n",
    "df = pd.read_csv('banking_classification/new_train.csv')\n",
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "7df483df",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.11265553869499241"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "(df['y'] == 'yes').sum() / df.y.count()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ea5f81f2",
   "metadata": {},
   "source": [
    "### <ins> Exploring the data </ins>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "f1fce225",
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
       "      <th>count</th>\n",
       "      <th>mean</th>\n",
       "      <th>std</th>\n",
       "      <th>min</th>\n",
       "      <th>25%</th>\n",
       "      <th>50%</th>\n",
       "      <th>75%</th>\n",
       "      <th>max</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>age</th>\n",
       "      <td>32950.0</td>\n",
       "      <td>40.014112</td>\n",
       "      <td>10.403636</td>\n",
       "      <td>17.0</td>\n",
       "      <td>32.0</td>\n",
       "      <td>38.0</td>\n",
       "      <td>47.0</td>\n",
       "      <td>98.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>duration</th>\n",
       "      <td>32950.0</td>\n",
       "      <td>258.127466</td>\n",
       "      <td>258.975917</td>\n",
       "      <td>0.0</td>\n",
       "      <td>103.0</td>\n",
       "      <td>180.0</td>\n",
       "      <td>319.0</td>\n",
       "      <td>4918.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>campaign</th>\n",
       "      <td>32950.0</td>\n",
       "      <td>2.560607</td>\n",
       "      <td>2.752326</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>56.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>pdays</th>\n",
       "      <td>32950.0</td>\n",
       "      <td>962.052413</td>\n",
       "      <td>187.951096</td>\n",
       "      <td>0.0</td>\n",
       "      <td>999.0</td>\n",
       "      <td>999.0</td>\n",
       "      <td>999.0</td>\n",
       "      <td>999.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>previous</th>\n",
       "      <td>32950.0</td>\n",
       "      <td>0.174719</td>\n",
       "      <td>0.499025</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            count        mean         std   min    25%    50%    75%     max\n",
       "age       32950.0   40.014112   10.403636  17.0   32.0   38.0   47.0    98.0\n",
       "duration  32950.0  258.127466  258.975917   0.0  103.0  180.0  319.0  4918.0\n",
       "campaign  32950.0    2.560607    2.752326   1.0    1.0    2.0    3.0    56.0\n",
       "pdays     32950.0  962.052413  187.951096   0.0  999.0  999.0  999.0   999.0\n",
       "previous  32950.0    0.174719    0.499025   0.0    0.0    0.0    0.0     7.0"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# summary statistics\n",
    "df.describe().T"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "7ac4868c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "age            0\n",
       "job            0\n",
       "marital        0\n",
       "education      0\n",
       "default        0\n",
       "housing        0\n",
       "loan           0\n",
       "contact        0\n",
       "month          0\n",
       "day_of_week    0\n",
       "duration       0\n",
       "campaign       0\n",
       "pdays          0\n",
       "previous       0\n",
       "poutcome       0\n",
       "y              0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# checking null values\n",
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "43ecf16b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "8"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# check duplicetes\n",
    "df.duplicated().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "201ae40c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# remove dublicated rows\n",
    "df.drop_duplicates(inplace= True)\n",
    "df.duplicated().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "920e626d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "age             int64\n",
       "job            object\n",
       "marital        object\n",
       "education      object\n",
       "default        object\n",
       "housing        object\n",
       "loan           object\n",
       "contact        object\n",
       "month          object\n",
       "day_of_week    object\n",
       "duration        int64\n",
       "campaign        int64\n",
       "pdays           int64\n",
       "previous        int64\n",
       "poutcome       object\n",
       "y              object\n",
       "dtype: object"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# check data types\n",
    "df.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "c0d2994c",
   "metadata": {},
   "outputs": [],
   "source": [
    "#seperating df into two dataframes, one for dategorical data and the other for numerical data.\n",
    "df_categorical = pd.DataFrame()\n",
    "df_numeric = pd.DataFrame()\n",
    "columns = df.columns.values\n",
    "for column in columns:\n",
    "    if df[column].dtype != np.int64 and df[column].dtype != np.float64: \n",
    "        df_categorical[column] = df[column]\n",
    "    else:\n",
    "        df_numeric[column] = df[column]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "6d7da2a4",
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
       "      <th>age</th>\n",
       "      <th>duration</th>\n",
       "      <th>campaign</th>\n",
       "      <th>pdays</th>\n",
       "      <th>previous</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>49</td>\n",
       "      <td>227</td>\n",
       "      <td>4</td>\n",
       "      <td>999</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>37</td>\n",
       "      <td>202</td>\n",
       "      <td>2</td>\n",
       "      <td>999</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>78</td>\n",
       "      <td>1148</td>\n",
       "      <td>1</td>\n",
       "      <td>999</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>36</td>\n",
       "      <td>120</td>\n",
       "      <td>2</td>\n",
       "      <td>999</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>59</td>\n",
       "      <td>368</td>\n",
       "      <td>2</td>\n",
       "      <td>999</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   age  duration  campaign  pdays  previous\n",
       "0   49       227         4    999         0\n",
       "1   37       202         2    999         1\n",
       "2   78      1148         1    999         0\n",
       "3   36       120         2    999         0\n",
       "4   59       368         2    999         0"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_numeric.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "55fe7f43",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_numeric.hist();"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f9d30713",
   "metadata": {},
   "source": [
    "#### <span style=\"color:red\"> Based on the data describtion and the destribution of pdays, it is more effiecient to drop it.</span>\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "2e62c608",
   "metadata": {},
   "outputs": [],
   "source": [
    "# droping pdays column\n",
    "df_numeric.drop(columns= 'pdays', axis= 1, inplace= True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "c2c707ce",
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
       "      <th>age</th>\n",
       "      <th>duration</th>\n",
       "      <th>campaign</th>\n",
       "      <th>previous</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>49</td>\n",
       "      <td>227</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>37</td>\n",
       "      <td>202</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>78</td>\n",
       "      <td>1148</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>36</td>\n",
       "      <td>120</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>59</td>\n",
       "      <td>368</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>29</td>\n",
       "      <td>256</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>26</td>\n",
       "      <td>449</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>30</td>\n",
       "      <td>126</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>50</td>\n",
       "      <td>574</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>33</td>\n",
       "      <td>498</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   age  duration  campaign  previous\n",
       "0   49       227         4         0\n",
       "1   37       202         2         1\n",
       "2   78      1148         1         0\n",
       "3   36       120         2         0\n",
       "4   59       368         2         0\n",
       "5   29       256         2         0\n",
       "6   26       449         1         0\n",
       "7   30       126         2         0\n",
       "8   50       574         1         0\n",
       "9   33       498         5         0"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_numeric.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "a8688c55",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "job            12\n",
       "marital         4\n",
       "education       8\n",
       "default         3\n",
       "housing         3\n",
       "loan            3\n",
       "contact         2\n",
       "month          10\n",
       "day_of_week     5\n",
       "poutcome        3\n",
       "y               2\n",
       "dtype: int64"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# number of unique values for eac categorical feature\n",
    "df_categorical.nunique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "8ae57229",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ploting bar charts for each categorical feature\n",
    "for i in df_categorical.columns:\n",
    "    plt.figure()\n",
    "    plt.title('{}'.format(i))\n",
    "    df.groupby(i)[i].count().plot(kind= 'bar')\n",
    "    plt.xticks(rotation= 45);"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e5f8acc8",
   "metadata": {},
   "source": [
    "###  <ins> Creating dummy variables </ins>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "f9f4bee6",
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
       "      <th>job_admin.</th>\n",
       "      <th>job_blue-collar</th>\n",
       "      <th>job_entrepreneur</th>\n",
       "      <th>job_housemaid</th>\n",
       "      <th>job_management</th>\n",
       "      <th>job_retired</th>\n",
       "      <th>job_self-employed</th>\n",
       "      <th>job_services</th>\n",
       "      <th>job_student</th>\n",
       "      <th>job_technician</th>\n",
       "      <th>...</th>\n",
       "      <th>month_sep</th>\n",
       "      <th>day_of_week_fri</th>\n",
       "      <th>day_of_week_mon</th>\n",
       "      <th>day_of_week_thu</th>\n",
       "      <th>day_of_week_tue</th>\n",
       "      <th>day_of_week_wed</th>\n",
       "      <th>poutcome_failure</th>\n",
       "      <th>poutcome_nonexistent</th>\n",
       "      <th>poutcome_success</th>\n",
       "      <th>y_yes</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 54 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   job_admin.  job_blue-collar  job_entrepreneur  job_housemaid  \\\n",
       "0           0                1                 0              0   \n",
       "1           0                0                 1              0   \n",
       "2           0                0                 0              0   \n",
       "3           1                0                 0              0   \n",
       "4           0                0                 0              0   \n",
       "\n",
       "   job_management  job_retired  job_self-employed  job_services  job_student  \\\n",
       "0               0            0                  0             0            0   \n",
       "1               0            0                  0             0            0   \n",
       "2               0            1                  0             0            0   \n",
       "3               0            0                  0             0            0   \n",
       "4               0            1                  0             0            0   \n",
       "\n",
       "   job_technician  ...  month_sep  day_of_week_fri  day_of_week_mon  \\\n",
       "0               0  ...          0                0                0   \n",
       "1               0  ...          0                0                0   \n",
       "2               0  ...          0                0                1   \n",
       "3               0  ...          0                0                1   \n",
       "4               0  ...          0                0                0   \n",
       "\n",
       "   day_of_week_thu  day_of_week_tue  day_of_week_wed  poutcome_failure  \\\n",
       "0                0                0                1                 0   \n",
       "1                0                0                1                 1   \n",
       "2                0                0                0                 0   \n",
       "3                0                0                0                 0   \n",
       "4                0                1                0                 0   \n",
       "\n",
       "   poutcome_nonexistent  poutcome_success  y_yes  \n",
       "0                     1                 0      0  \n",
       "1                     0                 0      0  \n",
       "2                     1                 0      1  \n",
       "3                     1                 0      0  \n",
       "4                     1                 0      0  \n",
       "\n",
       "[5 rows x 54 columns]"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# dummy variables\n",
    "columns = df_categorical.columns\n",
    "df_encoded = pd.get_dummies(df_categorical[columns])\n",
    "df_encoded.drop(columns= 'y_no', inplace= True)\n",
    "df_encoded.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "578b8bf1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid',\n",
       "       'job_management', 'job_retired', 'job_self-employed', 'job_services',\n",
       "       'job_student', 'job_technician', 'job_unemployed', 'job_unknown',\n",
       "       'marital_divorced', 'marital_married', 'marital_single',\n",
       "       'marital_unknown', 'education_basic.4y', 'education_basic.6y',\n",
       "       'education_basic.9y', 'education_high.school', 'education_illiterate',\n",
       "       'education_professional.course', 'education_university.degree',\n",
       "       'education_unknown', 'default_no', 'default_unknown', 'default_yes',\n",
       "       'housing_no', 'housing_unknown', 'housing_yes', 'loan_no',\n",
       "       'loan_unknown', 'loan_yes', 'contact_cellular', 'contact_telephone',\n",
       "       'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun',\n",
       "       'month_mar', 'month_may', 'month_nov', 'month_oct', 'month_sep',\n",
       "       'day_of_week_fri', 'day_of_week_mon', 'day_of_week_thu',\n",
       "       'day_of_week_tue', 'day_of_week_wed', 'poutcome_failure',\n",
       "       'poutcome_nonexistent', 'poutcome_success', 'y_yes'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_encoded.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "6ab0e5ca",
   "metadata": {},
   "outputs": [],
   "source": [
    "# droping redundat columns\n",
    "redundant_cols = ['job_unknown', 'marital_unknown', 'education_unknown', 'default_unknown', 'housing_unknown',\n",
    "                 'loan_unknown', 'contact_telephone', 'poutcome_nonexistent']\n",
    "df_encoded.drop(columns= redundant_cols, axis= 1, inplace= True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "f0515810",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(32942, 46)"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_encoded.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "fd979544",
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
       "      <th>age</th>\n",
       "      <th>duration</th>\n",
       "      <th>campaign</th>\n",
       "      <th>previous</th>\n",
       "      <th>job_admin.</th>\n",
       "      <th>job_blue-collar</th>\n",
       "      <th>job_entrepreneur</th>\n",
       "      <th>job_housemaid</th>\n",
       "      <th>job_management</th>\n",
       "      <th>job_retired</th>\n",
       "      <th>...</th>\n",
       "      <th>month_oct</th>\n",
       "      <th>month_sep</th>\n",
       "      <th>day_of_week_fri</th>\n",
       "      <th>day_of_week_mon</th>\n",
       "      <th>day_of_week_thu</th>\n",
       "      <th>day_of_week_tue</th>\n",
       "      <th>day_of_week_wed</th>\n",
       "      <th>poutcome_failure</th>\n",
       "      <th>poutcome_success</th>\n",
       "      <th>y_yes</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>49</td>\n",
       "      <td>227</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>37</td>\n",
       "      <td>202</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>78</td>\n",
       "      <td>1148</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>36</td>\n",
       "      <td>120</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>59</td>\n",
       "      <td>368</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 50 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   age  duration  campaign  previous  job_admin.  job_blue-collar  \\\n",
       "0   49       227         4         0           0                1   \n",
       "1   37       202         2         1           0                0   \n",
       "2   78      1148         1         0           0                0   \n",
       "3   36       120         2         0           1                0   \n",
       "4   59       368         2         0           0                0   \n",
       "\n",
       "   job_entrepreneur  job_housemaid  job_management  job_retired  ...  \\\n",
       "0                 0              0               0            0  ...   \n",
       "1                 1              0               0            0  ...   \n",
       "2                 0              0               0            1  ...   \n",
       "3                 0              0               0            0  ...   \n",
       "4                 0              0               0            1  ...   \n",
       "\n",
       "   month_oct  month_sep  day_of_week_fri  day_of_week_mon  day_of_week_thu  \\\n",
       "0          0          0                0                0                0   \n",
       "1          0          0                0                0                0   \n",
       "2          0          0                0                1                0   \n",
       "3          0          0                0                1                0   \n",
       "4          0          0                0                0                0   \n",
       "\n",
       "   day_of_week_tue  day_of_week_wed  poutcome_failure  poutcome_success  y_yes  \n",
       "0                0                1                 0                 0      0  \n",
       "1                0                1                 1                 0      0  \n",
       "2                0                0                 0                 0      1  \n",
       "3                0                0                 0                 0      0  \n",
       "4                1                0                 0                 0      0  \n",
       "\n",
       "[5 rows x 50 columns]"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# cocatinate the numeric and categorigal dataframes in on dataframe\n",
    "df = pd.concat([df_numeric, df_encoded], axis= 1)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "4b5d1016",
   "metadata": {},
   "outputs": [],
   "source": [
    "# correlation coesfficients \n",
    "sns.heatmap(df.corr());"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "604f724c",
   "metadata": {},
   "source": [
    "#### <span style=\"color:red\"> As shown from the heat map, there are some higly correlated features, which could create some noise.</span>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b3777b16",
   "metadata": {},
   "source": [
    "### <ins> Seperating Variables from our target variable. </ins>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "8cb286ef",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df.iloc[: , : -1].values\n",
    "y = df.iloc[:, -1].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "9c1822ac",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((32942, 49), (32942,))"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X.shape, y.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "3b8163b2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[  49,  227,    4, ...,    1,    0,    0],\n",
       "       [  37,  202,    2, ...,    1,    1,    0],\n",
       "       [  78, 1148,    1, ...,    0,    0,    0],\n",
       "       ...,\n",
       "       [  54,  131,    4, ...,    0,    0,    0],\n",
       "       [  29,  165,    1, ...,    0,    0,    0],\n",
       "       [  35,  544,    3, ...,    0,    0,    0]])"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2812cadc",
   "metadata": {},
   "source": [
    "### <ins> Scaling numerical values. </ins>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "4740f1f3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# It is useful in classification problems to normalize numerical features\n",
    "# standard scalling based on min. and max. values because most of the features we have are hot encoded.\n",
    "from sklearn.preprocessing import MinMaxScaler \n",
    "columns_to_normalize = [0, 1, 2]\n",
    "# creating scalar\n",
    "scaler = MinMaxScaler()\n",
    "# tranforming data\n",
    "X[:, columns_to_normalize] = scaler.fit_transform(X[:, columns_to_normalize])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "a5e34edb",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0, 0, 0, ..., 1, 0, 0],\n",
       "       [0, 0, 0, ..., 1, 1, 0],\n",
       "       [0, 0, 0, ..., 0, 0, 0],\n",
       "       ...,\n",
       "       [0, 0, 0, ..., 0, 0, 0],\n",
       "       [0, 0, 0, ..., 0, 0, 0],\n",
       "       [0, 0, 0, ..., 0, 0, 0]])"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b4a1048b",
   "metadata": {},
   "source": [
    "### <ins> Applying Gaussian PCA to reduce dimensions. </ins>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "6bd24b10",
   "metadata": {},
   "outputs": [],
   "source": [
    "A = X.copy()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9520e96e",
   "metadata": {},
   "source": [
    "from sklearn.decomposition import KernelPCA\n",
    "###### Define the RBF kernel PCA with the desired gamma (kernel width)\n",
    "gamma = 15\n",
    "kpca = KernelPCA(kernel='rbf', gamma=gamma, n_components=None)\n",
    "\n",
    "###### Fit the kernel PCA model to the data\n",
    "X_kpca = kpca.fit_transform(A)\n",
    "\n",
    "###### Compute the explained variance ratio\n",
    "explained_variance_ratio = kpca.explained_variance_ratio_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "85396c71",
   "metadata": {},
   "outputs": [],
   "source": [
    "# import PCA class\n",
    "from sklearn.decomposition import PCA\n",
    "# creating pca object \n",
    "pca = PCA(n_components= None)\n",
    "A = pca.fit_transform(A)\n",
    "explained_variance = pca.explained_variance_ratio_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "426cef20",
   "metadata": {},
   "outputs": [],
   "source": [
    "explained_variance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f788668d",
   "metadata": {},
   "outputs": [],
   "source": [
    "cumulative_variance = np.cumsum(explained_variance)\n",
    "# for noise reduction purpose\n",
    "# Find the number of components needed to explain at least 95% of the variance\n",
    "n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e975f6df",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "print(f\"Number of components to explain 95% of variance: {n_components_95}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0adb4891",
   "metadata": {},
   "outputs": [],
   "source": [
    "# show explained variance vs # of components\n",
    "plt.figure()\n",
    "x= np.arange(len(explained_variance))\n",
    "plt.plot(explained_variance)\n",
    "fill_value= n_components_95\n",
    "plt.fill_between(x, explained_variance, where=(x <= fill_value), alpha=0.5, color='grey', label='Fill Area')\n",
    "plt.xlabel('n-components')\n",
    "plt.ylabel('explined variance');"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "29645a6b",
   "metadata": {},
   "outputs": [],
   "source": [
    "B = X.copy()\n",
    "# for data Visualization purpose\n",
    "pca = PCA(n_components= 2)\n",
    "B = pca.fit_transform(B)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fe4c0d7a",
   "metadata": {},
   "outputs": [],
   "source": [
    "B"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cd4010d9",
   "metadata": {},
   "outputs": [],
   "source": [
    "# plot PC1 vs PC2\n",
    "plt.scatter(B[:, 0], B[:, 1])\n",
    "plt.xlabel('PC1')\n",
    "plt.ylabel('PC2');"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "82387b9e",
   "metadata": {},
   "source": [
    "#### <span style=\"color:red\"> Based on the graph, we think in SVC as a model .</span>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d399316e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# reduce the dimensions to n_components that can explain 95% of the data variance\n",
    "# creating pca object \n",
    "pca = PCA(n_components= n_components_95)\n",
    "X = pca.fit_transform(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "443ac438",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "X.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "043ef190",
   "metadata": {},
   "outputs": [],
   "source": [
    "X"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1b79639a",
   "metadata": {},
   "source": [
    "# Modeling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "747871ab",
   "metadata": {},
   "outputs": [],
   "source": [
    "# splitting the data into train-set and test-set\n",
    "from sklearn.model_selection import train_test_split\n",
    "# Split the data into training and testing sets (e.g., 80% train and 20% test)\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "644d65f8",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_copy1 = X_train.copy()\n",
    "y_train_copy1 = y_train.copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5a4eee12",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_copy1"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ac45fa6f",
   "metadata": {},
   "source": [
    "### <ins> Logistic Regression Model </ins>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "788c91f8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import statsmodels.api as sm\n",
    "\n",
    "# building the model and fitting the data\n",
    "log_reg = sm.Logit(y_train_copy1, X_train_copy1).fit()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6eb21d87",
   "metadata": {},
   "outputs": [],
   "source": [
    "log_reg.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "718ea0aa",
   "metadata": {},
   "outputs": [],
   "source": [
    "# predicting x_test values \n",
    "# cm between y_predicted and y_test\n",
    "# drop nonsignificant features \n",
    "# then fit, predict, cm , compare"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3ba67bab",
   "metadata": {},
   "source": [
    "### <ins> Applying logistic Regression before eliminating non-significant variables. </ins>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7306b814",
   "metadata": {},
   "outputs": [],
   "source": [
    "# performing predictions on the test dataset\n",
    "yhat = log_reg.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d9b25769",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Building LogisticRegression model using sklearn package\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "659995f0",
   "metadata": {},
   "outputs": [],
   "source": [
    "# making classifier object from a class\n",
    "logistic_regression = LogisticRegression()\n",
    "logistic_regression.fit(X_train_copy1, y_train_copy1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "740ef67c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# predict x_test values\n",
    "y_pred = logistic_regression.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4d191e9a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# compare y_pred, y_test\n",
    "y_pred, y_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4525eb39",
   "metadata": {},
   "outputs": [],
   "source": [
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "confusion = confusion_matrix(y_test, y_pred)\n",
    "classification_rep = classification_report(y_test, y_pred)\n",
    "\n",
    "print(f'Accuracy: {accuracy}')\n",
    "print(f'Confusion Matrix:\\n{confusion}')\n",
    "print(f'Classification Report:\\n{classification_rep}')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7f5c2b27",
   "metadata": {},
   "source": [
    "### <ins> Applying logistic Regression after eliminating non-significant variables. </ins>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2be8dff5",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_copy2 = X_train.copy()\n",
    "y_train_copy2 = y_train.copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9eab0748",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_copy2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0510f506",
   "metadata": {},
   "outputs": [],
   "source": [
    "# indices for variables with low significant effect on the target variable\n",
    "non_significant_variables_indices = [5,7,10,13,14]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6a0b5b51",
   "metadata": {},
   "outputs": [],
   "source": [
    "# remove those variables\n",
    "X_train_del = np.delete(X_train_copy2, non_significant_variables_indices, 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e542bdf6",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_copy2.shape, X_train_del.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "96ab28f3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# making classifier object from a class\n",
    "logistic_regression2 = LogisticRegression()\n",
    "logistic_regression2.fit(X_train_del, y_train_copy1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "09db8a23",
   "metadata": {},
   "outputs": [],
   "source": [
    "# predict x_test values\n",
    "y_pred2 = logistic_regression.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dd60ae75",
   "metadata": {},
   "outputs": [],
   "source": [
    "# compare y_pred, y_test\n",
    "y_pred2, y_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "36647e79",
   "metadata": {},
   "outputs": [],
   "source": [
    "accuracy2 = accuracy_score(y_test, y_pred2)\n",
    "confusion2 = confusion_matrix(y_test, y_pred2)\n",
    "classification_rep2 = classification_report(y_test, y_pred2)\n",
    "\n",
    "print(f'Accuracy: {accuracy2}')\n",
    "print(f'Confusion Matrix:\\n{confusion2}')\n",
    "print(f'Classification Report:\\n{classification_rep2}')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "00a750d5",
   "metadata": {},
   "source": [
    "### <ins> Artificial NuralNetwork(ANN) </ins>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bf4d50ae",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b0c9ca31",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_copy3 = X_train.copy()\n",
    "y_train_copy3 = y_train.copy()\n",
    "X_train_copy3.shape, y_train_copy3.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6eaeb510",
   "metadata": {},
   "outputs": [],
   "source": [
    "# required libraries\n",
    "import tensorflow as tf \n",
    "import keras\n",
    "from keras.models import Sequential\n",
    "from keras.layers import Dense\n",
    "from tensorflow.keras.callbacks import History\n",
    "from tensorflow.keras.utils import plot_model "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a9f0d175",
   "metadata": {},
   "source": [
    "#### <span style=\"color:red\"> Try without eliminating nonsignificant values..</span>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e5685de3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initializing ANN model\n",
    "model = Sequential()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e59d4a40",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Input layer (specify input_dim as the number of features)\n",
    "model.add(Dense(units=64, activation='relu', input_dim=X_train_copy3.shape[1]))\n",
    "\n",
    "# Hidden layers\n",
    "model.add(Dense(units=32, activation='relu'))\n",
    "\n",
    "# Hidden layers\n",
    "#model.add(Dense(units=16, activation='relu'))\n",
    "\n",
    "# Output layer (use 'sigmoid' activation for binary classification)\n",
    "model.add(Dense(units=1, activation='sigmoid'))\n",
    "\n",
    "# Compile the model\n",
    "model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])\n",
    "\n",
    "# Define a history object to store training metrics\n",
    "history = History()\n",
    "\n",
    "# Train the model\n",
    "model.fit(X_train_copy3, y_train_copy3, epochs=10, batch_size= 100,validation_data=(X_test, y_test), callbacks=[history])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c9373330",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot training and validation loss over epochs\n",
    "plt.plot(history.history['loss'], label='Training Loss')\n",
    "plt.plot(history.history['val_loss'], label='Validation Loss')\n",
    "plt.title('Training and Validation Loss')\n",
    "plt.xlabel('Epochs')\n",
    "plt.ylabel('Loss')\n",
    "plt.legend();"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f50a6e08",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Make predictions on the test set\n",
    "ANN_y_pred = model.predict(X_test)\n",
    "ANN_y_pred = (y_pred > 0.5)  # Convert probabilities to binary predictions\n",
    "\n",
    "# Evaluate the model\n",
    "ANN_confusion = confusion_matrix(y_test, ANN_y_pred)\n",
    "ANN_accuracy = accuracy_score(y_test, ANN_y_pred)\n",
    "print(f'Accuracy: {ANN_accuracy}')\n",
    "print(\"Confusion Matrix:\\n\", ANN_confusion)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5de30b15",
   "metadata": {},
   "source": [
    "### <ins> Support Vector Machine Model </ins>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2f424e0e",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_copy4 = X_train.copy()\n",
    "y_train_copy4 = y_train.copy()\n",
    "X_train_copy4.shape, y_train_copy4.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "beb39ba6",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.svm import SVC\n",
    "\n",
    "# Create an SVC model with an RBF kernel (you can choose a different kernel)\n",
    "svc_model = SVC(kernel='rbf', C=1.0)\n",
    "\n",
    "# Train the model\n",
    "svc_model.fit(X_train_copy4, y_train_copy4)\n",
    "\n",
    "# Make predictions on the test set\n",
    "SVC_y_pred = svc_model.predict(X_test)\n",
    "\n",
    "# Evaluate the model\n",
    "SVC_accuracy = accuracy_score(y_test, SVC_y_pred)\n",
    "SVC_confusion = confusion_matrix(y_test, SVC_y_pred)\n",
    "\n",
    "print(\"Accuracy:\", SVC_accuracy)\n",
    "print(\"Confusion Matrix:\\n\", SVC_confusion)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "118c2f33",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# comparing confusion matrix\n",
    "plt.figure()\n",
    "plt.subplot(2,2,1)\n",
    "plt.title('Logistic Regression_1')\n",
    "sns.heatmap(confusion, annot= True)\n",
    "\n",
    "plt.subplot(2,2,2)\n",
    "plt.title('Logistic Regression_2')\n",
    "sns.heatmap(confusion2, annot= True)\n",
    "\n",
    "plt.subplot(2,2,3)\n",
    "plt.title('ANN')\n",
    "sns.heatmap(ANN_confusion, annot= True)\n",
    "\n",
    "plt.subplot(2,2,4)\n",
    "plt.title('SVC')\n",
    "sns.heatmap(SVC_confusion, annot= True);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dcab6902",
   "metadata": {},
   "outputs": [],
   "source": [
    "# comparing accuracy between models\n",
    "plt.figure()\n",
    "models_name = [\"LogisticRegression_1\", \"LogisticRegression_2\", \"ANN\", \"SVC\"]\n",
    "values = [accuracy, accuracy2, ANN_accuracy, SVC_accuracy]\n",
    "plt.bar(models_name, values)\n",
    "plt.ylabel('Accuracy')\n",
    "plt.xticks(rotation= 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "58180601",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
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
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "52dfa13d-9168-4b53-8586-c974a05b8d0f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: mysql-connector-python in c:\\users\\aman chauhan\\appdata\\roaming\\python\\python312\\site-packages (9.6.0)\n"
     ]
    }
   ],
   "source": [
    "!pip install mysql-connector-python --user"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "cea4dd08-1fbc-486e-999e-b157b335c782",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "MySQL connector installed successfully\n"
     ]
    }
   ],
   "source": [
    "import mysql.connector\n",
    "print(\"MySQL connector installed successfully\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "a7880b81-f6f5-42e7-810e-ab0b7edc02e5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Connected successfully ✅\n"
     ]
    }
   ],
   "source": [
    "import mysql.connector\n",
    "\n",
    "mydb = mysql.connector.connect(\n",
    "    host=\"localhost\",\n",
    "    user=\"root\",\n",
    "    password=\"twinkle12\"\n",
    ")\n",
    "\n",
    "print(\"Connected successfully ✅\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "97a34c20-4520-4d5d-baee-a0cba88bdb3a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Student_ID Student_Age     Sex High_School_Type Scholarship Additional_Work  \\\n",
      "0   STUDENT1       19-22    Male            Other         50%             Yes   \n",
      "1   STUDENT2       19-22    Male            Other         50%             Yes   \n",
      "2   STUDENT3       19-22    Male            State         50%              No   \n",
      "3   STUDENT4          18  Female          Private         50%             Yes   \n",
      "4   STUDENT5       19-22    Male          Private         50%              No   \n",
      "\n",
      "  Sports_activity Transportation  Weekly_Study_Hours Attendance Reading Notes  \\\n",
      "0              No        Private                   0     Always     Yes   Yes   \n",
      "1              No        Private                   0     Always     Yes    No   \n",
      "2              No        Private                   2      Never      No    No   \n",
      "3              No            Bus                   2     Always      No   Yes   \n",
      "4              No            Bus                  12     Always     Yes    No   \n",
      "\n",
      "  Listening_in_Class Project_work Grade  \n",
      "0                 No           No    AA  \n",
      "1                Yes          Yes    AA  \n",
      "2                 No          Yes    AA  \n",
      "3                 No           No    AA  \n",
      "4                Yes          Yes    AA  \n",
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 145 entries, 0 to 144\n",
      "Data columns (total 15 columns):\n",
      " #   Column              Non-Null Count  Dtype \n",
      "---  ------              --------------  ----- \n",
      " 0   Student_ID          145 non-null    object\n",
      " 1   Student_Age         145 non-null    object\n",
      " 2   Sex                 145 non-null    object\n",
      " 3   High_School_Type    145 non-null    object\n",
      " 4   Scholarship         144 non-null    object\n",
      " 5   Additional_Work     145 non-null    object\n",
      " 6   Sports_activity     145 non-null    object\n",
      " 7   Transportation      145 non-null    object\n",
      " 8   Weekly_Study_Hours  145 non-null    int64 \n",
      " 9   Attendance          145 non-null    object\n",
      " 10  Reading             145 non-null    object\n",
      " 11  Notes               145 non-null    object\n",
      " 12  Listening_in_Class  145 non-null    object\n",
      " 13  Project_work        145 non-null    object\n",
      " 14  Grade               145 non-null    object\n",
      "dtypes: int64(1), object(14)\n",
      "memory usage: 17.1+ KB\n",
      "None\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "# Load CSV file\n",
    "df = pd.read_csv(\"Students Performance -Copy1.csv\")\n",
    "\n",
    "print(df.head())\n",
    "print(df.info())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "3de349eb-f6a4-479a-a2be-376c4f47661d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Index(['Student_ID', 'Student_Age', 'Sex', 'High_School_Type', 'Scholarship',\n",
      "       'Additional_Work', 'Sports_activity', 'Transportation',\n",
      "       'Weekly_Study_Hours', 'Attendance', 'Reading', 'Notes',\n",
      "       'Listening_in_Class', 'Project_work', 'Grade'],\n",
      "      dtype='object')\n"
     ]
    }
   ],
   "source": [
    "print(df.columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "7308feba-a1a6-434c-86cf-1416f82e898f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Table recreated successfully ✅\n"
     ]
    }
   ],
   "source": [
    "from sqlalchemy import create_engine\n",
    "\n",
    "engine = create_engine(\n",
    "    \"mysql+mysqlconnector://root:twinkle12@localhost/student_ml\"\n",
    ")\n",
    "\n",
    "# Replace old table completely\n",
    "df.to_sql(\"students\", engine, if_exists=\"replace\", index=False)\n",
    "\n",
    "print(\"Table recreated successfully ✅\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "38476ff3-3259-4086-b169-5e67dce5e31f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "object\n"
     ]
    }
   ],
   "source": [
    "print(df[\"Grade\"].dtype)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "e74843d7-5a27-40b1-8d89-ff5e7532bfb6",
   "metadata": {},
   "outputs": [],
   "source": [
    "df[\"Grade\"] = pd.to_numeric(df[\"Grade\"], errors=\"coerce\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "42a9c4d9-93fa-4eaf-b8ef-552187075d23",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "float64\n"
     ]
    }
   ],
   "source": [
    "print(df[\"Grade\"].dtype)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "7343285f-c831-46ee-a278-a307f96eac08",
   "metadata": {},
   "outputs": [],
   "source": [
    "df[\"result\"] = df[\"Grade\"].apply(lambda x: 1 if x >= 2 else 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "09799ba7-f61e-4b60-91d1-a748c658a7bd",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_encoded = pd.get_dummies(df, drop_first=True)\n",
    "\n",
    "X = df_encoded.drop([\"Grade\", \"result\"], axis=1)\n",
    "y = df_encoded[\"result\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "15b76bab-6afe-4e82-9637-c8226b49457f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0    AA\n",
      "1    AA\n",
      "2    AA\n",
      "3    AA\n",
      "4    AA\n",
      "Name: Grade, dtype: object\n",
      "Grade\n",
      "AA      35\n",
      "BA      24\n",
      "BB      21\n",
      "CC      17\n",
      "DD      17\n",
      "DC      13\n",
      "CB      10\n",
      "Fail     8\n",
      "Name: count, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "df = pd.read_csv(\"Students Performance -Copy1.csv\")\n",
    "\n",
    "print(df[\"Grade\"].head())\n",
    "print(df[\"Grade\"].value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "585b5649-65c4-4074-93ec-98bb13365558",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fresh table created ✅\n"
     ]
    }
   ],
   "source": [
    "from sqlalchemy import create_engine\n",
    "\n",
    "engine = create_engine(\n",
    "    \"mysql+mysqlconnector://root:twinkle12@localhost/student_ml\"\n",
    ")\n",
    "\n",
    "df.to_sql(\"students\", engine, if_exists=\"replace\", index=False)\n",
    "\n",
    "print(\"Fresh table created ✅\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "290ffa1b-bf10-4810-ab7e-a45d4ea1acb0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Grade\n",
      "AA      35\n",
      "BA      24\n",
      "BB      21\n",
      "CC      17\n",
      "DD      17\n",
      "DC      13\n",
      "CB      10\n",
      "Fail     8\n",
      "Name: count, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "df = pd.read_sql(\"SELECT * FROM students\", engine)\n",
    "\n",
    "print(df[\"Grade\"].value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "e4a36e9f-f63e-43b9-ade5-e5f59cd2ffb5",
   "metadata": {},
   "outputs": [],
   "source": [
    "df[\"Grade\"] = df[\"Grade\"].str.strip().str.upper()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "73710e0d-bd02-4cd8-91e3-0cc69ead115b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "result\n",
      "1    120\n",
      "0     25\n",
      "Name: count, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "pass_grades = [\"AA\", \"BA\", \"BB\", \"CB\", \"CC\", \"DC\"]\n",
    "\n",
    "df[\"result\"] = df[\"Grade\"].apply(\n",
    "    lambda x: 1 if x in pass_grades else 0\n",
    ")\n",
    "\n",
    "print(df[\"result\"].value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "c05eb6fa-cf94-44d7-b2c5-bf4c296a488c",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_encoded = pd.get_dummies(df, drop_first=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "89b33c56-08b2-47f2-84ee-a98be6583361",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Remove all columns that start with \"Grade_\"\n",
    "grade_cols = [col for col in df_encoded.columns if col.startswith(\"Grade_\")]\n",
    "\n",
    "X = df_encoded.drop([\"result\"] + grade_cols, axis=1)\n",
    "y = df_encoded[\"result\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "a67d735b-81e3-4345-9922-fd11eea15080",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.7931034482758621\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.2, random_state=42\n",
    ")\n",
    "\n",
    "model = LogisticRegression(max_iter=1000)\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "y_pred = model.predict(X_test)\n",
    "\n",
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "bab40d8f-0beb-4534-a024-3171d3755bfd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 0  6]\n",
      " [ 0 23]]\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix\n",
    "\n",
    "cm = confusion_matrix(y_test, y_pred)\n",
    "print(cm)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "60fdf219-0e95-4a72-8983-ed1e68313f37",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.00      0.00      0.00         6\n",
      "           1       0.79      1.00      0.88        23\n",
      "\n",
      "    accuracy                           0.79        29\n",
      "   macro avg       0.40      0.50      0.44        29\n",
      "weighted avg       0.63      0.79      0.70        29\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Aman chauhan\\OneDrive\\Desktop\\New folder\\Lib\\site-packages\\sklearn\\metrics\\_classification.py:1531: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.\n",
      "  _warn_prf(average, modifier, f\"{metric.capitalize()} is\", len(result))\n",
      "C:\\Users\\Aman chauhan\\OneDrive\\Desktop\\New folder\\Lib\\site-packages\\sklearn\\metrics\\_classification.py:1531: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.\n",
      "  _warn_prf(average, modifier, f\"{metric.capitalize()} is\", len(result))\n",
      "C:\\Users\\Aman chauhan\\OneDrive\\Desktop\\New folder\\Lib\\site-packages\\sklearn\\metrics\\_classification.py:1531: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.\n",
      "  _warn_prf(average, modifier, f\"{metric.capitalize()} is\", len(result))\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import classification_report\n",
    "\n",
    "print(classification_report(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "4934e812-fe03-4649-8772-1d5ab008b8fb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Random Forest Accuracy: 0.7931034482758621\n"
     ]
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "rf = RandomForestClassifier(random_state=42)\n",
    "rf.fit(X_train, y_train)\n",
    "\n",
    "rf_pred = rf.predict(X_test)\n",
    "\n",
    "print(\"Random Forest Accuracy:\", accuracy_score(y_test, rf_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "39a84fb4-6052-4b7b-b95d-1225f3045e92",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = model.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "18fe49d3-36ca-4cba-b3f1-856203196552",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get test indices\n",
    "test_indices = X_test.index\n",
    "\n",
    "# Add predictions to original df\n",
    "df.loc[test_indices, \"Predicted_Result\"] = y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "e332f75f-c590-44f9-ade8-f722afa5b84b",
   "metadata": {},
   "outputs": [],
   "source": [
    "student_id = df.loc[index, \"Student_ID\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "c3acdc57-2d4d-4ac3-b60c-c0f4ff4286cd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predictions stored in MySQL ✅\n"
     ]
    }
   ],
   "source": [
    "for index in test_indices:\n",
    "    pred = int(df.loc[index, \"Predicted_Result\"])\n",
    "    student_id = df.loc[index, \"Student_ID\"]   # remove int()\n",
    "\n",
    "    cursor.execute(\"\"\"\n",
    "        UPDATE students\n",
    "        SET Predicted_Result = %s\n",
    "        WHERE Student_ID = %s\n",
    "    \"\"\", (pred, student_id))\n",
    "\n",
    "conn.commit()\n",
    "print(\"Predictions stored in MySQL ✅\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "e1e149a2-4d39-4b4a-9fc9-b892fefaa426",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "-1"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.to_sql(\"students\", engine, if_exists=\"replace\", index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "501d8b0e-518a-42fd-a17c-458c9acdf32e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "All predictions stored ✅\n"
     ]
    }
   ],
   "source": [
    "df[\"Predicted_Result\"] = model.predict(X)\n",
    "\n",
    "for index in df.index:\n",
    "    pred = int(df.loc[index, \"Predicted_Result\"])\n",
    "    student_id = df.loc[index, \"Student_ID\"]\n",
    "\n",
    "    cursor.execute(\"\"\"\n",
    "        UPDATE students\n",
    "        SET Predicted_Result = %s\n",
    "        WHERE Student_ID = %s\n",
    "    \"\"\", (pred, student_id))\n",
    "\n",
    "conn.commit()\n",
    "print(\"All predictions stored ✅\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "34ebce0b-a197-4296-852a-b86afa1c2c4f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model saved successfully ✅\n"
     ]
    }
   ],
   "source": [
    "import pickle\n",
    "\n",
    "# Save model\n",
    "with open(\"student_model.pkl\", \"wb\") as f:\n",
    "    pickle.dump(model, f)\n",
    "\n",
    "print(\"Model saved successfully ✅\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "d675fa7a-c532-4d69-ad1d-b19982e686d9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model loaded successfully ✅\n"
     ]
    }
   ],
   "source": [
    "with open(\"student_model.pkl\", \"rb\") as f:\n",
    "    loaded_model = pickle.load(f)\n",
    "\n",
    "print(\"Model loaded successfully ✅\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "f2b9a0ef-df17-4efc-a2a0-a68de4b0e4f7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: streamlit in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (1.37.1)\n",
      "Requirement already satisfied: altair<6,>=4.0 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from streamlit) (5.0.1)\n",
      "Requirement already satisfied: blinker<2,>=1.0.0 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from streamlit) (1.6.2)\n",
      "Requirement already satisfied: cachetools<6,>=4.0 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from streamlit) (5.3.3)\n",
      "Requirement already satisfied: click<9,>=7.0 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from streamlit) (8.1.7)\n",
      "Requirement already satisfied: numpy<3,>=1.20 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from streamlit) (1.26.4)\n",
      "Requirement already satisfied: packaging<25,>=20 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from streamlit) (24.1)\n",
      "Requirement already satisfied: pandas<3,>=1.3.0 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from streamlit) (2.2.2)\n",
      "Requirement already satisfied: pillow<11,>=7.1.0 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from streamlit) (10.4.0)\n",
      "Requirement already satisfied: protobuf<6,>=3.20 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from streamlit) (4.25.3)\n",
      "Requirement already satisfied: pyarrow>=7.0 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from streamlit) (16.1.0)\n",
      "Requirement already satisfied: requests<3,>=2.27 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from streamlit) (2.32.3)\n",
      "Requirement already satisfied: rich<14,>=10.14.0 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from streamlit) (13.7.1)\n",
      "Requirement already satisfied: tenacity<9,>=8.1.0 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from streamlit) (8.2.3)\n",
      "Requirement already satisfied: toml<2,>=0.10.1 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from streamlit) (0.10.2)\n",
      "Requirement already satisfied: typing-extensions<5,>=4.3.0 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from streamlit) (4.11.0)\n",
      "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from streamlit) (3.1.43)\n",
      "Requirement already satisfied: pydeck<1,>=0.8.0b4 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from streamlit) (0.8.0)\n",
      "Requirement already satisfied: tornado<7,>=6.0.3 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from streamlit) (6.4.1)\n",
      "Requirement already satisfied: watchdog<5,>=2.1.5 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from streamlit) (4.0.1)\n",
      "Requirement already satisfied: jinja2 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from altair<6,>=4.0->streamlit) (3.1.4)\n",
      "Requirement already satisfied: jsonschema>=3.0 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from altair<6,>=4.0->streamlit) (4.23.0)\n",
      "Requirement already satisfied: toolz in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from altair<6,>=4.0->streamlit) (0.12.0)\n",
      "Requirement already satisfied: colorama in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from click<9,>=7.0->streamlit) (0.4.6)\n",
      "Requirement already satisfied: gitdb<5,>=4.0.1 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.7)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from pandas<3,>=1.3.0->streamlit) (2.9.0.post0)\n",
      "Requirement already satisfied: pytz>=2020.1 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from pandas<3,>=1.3.0->streamlit) (2024.1)\n",
      "Requirement already satisfied: tzdata>=2022.7 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from pandas<3,>=1.3.0->streamlit) (2023.3)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from requests<3,>=2.27->streamlit) (3.3.2)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from requests<3,>=2.27->streamlit) (3.7)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from requests<3,>=2.27->streamlit) (2.2.3)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from requests<3,>=2.27->streamlit) (2025.6.15)\n",
      "Requirement already satisfied: markdown-it-py>=2.2.0 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from rich<14,>=10.14.0->streamlit) (2.2.0)\n",
      "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from rich<14,>=10.14.0->streamlit) (2.15.1)\n",
      "Requirement already satisfied: smmap<5,>=3.0.1 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.0)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from jinja2->altair<6,>=4.0->streamlit) (2.1.3)\n",
      "Requirement already satisfied: attrs>=22.2.0 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (23.1.0)\n",
      "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2023.7.1)\n",
      "Requirement already satisfied: referencing>=0.28.4 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.30.2)\n",
      "Requirement already satisfied: rpds-py>=0.7.1 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.10.6)\n",
      "Requirement already satisfied: mdurl~=0.1 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from markdown-it-py>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.0)\n",
      "Requirement already satisfied: six>=1.5 in c:\\users\\aman chauhan\\onedrive\\desktop\\new folder\\lib\\site-packages (from python-dateutil>=2.8.2->pandas<3,>=1.3.0->streamlit) (1.16.0)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install streamlit"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "d0c55cb6-c272-46a5-bcda-7f522250df44",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-02-21 17:17:33.510 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\Aman chauhan\\OneDrive\\Desktop\\New folder\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2026-02-21 17:17:33.510 Session state does not function when running a script without `streamlit run`\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pickle\n",
    "import numpy as np\n",
    "\n",
    "# Load model\n",
    "with open(\"student_model.pkl\", \"rb\") as f:\n",
    "    model = pickle.load(f)\n",
    "\n",
    "st.title(\"🎓 Student Performance Prediction\")\n",
    "\n",
    "study_hours = st.number_input(\"Weekly Study Hours\", 0, 50)\n",
    "attendance = st.number_input(\"Attendance Percentage\", 0, 100)\n",
    "\n",
    "if st.button(\"Predict\"):\n",
    "    input_data = np.array([[study_hours, attendance]])\n",
    "    prediction = model.predict(input_data)\n",
    "\n",
    "    if prediction[0] == 1:\n",
    "        st.success(\"Prediction: PASS ✅\")\n",
    "    else:\n",
    "        st.error(\"Prediction: FAIL ❌\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "4cc0c0ac-265e-4509-bb1e-c3ebfd35ee63",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Usage: streamlit run [OPTIONS] TARGET [ARGS]...\n",
      "Try 'streamlit run --help' for help.\n",
      "\n",
      "Error: Invalid value: File does not exist: app.py\n"
     ]
    }
   ],
   "source": [
    "!streamlit run app.py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b4b8d16f-c240-40c3-84e7-e89f458eb794",
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
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
import streamlit as st
import pickle
import numpy as np

with open("student_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Student Performance Prediction")

study_hours = st.number_input("Weekly Study Hours", 0, 50)
attendance = st.number_input("Attendance Percentage", 0, 100)

if st.button("Predict"):
    input_data = np.array([[study_hours, attendance]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Prediction: PASS ✅")
    else:
        st.error("Prediction: FAIL ❌")
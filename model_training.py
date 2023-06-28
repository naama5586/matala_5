import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import statsmodels.formula.api as smf
import statsmodels.api as sm

data = pd.read_excel('output_all_students_Train_v10.xlsx')

def change_price_to_numbers (row):
    if isinstance(row['price'], int):
        return row['price']
    elif row['price'] == 'NaN':
        return None
    else:
        price = re.findall(r'(\d{1,3}(?:,\d{3})*)(?=\D|$)', str(row['price']))
        if isinstance(price,list) and len(price) > 0 :
            price = float(price[0].replace(',', ''))
            return price
        else:
            return None
data['price'] = data.apply(change_price_to_numbers, axis = 1)
data = data.dropna(subset=['price'])

def prepare_data(data):
    
    def change_area_to_numbers (row):
        if isinstance(row, int):
            return float(row)
        elif row == 'NaN' or row == 'None':
            return  None
        else:
            try:
                area = re.search(r"\d+", str(row))
                area_new = area.group()
                if int(area_new) < 500:
                    return float(area_new)
                else:
                    return None
            except: 
                None
    
    def change_room_number_to_numbers(row):
        if isinstance(row, int) or isinstance(row, float):
            return float(row)
        elif row == 'NaN' or row == 'None':
            return None
        else:
            try:
                room_number =re.findall(r'\d+\.*\d*', str(row))  # Modified regex pattern to capture decimal numbers
                rooms_new = rooms[0]
                if float(rooms_new) < 30:  # Convert the extracted area to a float
                    return float(rooms_new)  # Return decimal number
                else:
                    return (float(rooms_new))/10
            except: 
                return None


    def number_in_street_to_number(row):
        if isinstance(row, int):
            return float(row)
        else:
            try:
                number_in_street =re.findall(r'\d+\.*\d*', str(row))  # Modified regex pattern to capture decimal numbers
                new_number = number_in_street[0]
                return float(new_number)  # Return decimal number
            except: 
                return None

    def published_days_to_numbers(row):
        if isinstance(row, int) or isinstance(row, float):
            return float(row)
        else:
            if "חדש" in str(row):
                return 1
            else:
                try:
                    published_days =re.findall(r'\d+\.*\d*', str(row))  # Modified regex pattern to capture decimal numbers
                    new_number = published_days[0]
                    return float(new_number)  # Return decimal number
                except: 
                    return None

    def remove_punctuation (value):
        try:
            witout_pun = re.findall(r"[\w\d\s]+", str(value))
            result = ' '.join(witout_pun)
            return result
        except:
            return None

    def date_to_categorial(date):
        if 'גמיש' in str(date):
            return 'flexible'
        elif 'מיידי' in str(date):
            return 'less_than_6 months'
        elif str(date) in ['NaN', 'None','לא צויין']:
            return 'not_defined'
        else:
            date = pd.to_datetime(date)
            extract_date = pd.to_datetime('30/05/2023', format='%d/%m/%Y')
            #today_date = current_datetime.date()
            date_diff = date - extract_date
            month_diff = date_diff.days/30.4
            if month_diff < 6:
                return 'less_than_6 months'
            elif month_diff < 12:
                return 'months_6_12'
            else:
                return 'above_year'

    def floor (row):
        if row['floor_out_of'] in ['NaN', 'None']:
            if row['type'] in ['בית פרטי',"קוטג'","דירת גן","קוטג' טורי",'אחר','נחלה']:
                return 1
            else:
                return None
        elif row['floor_out_of'] == "קומת קרקע":
            return 1
        elif row['floor_out_of'] == "קומת מרתף":
            return 0
        else:
            numbers = re.findall(r"\d+", str(row['floor_out_of']))
            
            return float(numbers[0]) if numbers else None

    def total_floors (row):
        if row['floor_out_of'] in ['NaN', 'None',"קומת קרקע","קומת מרתף"]:
            if row['type'] in ['בית פרטי',"קוטג'","קוטג' טורי",'אחר','נחלה']:
                return 1
            else:
                return None
            return None
        else:
            numbers = re.findall(r"\d+", str(row['floor_out_of']))
            return float(numbers[1]) if len(numbers) > 1 else None   

    def make_bool (value):
        if 'יש' in str(value) or str(value) in ['TRUE','True','כן','1','yes','נגיש לנכים']:
            return 1
        elif 'אין' in str(value) or str(value) in ['FALSE','False','לא','0','no','לא נגיש לנכים']:
            return 0
        else:
            return None  

    def nahariya(value):
        if 'נהרי' in value:
            return 'נהריה'
        else:
            return value
    
    def has_furniture (val):
        if 'אין' in val:
            return 'No'
        elif 'מלא' in val:
            return 'Full'
        elif 'חלקי' in val:
            return 'Partial'
        else:
            return 'Not_mentioned'
    
    #data = pd.read_excel('lms_data.xlsx') #דאטה מתוך אתר הלמס שנמצא אצלי באקסל ולא רציתי לפגוע בהרצה שלכם אז הבאתי אותו לפה
    #city_ratings = dict(zip(data['city'], data['rank']))
    def rank_by_lms (value):
        city_ratings = {'אבו גוש': 67, 'אבו סנאן': 62, 'אבן יהודה': 243, 'אום אל-פחם': 20, 'אופקים': 72, 'אור יהודה': 137, 'אור עקיבא': 126, 'אורנית': 230, 
                        'אזור': 174, 'אילת': 149, 'אכסאל': 42, 'אל - בטוף': 21, 'אל קסום': 5, 'אלונה': 217, 'אליכין': 163, 'אלעד': 19, 'אלפי מנשה': 219, 'אלקנה': 209,
                        'אעבלין': 68, 'אפרת': 171, 'אריאל': 151, 'אשדוד': 108, 'אשכול': 160, 'אשקלון': 119, 'באקה אל-גרביה': 75, 'באר טוביה': 200, 'באר יעקב': 197, 
                        'באר שבע': 128, "בוסתן אל -מרג'": 45, "בועיינה-נוג'ידאת": 34, 'בוקעאתא': 38, 'ביר אל-מכסור': 22, 'בית אל': 76, 'בית אריה-עופרים': 201,
                            "בית ג'ן": 93, 'בית דגן': 175, 'בית שאן': 102, 'בית שמש': 18, 'ביתר עילית': 10, 'בני ברק': 14, 'בני עי"ש': 136, 'בני שמעון': 206,
                            'בנימינה-גבעת עדה': 229, 'בסמ"ה': 33, 'בסמת טבעון': 48, 'בענה': 36, 'ברנר': 227, 'בת ים': 112, "ג'דיידה-מכר": 50, "ג'ולס": 106, "ג'לג'וליה": 41,
                            "ג'סר א-זרקא": 15, "ג'ש (גוש חלב)": 168, "ג'ת": 121, 'גבעת זאב': 107, 'גבעת שמואל': 212, 'גבעתיים': 242, 'גדרה': 189, 'גדרות': 249, 'גולן': 148,
                            'גוש עציון': 144, 'גזר': 214, 'גן יבנה': 191, 'גן רווה': 233, 'גני תקווה': 239, 'דאלית אל-כרמל': 103, 'דבורייה': 86, 'דייר אל-אסד': 80,
                            'דייר חנא': 78, 'דימונה': 100, 'דרום השרון': 236, 'הגלבוע': 116, 'הגליל העליון': 176, 'הגליל התחתון': 172, 'הוד השרון': 245, 'הערבה התיכונה': 199,
                            'הר אדר': 250, 'הר חברון': 122, 'הרצליה': 225, 'זבולון': 135, 'זכרון יעקב': 215, 'זמר': 90, 'זרזיר': 35, 'חבל אילות': 131, 'חבל יבנה': 130, 
                            'חבל מודיעין': 202, 'חדרה': 150, 'חולון': 166, 'חוף אשקלון': 185, 'חוף הכרמל': 205, 'חוף השרון': 235, 'חורה': 7, 'חורפיש': 97, 'חיפה': 165, 
                            'חצור הגלילית': 89, 'חריש': 129, 'טבריה': 81, 'טובא-זנגרייה': 43, 'טורעאן': 54, 'טייבה': 63, 'טירה': 96, 'טירת כרמל': 134, 'טמרה': 49,
                            "יאנוח-ג'ת": 95, 'יבנאל': 28, 'יבנה': 170, 'יהוד': 208, 'יואב': 211, 'יסוד המעלה': 207, 'יפיע': 47, 'יקנעם עילית': 193, 'ירוחם': 87, 
                            'ירושלים': 31, 'ירכא': 66, 'כאבול': 55, "כאוכב אבו אל-היג'א": 91, 'כוכב יאיר': 251, 'כסיפה': 4, 'כסרא-סמיע': 65, "כעביה-טבאש-חג'אג'רה": 44, 
                            'כפר ברא': 84, 'כפר ורדים': 241, 'כפר יאסיף': 117, 'כפר יונה': 196, 'כפר כמא': 157, 'כפר כנא': 32, 'כפר מנדא': 16, 'כפר סבא': 222, 'כפר קאסם': 73,
                            'כפר קרע': 105, 'כפר שמריהו': 254, 'כפר תבור': 234, 'כרמיאל': 141, 'לב השרון': 220, 'להבים': 253, 'לוד': 82, 'לכיש': 177, 'לקיה': 9, 
                            'מבואות החרמון': 195, 'מבשרת ציון': 216, "מג'ד אל-כרום": 61, "מג'דל שמס": 57, 'מגאר': 56, 'מגדל': 140, 'מגדל העמק': 99, 'מגידו': 169,
                            'מגילות ים המלח': 187, 'מודיעין מכבים רעות': 237, 'מודיעין עילית': 6, 'מזכרת בתיה': 228, 'מזרעה': 88, 'מטה אשר': 173, 'מטה בנימין': 125,
                            'מטה יהודה': 192, 'מטולה': 203, 'מיתר': 244, 'מנשה': 181, 'מסעדה': 37, 'מעיליא': 198, 'מעלה אדומים': 159, 'מעלה אפרים': 85, 'מעלה יוסף': 186,
                            'מעלה עירון': 27, 'מעלות-תרשיחא': 124, 'מצפה רמון': 59, 'מרום הגליל': 133, 'מרחבים': 154, 'משגב': 179, 'משהד': 24, 'נהרייה': 152,
                            'נווה מדבר': 1, 'נוף הגליל': 109, 'נחל שורק': 120, 'נחף': 29, 'נס ציונה': 232, 'נצרת': 74, 'נשר': 182, 'נתיבות': 69, 'נתניה': 139, "סאג'ור": 83,
                            'סביון': 255, "סח'נין": 77, "ע'ג'ר": 40, 'עומר': 252, 'עיילבון': 123, 'עילוט': 17, 'עין מאהל': 25, 'עין קנייא': 52, 'עכו': 94, 'עמנואל': 12,
                            'עמק הירדן': 164, 'עמק המעיינות': 155, 'עמק חפר': 221, 'עמק יזרעאל': 213, 'עמק לוד': 146, 'עספיא': 101, 'עפולה': 104, 'עראבה': 51,
                            'ערבות הירדן': 161, 'ערד': 70, 'ערערה': 71, 'ערערה-בנגב': 2, 'פוריידיס': 53, 'פסוטה': 153, 'פקיעין (בוקייעה)': 118, 'פרדס חנה': 188,
                            'פרדסיה': 240, 'פתח תקווה': 178, 'צפת': 23, 'קדומים': 127, 'קדימה צורן': 226, 'קלנסווה': 58, 'קצרין': 110, 'קרית אונו': 246,
                            'קרית ארבע': 46, 'קריית אתא': 142, 'קרית ביאליק': 162, 'קרית גת': 92, 'קרית טבעון': 224, 'קרית ים': 111, 'קרית יערים': 26,
                            'קרית מוצקין': 167, 'קרית מלאכי': 79, 'קרית עקרון': 138, 'קרית שמונה': 115, 'קרני שומרון': 147, 'ראמה': 114, 'ראש העין': 194,
                            'ראש פינה': 180, 'ראשון לציון': 190, 'רהט': 11, 'רחובות': 183, 'ריינה': 39, 'רכסים': 13, 'רמלה': 98, 'רמת גן': 210, 'רמת השרון': 247, 'רמת ישי': 231, 
                            'רמת נגב': 158, 'רעננה': 223, 'שבלי - אום אל-גנם': 60, 'שגב-שלום': 8, 'שדות נגב': 145, 'שדרות': 113, 'שוהם': 248, 'שומרון': 156,
                        'שלומי': 143, 'שעב': 30, 'שער הנגב': 204, 'שפיר': 132, 'שפרעם': 64, 'תל אביב': 218, 'תל מונד': 238, 'תל שבע': 3, 'תמר': 184}
        try:
            rank= city_ratings.get(value)
            return float(rank)
        except:
            return None
    
    def descriotion_level (value):
        word_list = ['רכבת','מפואר','מבוקש','מתוחזק','השקעה','מניב','קרוב','מיקום','ליד','סמוך','פינוי בינוי','תמא','שקט','מואר',
                    'אוויר','אויר','משופ','התחדשות','ענק','מרווח','נוף','הורים','חכם','שמור','חדר כושר','חוף','גינה','בוטיק','תחבורה','כביש','מסחר','ספר','פארק'
                    ,'כנסת','חינוך','קניון','מרכז','ים','גדול','קניות','צמוד']
        if value in ['NaN','None']:
            return None
        else:
            level = 0
            for word in word_list:
                if word in value:
                    level += 1
            return float(level)

    def condition (value):
        if value in ['חדש','משופץ','דורש שיפוץ','ישן','שמור']:
            return value
        else:
            return 'לא צויין'

    data['Area'] = data['Area'].apply(change_area_to_numbers)
    data['room_number']= data['room_number'].apply(change_room_number_to_numbers)
    data['number_in_street']= data['number_in_street'].apply(number_in_street_to_number)
    data['publishedDays ']= data['publishedDays '].apply(published_days_to_numbers)
    data['description '] = data['description '].apply(remove_punctuation)
    data['Street']= data['Street'].apply(remove_punctuation)
    data['city_area'] = data['city_area'].apply(remove_punctuation)
    data['entranceDate ']= data['entranceDate '].apply(date_to_categorial)
    data['floor'] = data.apply(floor, axis= 1)
    data['total_floors'] = data.apply(total_floors, axis=1)
    data['hasElevator '] = data['hasElevator '].apply(make_bool)
    data['hasParking '] = data['hasParking '].apply(make_bool)
    data['hasBars '] = data['hasBars '].apply(make_bool)
    data['hasStorage '] = data['hasStorage '].apply(make_bool)
    data['hasAirCondition '] = data['hasAirCondition '].apply(make_bool)
    data['hasBalcony '] = data['hasBalcony '].apply(make_bool)
    data['hasMamad '] = data['hasMamad '].apply(make_bool)
    data['handicapFriendly '] = data['handicapFriendly '].apply(make_bool)
    data['City'] = data['City'].str.replace(' שוהם', 'שוהם')  
    data['City'] = data['City'].apply(nahariya)
    data['furniture '] = data['furniture '].apply(has_furniture)
    data['city_rank']= data['City'].apply(rank_by_lms)
    data['descriotion_level'] = data['description '].apply(descriotion_level)
    data['condition '] = data['condition '].apply(condition)
    return data

data = prepare_data(data)

#pip install mtranslate
from mtranslate import translate
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import  KFold
from sklearn.decomposition import PCA

def model_training(data):   
    median_imputer = SimpleImputer(strategy='median')
    mean_imputer = SimpleImputer(strategy='mean')
    most_frequent_imputer = SimpleImputer(strategy='most_frequent')
    median_columns = ['Area','num_of_images','number_in_street','descriotion_level','room_number','total_floors'
                    ,'hasElevator ', 'hasParking ', 'hasBars ','hasStorage ', 'hasAirCondition ', 'hasBalcony ', 'hasMamad ', 'handicapFriendly ']
    mean_columns = ['publishedDays ','floor','publishedDays ','city_rank']
    most_frequent_column = ['City','type','furniture ','condition ']
    data[median_columns] = median_imputer.fit_transform(data[median_columns]).astype(float)
    data[mean_columns] = mean_imputer.fit_transform(data[mean_columns]).astype(float)
    data[most_frequent_column] = most_frequent_imputer.fit_transform(data[most_frequent_column])
    data['description '] = data['description '].fillna('ללא תיאור')
    data = data.reset_index(drop=True)
    categorial_columns =  ['City', 'type', 'condition ', 'furniture ', 'entranceDate ']
    numerical_columns = ['Area','number_in_street','price', 'num_of_images', 'publishedDays ', 'floor', 'descriotion_level','city_rank','room_number','total_floors']

    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])


    for cat in categorial_columns:
        encoder = OneHotEncoder(drop = 'first',handle_unknown='ignore',sparse= False)
    #    data = data.drop(cat,axis = 1)
        encoder_fit = encoder.fit_transform(data[[cat]])
        one_hot = pd.DataFrame(encoder_fit, columns=encoder.get_feature_names_out([cat]))
        data = pd.concat([data,one_hot],axis=1)

    aa = ['Street','city_area','floor_out_of','description ']
    data = data.drop(categorial_columns, axis= 1)
    data= data.drop(aa,axis=1)



    def translate_columns(data):
        translated_columns = [translate(column, 'en').replace(' ', "_").replace("-", "_").replace("'", "") for column in data.columns]
        renamed_data = data.rename(columns=dict(zip(data.columns, translated_columns)))

        return renamed_data

    # Assuming model_data is your DataFrame with Hebrew column names

    # Translate the column names and replace spaces with underscores
    data = translate_columns(data)

    X = data.drop('price',axis = 1)
    y = data['price']

    def backward_elimination(X, y, significance_level=0.05):
        data = X.copy()
        data["target"] = y
        features = list(X.columns)
        while True:
            formula = "target ~ " + " + ".join(features)
            p_values = smf.ols(formula, data=data).fit().pvalues.iloc[1:]
            max_p_value = sorted(p_values, reverse=True)[0]

            if max_p_value > significance_level:
                excluded_feature = p_values[p_values == max_p_value].index[0]
                try:
                    features.remove(excluded_feature)
                except:
                    break
            else:
                break

        return features

    selected_features_backward = backward_elimination(X, y)
    X = X[selected_features_backward]
    

    pca = PCA(n_components=3)  
    principal_components = pca.fit_transform(X,y)

    # Explained variance ratio
    explained_var = pca.explained_variance_ratio_

    # Access the principal components and their loadings
    components = pca.components_
    pd.DataFrame(principal_components)
    
    model = ElasticNet()

    # Define the parameter grid
    param_grid = {
        'alpha': np.arange(0,0.0015, 0.0005),     # Generate 10 values between 0 and 1
        'l1_ratio': np.arange(0.2,0.4, 0.001)
    }

    # Perform grid search
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X, y)

    # Get the best parameters
    best_alpha = grid_search.best_params_['alpha']
    best_l1_ratio = grid_search.best_params_['l1_ratio']



    # חלוקת הדאטה לקבוצות קרוס ולידיישן
    #X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    #X = model_data[selected_features_backward]
    # יצירת מודל אלסטיק נט
    model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio)

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')

    # Convert the scores to positive values
    mse_scores = -scores
    model.fit(X, y)
    # Print the mean and standard deviation of the scores
    print("Mean MSE:", mse_scores.mean())
    print("Standard Deviation of MSE:", mse_scores.std())   
    return best_alpha , best_l1_ratio , selected_features_backward
model_training(data) 


#לא הצלחתי להכניס את כל המודל הזה לפייפליין ולכן עשיתי פייליין שכן מכיל מודל שנותן ביצועים טובים, אבל המודל הזה נותן ביצועים יפים יותר..


from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet

pipe_preproces_model = Pipeline([
    ('preprocessing_step', ColumnTransformer([
                            ('median_preprocessing',Pipeline([
                                                    ('numerical_imputation', SimpleImputer(strategy='median')),
                                                    ('scaling', StandardScaler())   ]), ['Area','num_of_images','number_in_street','descriotion_level','room_number','total_floors'
                                                                                        ,'hasElevator ', 'hasParking ', 'hasBars ','hasStorage ', 'hasAirCondition ', 'hasBalcony '
                                                                                        , 'hasMamad ', 'handicapFriendly ']),
                             ('mean_preprocessing', Pipeline([
                                                    ('numerical_imputation', SimpleImputer(strategy='mean')),
                                                    ('scaling', StandardScaler())   ]), ['floor','city_rank']),
                        ('categorical_preprocessing', Pipeline([
                                                    ('categorical_imputation', SimpleImputer(strategy='most_frequent')),
                                                    ('one_hot_encoding', OneHotEncoder(drop = 'first', sparse=False, handle_unknown='ignore'))  ]),  ['City', 'type', 'condition ','city_area'
                                                                                                                                                      , 'furniture ', 'entranceDate '])
    ], remainder='drop')) 
    ,('pca',PCA(n_components=4))
      ,('model', ElasticNet(alpha=0.001, l1_ratio=0.4110000000000002)),
])

import pickle

with open('trained_model.pkl', 'wb') as file:
    pickle.dump(pipe_preproces_model, file)
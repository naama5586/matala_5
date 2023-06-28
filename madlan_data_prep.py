
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

data2 = prepare_data(data)

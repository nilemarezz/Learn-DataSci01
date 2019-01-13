#!/usr/bin/env python
# coding: utf-8

# # ทำ Data Analytic ด้วย ชุดข้อมูล Taitanic
# [ลิงค์ชุดข้อมูล](https://github.com/Algoaddict/Python-for-data-science-beginners?fbclid=IwAR0rMHMi1inFBB8UrLDpfGqCafBj7ibKOtkknUr49QBkVwUTyxY4DRiNxEU)
# 
# ![ประเภทของข้อมูล](https://sv1.picz.in.th/images/2019/01/12/9Vfhf2.png "ประเภทข้อมูล")
#     
#     
#     PassengerId - ไอดีผู้โดยสาร
#     Survived - รอดหรือไม่ 1 รอด 0 ไม่รอด
#     Pclass - คลาสของผู้โดยสาร ระดับ 1 2 3 
#     Name - ชื่อ
#     Sex - เพศ
#     Age - อายุ
#     SibSp - ย่อมาจาก Sibling/Spouse มีพี่น้องหรือสามี/ภรรยาบนเรือหรือไม่ 1.2.3..ไม่มี 0 
#     Parch - Parent/Childern มีพ่อแม่หรือลูกอยู่บนเรือไหม 1.2.3... ไม่มี 0
#     Ticket - เลขตั๋ว
#     Fare - อัตราค่าโดยสาร
#     Cabin - เลขห้องพักผู้โดยสาร
#     Embarked - ลงเรือจากที่ไหน S - Southamton C - Cherbourg Q- Queenstone 
#     
# ### libery ที่ใช้มี
#         - pandas
#         - numpy
#         - matplotlib.pyplot

# In[14]:


#import Lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import ชุดข้อมูล
df = pd.read_csv('Data_Set/titanic_data.csv')




#  - df.head() , df.tail() ใส่เลขเพื่อดูลำดับแรกกับท้ายสุด
#  - df.dtypes ดูประเภทของข้อมูล 
#  - df.info() ดูข้อมูลทั้งหมดของตาราง
#  - df.describe() ดูข้อมูลทางคณิตศาสตร์ (ค่าเฉลี่ย , min , max)
#  - df.loc[] ดูข้อมูลใน row นั้นๆ ใส่เลขลงไป นับจาก 0 ถ้าอยากดู row สุดท้ายใช้ iloc[-1] ไล่ลงไป
#  - สามารถเข้าถึงข้อูลใน row ได้โดยใช้ df.ตามด้วยชื่อcol เช่น df.Age

# ## Question Ex1
#  1. แสดงข้อมูลของคนที่อายุ 35
#  2. แสดงจำนวนของคนที่มีอายุ 35 
#  3. แสดงข้อมูลคนที่มีเพศหญิง

# In[17]:


# 1.
df[df.Age == 35]


# In[ ]:


# 2. 
len(df[df.Age == 35])


# In[20]:


# 3.
df[df.Sex == 'female']


# ## Question Ex 2 
#  1. หาจำนวนของคนที่เสียชีวิต และ ไม่เสียชีวิต
#  2. หาเปอร์เซ็นของคนที่รอด 
#  3. แสดงจำนวนของเพศชายที่รอดและไม่รอด

# In[21]:


#1.
df.Survived.value_counts()


# In[23]:


#2.
sur = df.Survived.value_counts()[1] # <-- หาจำนวนในcol Survive ที่มีค่าเป็น 1 
psum = df.Survived.value_counts().sum() # <-- หาจำนวนใทั้งหมดในcol Survive  
print(sur/psum)


# In[24]:


# 3. 
df[df.Sex == 'male']['Survived'].value_counts()


# ## Question Ex 2 
#  1. หาสัดส่วนของผู้โดยสารในแต่ละคลาสนับเป็นกี่เปอร์เซ็นของทั้งหมด
#  2. หาผู้โดยสารในแต่ละชั้นว่ารอดชีวิตคิดเป็นเปอร์เซ็นของกี่คนในชั้น
#  3. 

# In[30]:


#1.
amount = df.Pclass.value_counts() # <-- จำนวนในแต่ละคลาส
psum = df.Pclass.value_counts().sum() # <-- จำนวนทั้งหมด
print(amount/psum)


# In[62]:


#2.
survive1 = df[df.Pclass ==1]['Survived'].value_counts()[1] #<-- คลาสเป็น 1 และ รอดเป็น 1
total1 = df[df.Pclass ==1]['Survived'].value_counts().sum()#<-- คลาสเป็น 1 ทั้งหมด
print(survive1/total1)

survive2 = df[df.Pclass ==2]['Survived'].value_counts()[1]
total2 = df[df.Pclass ==2]['Survived'].value_counts().sum()
print(survive2/total2)

survive3 = df[df.Pclass ==3]['Survived'].value_counts()[1]
total3 = df[df.Pclass ==3]['Survived'].value_counts().sum()
print(survive3/total3)


# # เพิ่ม/ลบ colum , row
#     เพิ่ม row  - ให้อ้างถึง index ที่จะเพิ่ม  และ =[]
#     ลบ  row  - ใช้คำสั่ง drop โดยใน [] ใส่ค่า index และตามด้วย  inplace= True สามารถใส่ index ได้หลายตัว
#     เพิ่ม col  - เรียกที่จะเพิ่มตามด้วย[ชื่อ column] = pd.Series(ค่าที่จะใส่)
#     ลบ col - del ตามด้วยชื่อแล้ว [ชื่อ colume]
#     

# In[63]:


# copy ชุดข้อมูล
df_copy = df.copy()


# In[74]:


# เพิ่มข้อมูลใน index ที่ 891
df_copy.loc[891] = [892,1,2,'Matas Paosriwong','male',32,0,0,370377,7.75,3,'S']
df_copy.head()


# In[90]:


# ลบข้อมูลที่ index ที่ 891
df_copy.drop(df_copy.loc[[890,889]].index,inplace= True)
df_copy.tail()


# In[94]:


# เพิ่ม col
df_copy['newColumn'] = pd.Series()
df_copy.head()


# In[98]:


# ลบ col
del df_copy['newColumn']
df_copy.head()


# # Handling Na (จัดการกับค่า null)
#    

# ## ลบค่า Null 
#     - หาค่า null โดย ใช้ .isnull()
#     - ลบ row ที่เป็น null โดยใช้ .dropna(inplace = True) เลือก col ที่จะลบค่า null ได้โดยใส่ subset = ['ชื่อcol '] หน้า inplace

# In[99]:


df_drop = df.copy()


# In[101]:


#ดูว่าแต่ละ col มี null กี่ตัว
df_drop.isnull().sum()


# In[102]:


#ดูโดยเจาะจงแต่ละ col
df_drop.Age.isnull().sum()


# In[106]:


#ลบค่า null ทั้งหมด
df_drop.dropna(inplace = True)
df_drop.info()


# In[112]:


#ลบค่า null เฉพาะแต่ละใน col 
df_drop = df.copy()
df_drop.dropna(subset = ['Embarked'] , inplace = True)
df_drop.isnull().sum()


# ## Fill in ค่า NA
#     - ดูกราฟโดยใช้ .hist
#     1. เติมด้วยค่าสถิติต่างๆ เช่น Mean 
#         - หาค่า Mean โดยใช้ .mean
#         - fill in  โยใช้ fillna(ค่าที่จะใส่)
#         ** ถ้าเกิดมีข้อมูลที่ผิดพลาดสุดโต่ง เช่นอายุ = 50000 อาจจะทำให้ค่า Mean เพิ่มขึ้นและทำให้ข้อมูลเสียหาย ควรจะใช้ median(ค่ากลาง) ในการใส่
#     2. ใช้ Machine Learning ในการทำนายข้อมูล 

# In[123]:


# ดูค่าเฉี่ย , กราฟ
df_drop = df.copy()
df_drop.mean()
df_drop['Age'].hist()


# In[133]:


# ใส่ค่าเฉลี่ยลงไปในข้อมูล null
mean = df_drop['Age'].mean()
df_drop['Age'].fillna(mean , inplace = True)
df_drop['Age']


# ## Fill in ค่า na โดยแบ่งตามกลุ่ม
#     - ที่ควรแบ่งตามกลุ่มเพราะถ้าเราใส่ค่าให้ na ค่าเดียว ทำให้ข้อมูลซ้ำกันเยอะเกินไป จึงควรแบ่งตามกลุ่มให้เหมาะสม
#     - ใส่ค่าโดยมี condition  ให้ดูที่ In[167]

# In[217]:


#หา่อายุเฉลี่ยนของคนในแต่ละคลาส
df_new = df.copy()
mean1 = df_new[df_new.Pclass == 1]['Age'].mean()


# In[218]:


# ใส่ค่าลงในค่า null ในแต่ละคลาส
df_new.loc[df_new['Pclass'] == 1,'Age'] = df_new[df_new.Pclass == 1]['Age'].fillna(mean1)

df_new[df_new.Pclass == 1]['Age'].isnull().sum()


# In[229]:


# เติมค่า null ด้วยค่าmedian ของอายุในแต่ละเพศของ คลาส3
median_female = df_new[(df_new.Pclass == 3) & (df_new.Sex == 'female')]['Age'].median()
median_male = df_new[(df_new.Pclass == 3) & (df_new.Sex == 'male')]['Age'].median()

df_new.loc[(df_new.Pclass == 3) & (df_new.Sex == 'male'),'Age'] = df_new[(df_new.Pclass == 3) & (df_new.Sex == 'male')]['Age'].fillna(median_male)
df_new.loc[(df_new.Pclass == 3) & (df_new.Sex == 'female'),'Age'] = df_new[(df_new.Pclass == 3) & (df_new.Sex == 'female')]['Age'].fillna(median_female)


# ## Convert String to int 
#     -เพื่อทำให้ประมวลผลข้อมูลได้

# In[231]:


# เปลี่ยนให้ male = 1 female = 0
df_pre = df.copy()
df_pre.loc[(df_pre.Sex == 'male'),'Sex_num'] = 1
df_pre.loc[(df_pre.Sex == 'female'),'Sex_num'] = 0


# In[236]:


# เปลี่ยน S = 0 C = 1 Q = 2 ในcol ชื่อ Embarked_num
df_pre.loc[(df_pre.Embarked == 'S'),'Embarked_num'] = 0
df_pre.loc[(df_pre.Embarked == 'Q'),'Embarked_num'] = 2
df_pre.loc[(df_pre.Embarked == 'C'),'Embarked_num'] = 1


# In[ ]:





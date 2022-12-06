#!/usr/bin/env python
# coding: utf-8

# # Project: Identify Customer Segments
# 
# In this project, you will apply unsupervised learning techniques to identify segments of the population that form the core customer base for a mail-order sales company in Germany. These segments can then be used to direct marketing campaigns towards audiences that will have the highest expected rate of returns. The data that you will use has been provided by our partners at Bertelsmann Arvato Analytics, and represents a real-life data science task.
# 
# This notebook will help you complete this task by providing a framework within which you will perform your analysis steps. In each step of the project, you will see some text describing the subtask that you will perform, followed by one or more code cells for you to complete your work. **Feel free to add additional code and markdown cells as you go along so that you can explore everything in precise chunks.** The code cells provided in the base template will outline only the major tasks, and will usually not be enough to cover all of the minor tasks that comprise it.
# 
# It should be noted that while there will be precise guidelines on how you should handle certain tasks in the project, there will also be places where an exact specification is not provided. **There will be times in the project where you will need to make and justify your own decisions on how to treat the data.** These are places where there may not be only one way to handle the data. In real-life tasks, there may be many valid ways to approach an analysis task. One of the most important things you can do is clearly document your approach so that other scientists can understand the decisions you've made.
# 
# At the end of most sections, there will be a Markdown cell labeled **Discussion**. In these cells, you will report your findings for the completed section, as well as document the decisions that you made in your approach to each subtask. **Your project will be evaluated not just on the code used to complete the tasks outlined, but also your communication about your observations and conclusions at each stage.**

# In[1]:


# import libraries here; add more as necessary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# magic word for producing visualizations in notebook
get_ipython().run_line_magic('matplotlib', 'inline')

'''
Import note: The classroom currently uses sklearn version 0.19.
If you need to use an imputer, it is available in sklearn.preprocessing.Imputer,
instead of sklearn.impute as in newer versions of sklearn.
'''


# ### Step 0: Load the Data
# 
# There are four files associated with this project (not including this one):
# 
# - `Udacity_AZDIAS_Subset.csv`: Demographics data for the general population of Germany; 891211 persons (rows) x 85 features (columns).
# - `Udacity_CUSTOMERS_Subset.csv`: Demographics data for customers of a mail-order company; 191652 persons (rows) x 85 features (columns).
# - `Data_Dictionary.md`: Detailed information file about the features in the provided datasets.
# - `AZDIAS_Feature_Summary.csv`: Summary of feature attributes for demographics data; 85 features (rows) x 4 columns
# 
# Each row of the demographics files represents a single person, but also includes information outside of individuals, including information about their household, building, and neighborhood. You will use this information to cluster the general population into groups with similar demographic properties. Then, you will see how the people in the customers dataset fit into those created clusters. The hope here is that certain clusters are over-represented in the customers data, as compared to the general population; those over-represented clusters will be assumed to be part of the core userbase. This information can then be used for further applications, such as targeting for a marketing campaign.
# 
# To start off with, load in the demographics data for the general population into a pandas DataFrame, and do the same for the feature attributes summary. Note for all of the `.csv` data files in this project: they're semicolon (`;`) delimited, so you'll need an additional argument in your [`read_csv()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) call to read in the data properly. Also, considering the size of the main dataset, it may take some time for it to load completely.
# 
# Once the dataset is loaded, it's recommended that you take a little bit of time just browsing the general structure of the dataset and feature summary file. You'll be getting deep into the innards of the cleaning in the first major step of the project, so gaining some general familiarity can help you get your bearings.

# In[2]:


# Load in the general demographics data.
azdias = pd.read_csv('Udacity_AZDIAS_Subset.csv', delimiter=';')

# Load in the feature summary file.
feat_info = pd.read_csv('AZDIAS_Feature_Summary.csv', delimiter=';')


# In[3]:


feat_info = feat_info.set_index('attribute')


# In[4]:


# Check the structure of the data after it's loaded (e.g. print the number of
# rows and columns, print the first few rows).
# columns
print(len(azdias.columns))
# rows
print(len(azdias))
# head
#azdias.head()


# In[5]:


#azdias['AGER_TYP'].between(-1, 0, inclusive=True)

#azdias['AGER_TYP'] = np.where((azdias['AGER_TYP'] <= -1) & (azdias['AGER_TYP'] >= 0), np.NaN, azdias['AGER_TYP'])


# In[6]:


"""
import re
for col in azdias.columns[2:3]:
    valid_data_range = feat_info['missing_or_unknown'][col]
    valid_data_range = [float(s) for s in re.findall(r'-?\d+\.?\d*', valid_data_range)]
    if valid_data_range != []:
        for idx,val in enumerate(azdias[col]):
            if not valid_data_range[0] <= val <= valid_data_range[-1]:
                azdias[col][idx] = np.nan
            print("done "+str(idx))
    print("done")
"""


# > **Tip**: Add additional cells to keep everything in reasonably-sized chunks! Keyboard shortcut `esc --> a` (press escape to enter command mode, then press the 'A' key) adds a new cell before the active cell, and `esc --> b` adds a new cell after the active cell. If you need to convert an active cell to a markdown cell, use `esc --> m` and to convert to a code cell, use `esc --> y`. 
# 
# ## Step 1: Preprocessing
# 
# ### Step 1.1: Assess Missing Data
# 
# The feature summary file contains a summary of properties for each demographics data column. You will use this file to help you make cleaning decisions during this stage of the project. First of all, you should assess the demographics data in terms of missing data. Pay attention to the following points as you perform your analysis, and take notes on what you observe. Make sure that you fill in the **Discussion** cell with your findings and decisions at the end of each step that has one!
# 
# #### Step 1.1.1: Convert Missing Value Codes to NaNs
# The fourth column of the feature attributes summary (loaded in above as `feat_info`) documents the codes from the data dictionary that indicate missing or unknown data. While the file encodes this as a list (e.g. `[-1,0]`), this will get read in as a string object. You'll need to do a little bit of parsing to make use of it to identify and clean the data. Convert data that matches a 'missing' or 'unknown' value code into a numpy NaN value. You might want to see how much data takes on a 'missing' or 'unknown' code, and how much data is naturally missing, as a point of interest.
# 
# **As one more reminder, you are encouraged to add additional cells to break up your analysis into manageable chunks.**

# In[7]:


# Identify missing or unknown data values and convert them to NaNs.
import re
for col in azdias.columns:
    valid_data_range = feat_info['missing_or_unknown'][col]
    valid_data_range = [float(s) for s in re.findall(r'-?\d+\.?\d*', valid_data_range)]
    if valid_data_range != [] and col!='CAMEO_DEUG_2015' and col!='CAMEO_DEU_2015' and col!='CAMEO_INTL_2015':
        azdias[col] = np.where(azdias[col].isin(valid_data_range),np.NaN, azdias[col])
    
# Hnadling "x" cases
for col in ['CAMEO_DEUG_2015','CAMEO_DEU_2015','CAMEO_INTL_2015']:
    valid_data_range = feat_info['missing_or_unknown'][col]
    valid_data_range = valid_data_range.strip('][').split(',')
    if valid_data_range != []:
        azdias[col] = np.where(azdias[col].isin(valid_data_range),np.NaN, azdias[col])


# #### Step 1.1.2: Assess Missing Data in Each Column
# 
# How much missing data is present in each column? There are a few columns that are outliers in terms of the proportion of values that are missing. You will want to use matplotlib's [`hist()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html) function to visualize the distribution of missing value counts to find these columns. Identify and document these columns. While some of these columns might have justifications for keeping or re-encoding the data, for this project you should just remove them from the dataframe. (Feel free to make remarks about these outlier columns in the discussion, however!)
# 
# For the remaining features, are there any patterns in which columns have, or share, missing data?

# In[8]:


# Perform an assessment of how much missing data there is in each column of the
# dataset.
azdias.isnull().sum()


# In[9]:


# Investigate patterns in the amount of missing data in each column.

len(azdias.columns)


# In[10]:


# Remove the outlier columns from the dataset. (You'll perform other data
# engineering tasks such as re-encoding and imputation later.)
# I am seeing the percentage of misssing values in each column and removing columns with greater than 30% missing values
azdias = azdias[azdias.columns[(azdias.isnull().sum() * 100) / len(azdias) < 30].tolist()].copy()

# validating no more column with >30% invalid values
print(((azdias.isnull().sum() * 100) / len(azdias) > 30).any())


# #### Discussion 1.1.2: Assess Missing Data in Each Column
# 
# (Double click this cell and replace this text with your own text, reporting your observations regarding the amount of missing data in each column. Are there any patterns in missing values? Which columns were removed from the dataset?)
# 
# 1) All the info  given in feat_info is used to convert invalid data to NaN's(columns where we have non numeric "X" has been handeled seperately)
# 
# 2) A simple print out of missing values in each column is provided
# 
# 3) columns with more than 30% Nan's has been removed
# 

# #### Step 1.1.3: Assess Missing Data in Each Row
# 
# Now, you'll perform a similar assessment for the rows of the dataset. How much data is missing in each row? As with the columns, you should see some groups of points that have a very different numbers of missing values. Divide the data into two subsets: one for data points that are above some threshold for missing values, and a second subset for points below that threshold.
# 
# In order to know what to do with the outlier rows, we should see if the distribution of data values on columns that are not missing data (or are missing very little data) are similar or different between the two groups. Select at least five of these columns and compare the distribution of values.
# - You can use seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) function to create a bar chart of code frequencies and matplotlib's [`subplot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html) function to put bar charts for the two subplots side by side.
# - To reduce repeated code, you might want to write a function that can perform this comparison, taking as one of its arguments a column to be compared.
# 
# Depending on what you observe in your comparison, this will have implications on how you approach your conclusions later in the analysis. If the distributions of non-missing features look similar between the data with many missing values and the data with few or no missing values, then we could argue that simply dropping those points from the analysis won't present a major issue. On the other hand, if the data with many missing values looks very different from the data with few or no missing values, then we should make a note on those data as special. We'll revisit these data later on. **Either way, you should continue your analysis for now using just the subset of the data with few or no missing values.**

# In[11]:


# How much data is missing in each row of the dataset?
azdias.isnull().sum(axis=1)


# In[12]:


# Write code to divide the data into two subsets based on the number of missing
# values in each row.
subset1 = azdias[azdias.isnull().sum(axis=1) <= 0.3 * len(azdias.columns)]
subset2 = azdias[azdias.isnull().sum(axis=1) > 0.3 * len(azdias.columns)]


# In[13]:


#subset1.isnull().sum()


# In[14]:


#subset2.isnull().sum()


# In[15]:


# Compare the distribution of values for at least five columns where there are
# no or few missing values, between the two subsets.
def plot_subsets(subset1, subset2):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False, sharex=True)
    sns.countplot(subset1,ax=axes[0])
    sns.countplot(subset2,ax=axes[1])
    
assert len(subset1.columns) == len(subset2.columns)
null_entries_1 = subset1.isnull().sum()
null_entries_2 = subset2.isnull().sum()
print_count = 0
for idx,x in enumerate(null_entries_1):
    if null_entries_1[idx]==0 and null_entries_2[idx]==0:
        print_count +=1 
        plot_subsets(subset1[subset1.columns[idx]], subset2[subset2.columns[idx]])
    if print_count == 5:
        break


# In[16]:


# subset 1 will be considered as data in further steps


# #### Discussion 1.1.3: Assess Missing Data in Each Row
# 
# (Double-click this cell and replace this text with your own text, reporting your observations regarding missing data in rows. Are the data with lots of missing values are qualitatively different from data with few or no missing values?)
# 
# 1) Rows are analysed for any invalid data. divided into 2 sets, subset1 with less than 30% missing values and subset2 with greater than 30% missing values
# 
# 2) A comaprision plot has been provided for 5 features in subset1 and subset2. Their distibution doesn't seem to match. even then(since insisted) going ahead with subset1 for further analysis
# 

# ### Step 1.2: Select and Re-Encode Features
# 
# Checking for missing data isn't the only way in which you can prepare a dataset for analysis. Since the unsupervised learning techniques to be used will only work on data that is encoded numerically, you need to make a few encoding changes or additional assumptions to be able to make progress. In addition, while almost all of the values in the dataset are encoded using numbers, not all of them represent numeric values. Check the third column of the feature summary (`feat_info`) for a summary of types of measurement.
# - For numeric and interval data, these features can be kept without changes.
# - Most of the variables in the dataset are ordinal in nature. While ordinal values may technically be non-linear in spacing, make the simplifying assumption that the ordinal variables can be treated as being interval in nature (that is, kept without any changes).
# - Special handling may be necessary for the remaining two variable types: categorical, and 'mixed'.
# 
# In the first two parts of this sub-step, you will perform an investigation of the categorical and mixed-type features and make a decision on each of them, whether you will keep, drop, or re-encode each. Then, in the last part, you will create a new data frame with only the selected and engineered columns.
# 
# Data wrangling is often the trickiest part of the data analysis process, and there's a lot of it to be done here. But stick with it: once you're done with this step, you'll be ready to get to the machine learning parts of the project!

# In[17]:


# How many features are there of each data type?

for typ in feat_info['type'].unique(): 
    print('Number of Feature of type '+ typ +" is "+str(len(feat_info[feat_info['type'] == typ])))


# In[ ]:





# In[ ]:





# #### Step 1.2.1: Re-Encode Categorical Features
# 
# For categorical data, you would ordinarily need to encode the levels as dummy variables. Depending on the number of categories, perform one of the following:
# - For binary (two-level) categoricals that take numeric values, you can keep them without needing to do anything.
# - There is one binary variable that takes on non-numeric values. For this one, you need to re-encode the values as numbers or create a dummy variable.
# - For multi-level categoricals (three or more values), you can choose to encode the values using multiple dummy variables (e.g. via [OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)), or (to keep things straightforward) just drop them from the analysis. As always, document your choices in the Discussion section.

# In[18]:


cat_feat = feat_info[feat_info['type'] == 'categorical']
for idx in cat_feat.index:
    try:           
        print("The unique category in this feature is/are "+ str(subset1[idx].unique()))

    except:
        print("checking next column since this has been deleted earlier as a pre processing step")        


# In[19]:


# Assess categorical variables: which are binary, which are multi-level, and
# which one needs to be re-encoded?
count_binary = 0
count_non_binary = 0
for idx in cat_feat.index:
    
    if idx in subset1.columns:
        # handling non numeric binary category case
        if len(subset1[idx].unique()) <= 3 and idx !='OST_WEST_KZ':
            # checkin case where we can have [1,2, nan] which is also binary and making sure [1,2,3] type are deleted(doesn't exist, just an edge case)
            if len(subset1[idx].unique()) == 3:
                if not subset1[idx].isnull().sum() > 0:
                    subset1 = subset1.drop(idx, axis=1)
                    count_non_binary += 1
                else:
                    count_binary += 1
            else:
                count_binary += 1
        else:
            if idx != 'OST_WEST_KZ':
                subset1 = subset1.drop(idx, axis=1) # Ignoring all categorical variables with more than 3 values
            count_non_binary += 1


# In[20]:


print("Number of binary(category) features: "+str(count_binary))
print("Number of Non binary(category) features: "+str(count_non_binary))


# In[21]:


"""
# Assess categorical variables: which are binary, which are multi-level, and
# which one needs to be re-encoded?
cat_feat = feat_info[feat_info['type'] == 'categorical']
for idx in cat_feat.index:
    try:
        # handling non numeric binary category case
        if len(subset1[idx].unique()) <= 3 and idx !='OST_WEST_KZ':
            # checkin case where we can have [1,2, nan] which is also binary and making sure [1,2,3] type are deleted(doesn't exist, just an edge case)
            if len(subset1[idx].unique()) == 3:
                if not subset1[idx].isnull().sum() > 0:
                    subset1 = subset1.drop(idx, axis=1) 
        else:
            subset1 = subset1.drop(idx, axis=1) # Ignoring all categorical variables with more than 3 values
            
        print("The unique category in this feature is/are "+ str(subset1[idx].unique()))
    except:
        print("checking next column since this has been deleted earlier as a pre processing step")
"""


# In[22]:


# Handling non-numeric case
subset1['OST_WEST_KZ'].replace(subset1['OST_WEST_KZ'].unique(),[0, 1], inplace=True)


# In[23]:


# Validation
subset1['OST_WEST_KZ'].unique()


# #### Discussion 1.2.1: Re-Encode Categorical Features
# 
# (Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding categorical features. Which ones did you keep, which did you drop, and what engineering steps did you perform?)
# 
# 1) All categorical features with more than 2 variables have been removed
# 
# 2)Some features might be deleted in the preprocessing step, hence using try, except
# 
# 3)Feature 'OST_WEST_KZ' has a non numeric entries which has been handeled seperately(converting to class 0 and 1)
# 
# 4) All other categorical features(bivariate) are kept as it is
# 

# #### Step 1.2.2: Engineer Mixed-Type Features
# 
# There are a handful of features that are marked as "mixed" in the feature summary that require special treatment in order to be included in the analysis. There are two in particular that deserve attention; the handling of the rest are up to your own choices:
# - "PRAEGENDE_JUGENDJAHRE" combines information on three dimensions: generation by decade, movement (mainstream vs. avantgarde), and nation (east vs. west). While there aren't enough levels to disentangle east from west, you should create two new variables to capture the other two dimensions: an interval-type variable for decade, and a binary variable for movement.
# - "CAMEO_INTL_2015" combines information on two axes: wealth and life stage. Break up the two-digit codes by their 'tens'-place and 'ones'-place digits into two new ordinal variables (which, for the purposes of this project, is equivalent to just treating them as their raw numeric values).
# - If you decide to keep or engineer new features around the other mixed-type features, make sure you note your steps in the Discussion section.
# 
# Be sure to check `Data_Dictionary.md` for the details needed to finish these tasks.

# In[24]:


#Data_Dictionary.md
from IPython.display import Markdown, display
display(Markdown("Data_Dictionary.md"))


# In[25]:


#mix_feat = feat_info[feat_info['type'] == 'mixed']
#for idx in mix_feat.index:
 #   if idx in subset1.columns:
  #      print(subset1[idx].unique())
feat_info[feat_info['type'] == 'mixed']


# In[26]:


# Investigate "PRAEGENDE_JUGENDJAHRE" and engineer two new variables.
# Handling Feature PRAEGENDE_JUGENDJAHRE
# steps
# 1. Create 2 new columns(features/variables) for decade(interval) and movement(binary)
subset1['PRAEGENDE_JUGENDJAHRE_DECADE'] = subset1['PRAEGENDE_JUGENDJAHRE'].replace([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],[40,40,50,50,60,60,60,70,70,80,80,80,80,90,90])
subset1['PRAEGENDE_JUGENDJAHRE_MOVEMENT'] = subset1['PRAEGENDE_JUGENDJAHRE'].replace([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],['Mainstream','Avantgarde','Mainstream','Avantgarde','Mainstream','Avantgarde','Avantgarde','Mainstream','Avantgarde','Mainstream','Avantgarde','Mainstream','Avantgarde','Mainstream','Avantgarde'])
# 2. Re-encode the as numeric
subset1['PRAEGENDE_JUGENDJAHRE_DECADE'].replace(np.delete(subset1['PRAEGENDE_JUGENDJAHRE_DECADE'].unique(), 5),[0, 1, 2, 3, 4, 5], inplace=True)
subset1['PRAEGENDE_JUGENDJAHRE_MOVEMENT'].replace(np.delete(subset1['PRAEGENDE_JUGENDJAHRE_MOVEMENT'].unique(), 2),[0, 1], inplace=True)
# delete the old column
#subset1 = subset1.drop(['PRAEGENDE_JUGENDJAHRE'], axis=1)


# In[ ]:





# In[27]:


# validation
subset1['PRAEGENDE_JUGENDJAHRE_MOVEMENT'].unique()


# In[ ]:





# In[28]:


# Investigate "CAMEO_INTL_2015" and engineer two new variables.
subset1['CAMEO_INTL_2015'] = [list(l) if type(l)!=float else l for l in subset1['CAMEO_INTL_2015']]
# create two new variables "CAMEO_INTL_2015_WEALTH" and "CAMEO_INTL_2015_LIFE_STAGE"
subset1['CAMEO_INTL_2015_WEALTH'] = [l[0] if type(l)!=float else l for l in subset1['CAMEO_INTL_2015']]
subset1['CAMEO_INTL_2015_LIFE_STAGE'] = [l[1] if type(l)!=float else l for l in subset1['CAMEO_INTL_2015']]
# splitting a column is not possible since colmn has nan values
# deleting the extra column
#subset1 = subset1.drop(['CAMEO_INTL_2015'], axis=1)


# In[29]:


# validation
print(subset1['CAMEO_INTL_2015_WEALTH'].unique())
print(subset1['CAMEO_INTL_2015_LIFE_STAGE'].unique())


# In[30]:


#Dropping additional mixed features
mix_feat = feat_info[feat_info['type'] == 'mixed']

print("Number of columns before deleting mixed type features are "+str(len(subset1.columns)))
print("Number of mixed features in curret data are "+str(len(mix_feat)))

# Deleting all mixed features, since the new ones(for the required) is created and the rest are being ignored
for idx in mix_feat.index:
    try:
        subset1 = subset1.drop(idx, axis=1) # Ignoring all categorical variables with more than 3 values
    except:
        print("checking next column since this has been deleted earlier as a pre processing step")
                
print("Number of columns after deleting mixed type features are "+str(len(subset1.columns)))


# #### Discussion 1.2.2: Engineer Mixed-Type Features
# 
# (Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding mixed-value features. Which ones did you keep, which did you drop, and what engineering steps did you perform?)
# 
# 1) "PRAEGENDE_JUGENDJAHRE" feature has been handeled seperately using "Data_Dictionary.md" and the column is split into two. Original column is not yet deleted here, will be done in step 3 at once
# 
# 2) "CAMEO_INTL_2015" also has mixed features, it is also divided into two features with tenth and ones place respectively. Original column is not yet deleted here, will be done in step 3 at once
# 
# 3) Ignoring all other mixed type features and deleting them all(Remember newly added features remain, since they arent mixed type)

# #### Step 1.2.3: Complete Feature Selection
# 
# In order to finish this step up, you need to make sure that your data frame now only has the columns that you want to keep. To summarize, the dataframe should consist of the following:
# - All numeric, interval, and ordinal type columns from the original dataset.
# - Binary categorical features (all numerically-encoded).
# - Engineered features from other multi-level categorical features and mixed features.
# 
# Make sure that for any new columns that you have engineered, that you've excluded the original columns from the final dataset. Otherwise, their values will interfere with the analysis later on the project. For example, you should not keep "PRAEGENDE_JUGENDJAHRE", since its values won't be useful for the algorithm: only the values derived from it in the engineered features you created should be retained. As a reminder, your data should only be from **the subset with few or no missing values**.

# In[31]:


# If there are other re-engineering tasks you need to perform, make sure you
# take care of them here. (Dealing with missing data will come in step 2.1.)
for col in subset1.columns:
    try:
        print("---------------------")
        print(feat_info['type'][col])
        print("Valid categories are")
        print(subset1[col].unique())
        print("---------------------")
        if feat_info['type'][col] == 'categorical':
            assert len(subset1[col].unique()) <= 3
        assert feat_info['type'][col] != 'mixed'
    except:
        print(col + " is Engineerd feature")


# In[32]:


# Do whatever you need to in order to ensure that the dataframe only contains
# the columns that should be passed to the algorithm functions.

# Already done in abaove steps


# ### Step 1.3: Create a Cleaning Function
# 
# Even though you've finished cleaning up the general population demographics data, it's important to look ahead to the future and realize that you'll need to perform the same cleaning steps on the customer demographics data. In this substep, complete the function below to execute the main feature selection, encoding, and re-engineering steps you performed above. Then, when it comes to looking at the customer data in Step 3, you can just run this function on that DataFrame to get the trimmed dataset in a single step.

# In[33]:


def clean_data(df):
    """
    Perform feature trimming, re-encoding, and engineering for demographics
    data
    
    INPUT: Demographics DataFrame
    OUTPUT: Trimmed and cleaned demographics DataFrame
    """
    
    # Put in code here to execute all main cleaning steps:
    # convert missing value codes into NaNs, ...
    # feat_info = feat_info.set_index('attribute')
    for col in df.columns:
        valid_data_range = feat_info['missing_or_unknown'][col]
        valid_data_range = [float(s) for s in re.findall(r'-?\d+\.?\d*', valid_data_range)]
        if valid_data_range != [] and col!='CAMEO_DEUG_2015' and col!='CAMEO_DEU_2015' and col!='CAMEO_INTL_2015':
            df[col] = np.where(df[col].isin(valid_data_range),np.NaN, df[col])

    # Hnadling "x" cases
    for col in ['CAMEO_DEUG_2015','CAMEO_DEU_2015','CAMEO_INTL_2015']:
        valid_data_range = feat_info['missing_or_unknown'][col]
        valid_data_range = valid_data_range.strip('][').split(',')
        if valid_data_range != []:
            df[col] = np.where(df[col].isin(valid_data_range),np.NaN, df[col])    
            
    # remove selected columns and rows, ...
    # I am seeing the percentage of misssing values in each column and removing columns with greater than 30% missing values
    azdias = df[df.columns[(df.isnull().sum() * 100) / len(df) < 30].tolist()].copy()
    subset1 = azdias[azdias.isnull().sum(axis=1) <= 0.3 * len(azdias.columns)]
    subset2 = azdias[azdias.isnull().sum(axis=1) > 0.3 * len(azdias.columns)]
    
    # select, re-encode, and engineer column values.
    # Assess categorical variables: which are binary, which are multi-level, and
    # which one needs to be re-encoded?
    count_binary = 0
    count_non_binary = 0
    for idx in cat_feat.index:
        if idx in subset1.columns:
            # handling non numeric binary category case
            if len(subset1[idx].unique()) <= 3 and idx !='OST_WEST_KZ':
                # checkin case where we can have [1,2, nan] which is also binary and making sure [1,2,3] type are deleted(doesn't exist, just an edge case)
                if len(subset1[idx].unique()) == 3:
                    if not subset1[idx].isnull().sum() > 0:
                        subset1 = subset1.drop(idx, axis=1)
                        count_non_binary += 1
                    else:
                        count_binary += 1
                else:
                    count_binary += 1
            else:
                if idx != 'OST_WEST_KZ':
                    subset1 = subset1.drop(idx, axis=1) # Ignoring all categorical variables with more than 3 values
                count_non_binary += 1
    # Handling non-numeric case
    subset1['OST_WEST_KZ'].replace(subset1['OST_WEST_KZ'].unique(),[0, 1], inplace=True)
    
    # 1. Create 2 new columns(features/variables) for decade(interval) and movement(binary)
    subset1['PRAEGENDE_JUGENDJAHRE_DECADE'] = subset1['PRAEGENDE_JUGENDJAHRE'].replace([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],[40,40,50,50,60,60,60,70,70,80,80,80,80,90,90])
    subset1['PRAEGENDE_JUGENDJAHRE_MOVEMENT'] = subset1['PRAEGENDE_JUGENDJAHRE'].replace([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],['Mainstream','Avantgarde','Mainstream','Avantgarde','Mainstream','Avantgarde','Avantgarde','Mainstream','Avantgarde','Mainstream','Avantgarde','Mainstream','Avantgarde','Mainstream','Avantgarde'])
    # 2. Re-encode the as numeric
    subset1['PRAEGENDE_JUGENDJAHRE_DECADE'].replace(np.delete(subset1['PRAEGENDE_JUGENDJAHRE_DECADE'].unique(), 5),[0, 1, 2, 3, 4, 5], inplace=True)
    subset1['PRAEGENDE_JUGENDJAHRE_MOVEMENT'].replace(np.delete(subset1['PRAEGENDE_JUGENDJAHRE_MOVEMENT'].unique(), 2),[0, 1], inplace=True)
    
    # Investigate "CAMEO_INTL_2015" and engineer two new variables.
    subset1['CAMEO_INTL_2015'] = [list(l) if type(l)!=float else l for l in subset1['CAMEO_INTL_2015']]
    # create two new variables "CAMEO_INTL_2015_WEALTH" and "CAMEO_INTL_2015_LIFE_STAGE"
    subset1['CAMEO_INTL_2015_WEALTH'] = [l[0] if type(l)!=float else l for l in subset1['CAMEO_INTL_2015']]
    subset1['CAMEO_INTL_2015_LIFE_STAGE'] = [l[1] if type(l)!=float else l for l in subset1['CAMEO_INTL_2015']]

    #Dropping additional mixed features
    mix_feat = feat_info[feat_info['type'] == 'mixed']

    print("Number of columns before deleting mixed type features are "+str(len(subset1.columns)))
    print("Number of mixed features in curret data are "+str(len(mix_feat)))

    # Deleting all mixed features, since the new ones(for the required) is created and the rest are being ignored
    for idx in mix_feat.index:
        try:
            subset1 = subset1.drop(idx, axis=1) # Ignoring all categorical variables with more than 3 values
        except:
            print("checking next column since this has been deleted earlier as a pre processing step")

    print("Number of columns after deleting mixed type features are "+str(len(subset1.columns)))
    # Return the cleaned dataframe.
    return subset1
    


# ## Step 2: Feature Transformation
# 
# ### Step 2.1: Apply Feature Scaling
# 
# Before we apply dimensionality reduction techniques to the data, we need to perform feature scaling so that the principal component vectors are not influenced by the natural differences in scale for features. Starting from this part of the project, you'll want to keep an eye on the [API reference page for sklearn](http://scikit-learn.org/stable/modules/classes.html) to help you navigate to all of the classes and functions that you'll need. In this substep, you'll need to check the following:
# 
# - sklearn requires that data not have missing values in order for its estimators to work properly. So, before applying the scaler to your data, make sure that you've cleaned the DataFrame of the remaining missing values. This can be as simple as just removing all data points with missing data, or applying an [Imputer](https://scikit-learn.org/0.16/modules/generated/sklearn.preprocessing.Imputer.html) to replace all missing values. You might also try a more complicated procedure where you temporarily remove missing values in order to compute the scaling parameters before re-introducing those missing values and applying imputation. Think about how much missing data you have and what possible effects each approach might have on your analysis, and justify your decision in the discussion section below.
# - For the actual scaling function, a [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) instance is suggested, scaling each feature to mean 0 and standard deviation 1.
# - For these classes, you can make use of the `.fit_transform()` method to both fit a procedure to the data as well as apply the transformation to the data at the same time. Don't forget to keep the fit sklearn objects handy, since you'll be applying them to the customer demographics data towards the end of the project.

# In[34]:


# If you've not yet cleaned the dataset of all NaN values, then investigate and
# do that now.
subset1 = subset1.dropna() # we will loose close to 150k data points, we still have good amount of data for application like clustering


# In[35]:


# Apply feature scaling to the general population demographics data.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler_fit_transform = scaler.fit_transform(subset1)


# ### Discussion 2.1: Apply Feature Scaling
# 
# (Double-click this cell and replace this text with your own text, reporting your decisions regarding feature scaling.)

# ### Step 2.2: Perform Dimensionality Reduction
# 
# On your scaled data, you are now ready to apply dimensionality reduction techniques.
# 
# - Use sklearn's [PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) class to apply principal component analysis on the data, thus finding the vectors of maximal variance in the data. To start, you should not set any parameters (so all components are computed) or set a number of components that is at least half the number of features (so there's enough features to see the general trend in variability).
# - Check out the ratio of variance explained by each principal component as well as the cumulative variance explained. Try plotting the cumulative or sequential values using matplotlib's [`plot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html) function. Based on what you find, select a value for the number of transformed features you'll retain for the clustering part of the project.
# - Once you've made a choice for the number of components to keep, make sure you re-fit a PCA instance to perform the decided-on transformation.

# In[41]:


# Apply PCA to the data.
from sklearn.decomposition import PCA
collect_tot_var = []
for i in range(1,int(len(subset1.columns)/2)):
    pca = PCA(n_components=i)
    # pca.fit(scaler_fit_transform)
    X_pca = pca.fit_transform(scaler_fit_transform)
    print("Number of components is "+str(i)+". Variance explained by each component are ")
    print(pca.explained_variance_)
    collect_tot_var.append(np.sum(pca.explained_variance_))


# In[42]:


plt.plot(collect_tot_var)


# In[93]:


# pca.explained_variance_


# In[94]:


# np.sum(pca.explained_variance_)


# In[47]:


# Investigate the variance accounted for by each principal component.
# plt.plot(pca.explained_variance_)
# pd.DataFrame(pca.components_,columns=subset1.columns)


# In[44]:


# Re-apply PCA to the data while selecting for number of components to retain.
pca = PCA(n_components=20)
# pca.fit(scaler_fit_transform)
X_pca = pca.fit_transform(scaler_fit_transform)


# In[97]:


# components = pd.DataFrame(np.round(pca.components_, 4), columns = subset1.keys())


# In[45]:


components = pd.DataFrame(pca.components_,columns=subset1.columns)
components['EXPLAINED_VARIANCE'] = pca.explained_variance_
first_column = components.pop('EXPLAINED_VARIANCE')
components.insert(0, 'EXPLAINED_VARIANCE', first_column)


# In[46]:


components


# ### Discussion 2.2: Perform Dimensionality Reduction
# 
# (Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding dimensionality reduction. How many principal components / transformed features are you retaining for the next step of the analysis?)
# 
# 
# After making sure of proper feature scaling, we have to transform our data to higher dimension using PCA. This reduces the compoenents for analysis since it given out the features/components which captures highest variability.
# 
# Sklearn's PCA has been used on transformed data with 5 priniciple components which accounts to about 30% variablity
# 
# A new dataframe(components) is created which has explained variance of each components along with its featur weights

# ### Step 2.3: Interpret Principal Components
# 
# Now that we have our transformed principal components, it's a nice idea to check out the weight of each variable on the first few components to see if they can be interpreted in some fashion.
# 
# As a reminder, each principal component is a unit vector that points in the direction of highest variance (after accounting for the variance captured by earlier principal components). The further a weight is from zero, the more the principal component is in the direction of the corresponding feature. If two features have large weights of the same sign (both positive or both negative), then increases in one tend expect to be associated with increases in the other. To contrast, features with different signs can be expected to show a negative correlation: increases in one variable should result in a decrease in the other.
# 
# - To investigate the features, you should map each weight to their corresponding feature name, then sort the features according to weight. The most interesting features for each principal component, then, will be those at the beginning and end of the sorted list. Use the data dictionary document to help you understand these most prominent features, their relationships, and what a positive or negative value on the principal component might indicate.
# - You should investigate and interpret feature associations from the first three principal components in this substep. To help facilitate this, you should write a function that you can call at any time to print the sorted list of feature weights, for the *i*-th principal component. This might come in handy in the next step of the project, when you interpret the tendencies of the discovered clusters.

# In[48]:


# Map weights for the first principal component to corresponding feature names
# and then print the linked values, sorted by weight.
# HINT: Try defining a function here or in a new cell that you can reuse in the
# other cells.
# pca.score(scaler_fit_transform)
pca_components = pd.DataFrame(np.round(pca.components_, 4), columns = subset1.keys())
pc1 = (pca_components.loc[0]).sort_values()
print(pc1)


# In[ ]:





# In[49]:


# Map weights for the second principal component to corresponding feature names
# and then print the linked values, sorted by weight.

pc2 = (components.loc[1]).sort_values()
print(pc2)


# In[50]:


# Map weights for the third principal component to corresponding feature names
# and then print the linked values, sorted by weight.

pc3 = (components.loc[2]).sort_values()
print(pc3)


# ### Discussion 2.3: Interpret Principal Components
# 
# (Double-click this cell and replace this text with your own text, reporting your observations from detailed investigation of the first few principal components generated. Can we interpret positive and negative values from them in a meaningful way?)
# 
# I feel negative and positive is not a critical thing here, since we are talking about variability. An absolute value can also be a good indicator. but I have used the numbers as it is to sort features for each component
# 

# ## Step 3: Clustering
# 
# ### Step 3.1: Apply Clustering to General Population
# 
# You've assessed and cleaned the demographics data, then scaled and transformed them. Now, it's time to see how the data clusters in the principal components space. In this substep, you will apply k-means clustering to the dataset and use the average within-cluster distances from each point to their assigned cluster's centroid to decide on a number of clusters to keep.
# 
# - Use sklearn's [KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans) class to perform k-means clustering on the PCA-transformed data.
# - Then, compute the average difference from each point to its assigned cluster's center. **Hint**: The KMeans object's `.score()` method might be useful here, but note that in sklearn, scores tend to be defined so that larger is better. Try applying it to a small, toy dataset, or use an internet search to help your understanding.
# - Perform the above two steps for a number of different cluster counts. You can then see how the average distance decreases with an increasing number of clusters. However, each additional cluster provides a smaller net benefit. Use this fact to select a final number of clusters in which to group the data. **Warning**: because of the large size of the dataset, it can take a long time for the algorithm to resolve. The more clusters to fit, the longer the algorithm will take. You should test for cluster counts through at least 10 clusters to get the full picture, but you shouldn't need to test for a number of clusters above about 30.
# - Once you've selected a final number of clusters to use, re-fit a KMeans instance to perform the clustering operation. Make sure that you also obtain the cluster assignments for the general demographics data, since you'll be using them in the final Step 3.3.

# In[52]:


"""
# Over a number of different cluster counts...
from sklearn.cluster import KMeans
scores=[]
for clusters in range(1,20):

    # run k-means clustering on the data and...
    kmeans = KMeans(n_clusters=clusters, random_state=42)
    kmeans.fit(X_pca)
    # compute the average within-cluster distances.
    score = kmeans.score(X_pca)
    scores.append(-score)
    # print(score)
"""    


# In[ ]:





# In[ ]:


# Investigate the change in within-cluster distance across number of clusters.
# HINT: Use matplotlib's plot function to visualize this relationship.
"""
plt.plot(scores)
kmeans.cluster_centers_
"""


# In[ ]:





# In[53]:


from sklearn.cluster import KMeans
def drawSSEPlot(df, n_clusters=15):
    import matplotlib.pyplot as plt
    inertia_values = []
    for i in range(1, n_clusters+1):
        km = KMeans(n_clusters=i,random_state=42)
        km.fit_predict(df)
        inertia_values.append(km.inertia_)
        print(i)
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(range(1, n_clusters+1), inertia_values, color='red')
    plt.xlabel('No. of Clusters', fontsize=15)
    plt.ylabel('SSE / Inertia', fontsize=15)
    plt.title('SSE / Inertia vs No. Of Clusters', fontsize=15)
    plt.grid()
    plt.show()
    
drawSSEPlot(X_pca)


# In[54]:


# Re-fit the k-means model with the selected number of clusters and obtain
# cluster predictions for the general population demographics data.
n_clusters= 13
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_pca)
X_train_predict = kmeans.predict(X_pca)


# In[55]:


general_population = []
for i in range(n_clusters):
    general_population.append(np.count_nonzero(X_train_predict==i)/len(X_train_predict))


# In[ ]:





# ### Discussion 3.1: Apply Clustering to General Population
# 
# (Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding clustering. Into how many clusters have you decided to segment the population?)
# 
# based on elbow method(not exactly elbow in my case since using continuous numbers), number of cluster was decided to be 10(also looking at complexity). More the components more the time taken to solve
# 
# Clustering has been done accordingly and number of data samples entering each cluster has been counted for further application

# ### Step 3.2: Apply All Steps to the Customer Data
# 
# Now that you have clusters and cluster centers for the general population, it's time to see how the customer data maps on to those clusters. Take care to not confuse this for re-fitting all of the models to the customer data. Instead, you're going to use the fits from the general population to clean, transform, and cluster the customer data. In the last step of the project, you will interpret how the general population fits apply to the customer data.
# 
# - Don't forget when loading in the customers data, that it is semicolon (`;`) delimited.
# - Apply the same feature wrangling, selection, and engineering steps to the customer demographics using the `clean_data()` function you created earlier. (You can assume that the customer demographics data has similar meaning behind missing data patterns as the general demographics data.)
# - Use the sklearn objects from the general demographics data, and apply their transformations to the customers data. That is, you should not be using a `.fit()` or `.fit_transform()` method to re-fit the old objects, nor should you be creating new sklearn objects! Carry the data through the feature scaling, PCA, and clustering steps, obtaining cluster assignments for all of the data in the customer demographics data.

# In[56]:


# Load in the customer demographics data.
customers = pd.read_csv('Udacity_CUSTOMERS_Subset.csv', delimiter=';')


# In[57]:


# Apply preprocessing, feature transformation, and clustering from the general
# demographics onto the customer data, obtaining cluster predictions for the
# customer demographics data.
subset1 = clean_data(customers)
subset1 = subset1.dropna()
# scaler = StandardScaler()
scaler_fit_transform = scaler.fit_transform(subset1)
# pca = PCA(n_components=5)
# pca.fit(scaler_fit_transform)
X_pca = pca.fit_transform(scaler_fit_transform)
X_predict = kmeans.predict(X_pca)


# In[58]:


customer_data= []
for i in range(n_clusters):
    customer_data.append(np.count_nonzero(X_predict==i)/len(X_predict))


# ### Step 3.3: Compare Customer Data to Demographics Data
# 
# At this point, you have clustered data based on demographics of the general population of Germany, and seen how the customer data for a mail-order sales company maps onto those demographic clusters. In this final substep, you will compare the two cluster distributions to see where the strongest customer base for the company is.
# 
# Consider the proportion of persons in each cluster for the general population, and the proportions for the customers. If we think the company's customer base to be universal, then the cluster assignment proportions should be fairly similar between the two. If there are only particular segments of the population that are interested in the company's products, then we should see a mismatch from one to the other. If there is a higher proportion of persons in a cluster for the customer data compared to the general population (e.g. 5% of persons are assigned to a cluster for the general population, but 15% of the customer data is closest to that cluster's centroid) then that suggests the people in that cluster to be a target audience for the company. On the other hand, the proportion of the data in a cluster being larger in the general population than the customer data (e.g. only 2% of customers closest to a population centroid that captures 6% of the data) suggests that group of persons to be outside of the target demographics.
# 
# Take a look at the following points in this step:
# 
# - Compute the proportion of data points in each cluster for the general population and the customer data. Visualizations will be useful here: both for the individual dataset proportions, but also to visualize the ratios in cluster representation between groups. Seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) or [`barplot()`](https://seaborn.pydata.org/generated/seaborn.barplot.html) function could be handy.
#   - Recall the analysis you performed in step 1.1.3 of the project, where you separated out certain data points from the dataset if they had more than a specified threshold of missing values. If you found that this group was qualitatively different from the main bulk of the data, you should treat this as an additional data cluster in this analysis. Make sure that you account for the number of data points in this subset, for both the general population and customer datasets, when making your computations!
# - Which cluster or clusters are overrepresented in the customer dataset compared to the general population? Select at least one such cluster and infer what kind of people might be represented by that cluster. Use the principal component interpretations from step 2.3 or look at additional components to help you make this inference. Alternatively, you can use the `.inverse_transform()` method of the PCA and StandardScaler objects to transform centroids back to the original data space and interpret the retrieved values directly.
# - Perform a similar investigation for the underrepresented clusters. Which cluster or clusters are underrepresented in the customer dataset compared to the general population, and what kinds of people are typified by these clusters?

# In[59]:


# Compare the proportion of data in each cluster for the customer data to the
# proportion of data in each cluster for the general population.
#plt.plot(general_population)
#plt.plot(customer_data)
# sns.barplot(customer_data)
df = pd.DataFrame([general_population, customer_data])
df.index = ["general_population","customer_data"]
plt.plot(df.iloc[1]/df.iloc[0])
plt.title("Number of cluster vs proportion of each cluster in customer and general data")
plt.xlabel("# of clusters")
plt.ylabel("ratio of customer:general")
print(df)
# hence cluster 9 has most discrepancy


# In[ ]:





# In[60]:


# What kinds of people are part of a cluster that is overrepresented in the
# customer data compared to the general population?
# cluster 4(idx=3) is over represnted in customer data
idx = np.where(X_predict==9)
cluster_9_scaled_pca = X_pca[idx]
cluster_9_scaled = pca.inverse_transform(cluster_9_scaled_pca)
cluster_9 = scaler.inverse_transform(cluster_9_scaled)
df_9 = pd.DataFrame(cluster_9,columns=subset1.columns)
print(df_9)


# In[61]:


# What kinds of people are part of a cluster that is underrepresented in the
# customer data compared to the general population?
# cluster 2(idx=1) is under represented
idx = np.where(X_predict==13)
cluster_13_scaled_pca = X_pca[idx]
cluster_13_scaled = pca.inverse_transform(cluster_13_scaled_pca)
cluster_13= scaler.inverse_transform(cluster_13_scaled)
df_1 =  pd.DataFrame(cluster_13,columns=subset1.columns)
print(df_1)


# ### Discussion 3.3: Compare Customer Data to Demographics Data
# 
# (Double-click this cell and replace this text with your own text, reporting findings and conclusions from the clustering analysis. Can we describe segments of the population that are relatively popular with the mail-order company, or relatively unpopular with the company?)
# 
# cluster 10 is over represented and thia suggests the people in that cluster to be a target audience for the company
# cluster 13 is under represented and this suggests that group of persons to be outside of the target demographics(nothing here).

# > Congratulations on making it this far in the project! Before you finish, make sure to check through the entire notebook from top to bottom to make sure that your analysis follows a logical flow and all of your findings are documented in **Discussion** cells. Once you've checked over all of your work, you should export the notebook as an HTML document to submit for evaluation. You can do this from the menu, navigating to **File -> Download as -> HTML (.html)**. You will submit both that document and this notebook for your project submission.

# In[ ]:





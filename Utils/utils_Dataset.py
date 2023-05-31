import pandas as pd

def process_dataset(Dataframe):
   Dataframe['ASSIGNED'] = pd.to_datetime(Dataframe['ASSIGNED'])
   Filtered_Task_data = Dataframe[Dataframe['ASSIGNED'].dt.year > 2020]
   Filtered_Task_data = Filtered_Task_data.drop(columns =['ASSIGNED', 'READY', 'WORKING', 
                                                            'DELIVERED', 'RECEIVED', 'CLOSE', 
                                                            'PROJECT_ID', 'TASK_ID', 'START', 
                                                            'END', 'MANUFACTURER_INDUSTRY_GROUP', 
                                                            'MANUFACTURER_INDUSTRY', 'MANUFACTURER_SUBINDUSTRY']).reset_index(drop=True)
   # Count the number of occurrences for each translator
   translator_counts = Filtered_Task_data['TRANSLATOR'].value_counts()

   # Get the translators with at least 10 tasks done
   translators_with_at_least_10_tasks = translator_counts[translator_counts >= 500].index

   # Filter the dataset to get rows where the translator has at least 10 tasks done
   Filtered_Task_data = Filtered_Task_data[Filtered_Task_data['TRANSLATOR'].isin(translators_with_at_least_10_tasks)]

   return Filtered_Task_data


def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res     = pd.concat([original_dataframe, dummies], axis=1)
    return(res) 


def OneHotDataframe(Dataframe):
    categorical2oneHot = ['PM', 'TASK_TYPE', 'SOURCE_LANG', 'TARGET_LANG',
                       'MANUFACTURER', 'MANUFACTURER_SECTOR']
    
    One_hot_Dataframe = Dataframe.copy()
    for feature in categorical2oneHot:
        One_hot_Dataframe = encode_and_bind(One_hot_Dataframe, feature)

    One_hot_Dataframe = One_hot_Dataframe.drop(categorical2oneHot, axis = 1).reset_index(drop=True)

    return One_hot_Dataframe
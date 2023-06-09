import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import torch
import sys


sys.path.insert(0, r'C:\Users\34644\Desktop\Second Semester\Synthesis Project\Code_Project\Models')
sys.path.insert(0, r'C:\Users\34644\Desktop\Second Semester\Synthesis Project\Code_Project\Utils')

from model import NeuralNet
from utils_Dataset import process_dataset, OneHotDataframe

import torch.nn as nn

# Obtain the dataset
pkl_file = r"C:\Users\34644\Desktop\Second Semester\Synthesis Project\Language_GrauIAI_UAB.pkl"

Language_Dataset          = pd.read_pickle(pkl_file)               # Read the pkl file containg the pandas dataframe object
Dataset_process           = process_dataset(Language_Dataset)             # Obtain the preprocess Dataset
One_hot_Dataframe         = OneHotDataframe(Dataset_process)       # Changed categorical columns using one hot vectors


# Model weights
num_classes = len(One_hot_Dataframe["TRANSLATOR"].unique()) # Number of translators
input_size  = len(One_hot_Dataframe.columns) - 1 

model = NeuralNet(input_size, num_classes)
weights = torch.load(r'C:\Users\34644\Desktop\Second Semester\Synthesis Project\Code_Project\CheckPoints\256_Batch_Size_30_epocs.pth', map_location="cpu")
model.load_state_dict(weights)
label2translator = {0: 'Casiano',
 1: 'Victorino Yamila',
 2: 'Almudena Fiamma',
 3: 'Acacio Poncio',
 4: 'Ramiro Josafat',
 5: 'Isaias Venancio',
 6: 'Ubaldo Maitane',
 7: 'Nieves Leocadia',
 8: 'Dacio Abel',
 9: 'Carles',
 10: 'Amaro',
 11: 'Sussana',
 12: 'Ireneo',
 13: 'Columbano Cleofas',
 14: 'Lucano',
 15: 'Damiana',
 16: 'Adalberto Anatolio',
 17: 'Alejandro',
 18: 'Elias Daiana',
 19: 'Mariano Valeriano',
 20: 'David Antonio',
 21: 'Aline',
 22: 'Vinicius',
 23: 'Celton',
 24: 'Sara',
 25: 'Edmundo',
 26: 'Celso',
 27: 'Beatriz',
 28: 'Severino',
 29: 'Eliseo',
 30: 'Maximo',
 31: 'Mariano Fidel',
 32: 'Daiana Rosario',
 33: 'Ambrosia Adon',
 34: 'Gregorio Monica',
 35: 'Marcos',
 36: 'Salma Benedicto',
 37: 'Enith',
 38: 'Killa',
 39: 'Gala',
 40: 'Oscar',
 41: 'Luis Felipe',
 42: 'Markel',
 43: 'Ildefonso Ambrosio',
 44: 'Inocencio Lucas',
 45: 'Marcelo German',
 46: 'Dario',
 47: 'Ascension',
 48: 'Guido',
 49: 'Francisco Jose',
 50: 'Abelardo',
 51: 'Pancracio Adolfo',
 52: 'Maria Fernanda',
 53: 'Juan Carlos',
 54: 'Laureano Facundo',
 55: 'Gorka',
 56: 'Roque Marlene',
 57: 'Petronila Teofila',
 58: 'Constantino Carmen',
 59: 'Priscila',
 60: 'Salome',
 61: 'Fortunato',
 62: 'Ester',
 63: 'Christian',
 64: 'Conchi Marciano',
 65: 'Breixo',
 66: 'Pablo Martin',
 67: 'Rafaela',
 68: 'Ariadna Laurina',
 69: 'Hector',
 70: 'Pio Pio',
 71: 'Agueda',
 72: 'Octavi',
 73: 'Artur Fulgencio',
 74: 'Xoana',
 75: 'Anselma Daciano',
 76: 'Porfirio',
 77: 'Maria Aurora',
 78: 'Erico',
 79: 'Margarita',
 80: 'Ana Daniela',
 81: 'Juvenal Vicente',
 82: 'Jonas Tito',
 83: 'Laurina Rafael',
 84: 'Octavio Jana',
 85: 'Paula',
 86: 'Guadalupe Ildefonso',
 87: 'Rufo',
 88: 'Mercedes Catalina',
 89: 'Laurina Santiago',
 90: 'Otilia Rebeca',
 91: 'Jose Ignacio',
 92: 'Greta Casiano',
 93: 'Ivana'}

def data_visual():
    new_window1 = tk.Toplevel(root)
    label = tk.Label(new_window1, text="What data do you want to see?")
    label.pack()

    origen_language = tk.Button(new_window1, text="Origen language", command=origen_lang)
    origen_language.pack()

    responsable = tk.Button(new_window1, text="Responsable group", command=responsable_group)
    responsable.pack()


    person_info = tk.Label(new_window1, text="Search data for a specific person")
    person_info.pack()
    take_person_info = tk.Entry(new_window1, justify="center")
    take_person_info.pack()
    person = tk.Button(new_window1, text="Search", command = lambda: personal_info(new_window1,take_person_info))
    
    person.pack()

def predict_translator():
    new_window2 = tk.Toplevel(root)

    frame = tk.Frame(new_window2)
    frame.pack()

    PM = tk.Label(frame, text="PM:", font=("Arial", 16))
    PM.pack()

    PM_ = ttk.Combobox(frame, font=("Arial", 16), state="readonly", justify="center")
    PM_['values'] = sorted(list(set(Dataset_process["PM"])))
    PM_.pack()

    task_type = tk.Label(frame, text="Task Type:", font=("Arial", 16))
    task_type.pack()
    task_type_ = ttk.Combobox(frame, font=("Arial", 16), state="readonly", justify="center")
    task_type_['values'] = sorted(list(set(Dataset_process["TASK_TYPE"])))
    task_type_.pack()

    Source_lang = tk.Label(frame, text="Source Language:", font=("Arial", 16))
    Source_lang.pack()
    Source_lang_ = ttk.Combobox(frame, font=("Arial", 16), state="readonly", justify="center")
    Source_lang_['values'] = sorted(list(set(Dataset_process["SOURCE_LANG"])))
    Source_lang_.pack()

    Target_lang = tk.Label(frame, text="Target Language:", font=("Arial", 16))
    Target_lang.pack()
    Target_lang_ = ttk.Combobox(frame, font=("Arial", 16), state="readonly", justify="center")
    Target_lang_['values'] = sorted(list(set(Dataset_process["TARGET_LANG"])))
    Target_lang_.pack()

    Forecast = tk.Label(frame, text="Forecast:", font=("Arial", 16))
    Forecast.pack()
    Forecast_ = tk.Entry(frame, justify="center", font=("Arial", 16))
    Forecast_.pack()

    Hourly_Rate = tk.Label(frame, text="Hourly Rate:", font=("Arial", 16))
    Hourly_Rate.pack()
    Hourly_Rate_ = tk.Entry(frame, justify="center", font=("Arial", 16))
    Hourly_Rate_.pack()

    Cost = tk.Label(frame, text="Cost:", font=("Arial", 16))
    Cost.pack()
    Cost_ = tk.Entry(frame, justify="center", font=("Arial", 16))
    Cost_.pack()

    Quality_evaluation = tk.Label(frame, text="Quality Evaluation:", font=("Arial", 16))
    Quality_evaluation.pack()
    Quality_evaluation_ = tk.Entry(frame, justify="center", font=("Arial", 16))
    Quality_evaluation_.pack()

    Manufacturer = tk.Label(frame, text="Manufacturer:", font=("Arial", 16))
    Manufacturer.pack()
    Manufacturer_ = ttk.Combobox(frame, font=("Arial", 16), state="readonly", justify="center")
    Manufacturer_['values'] = sorted(list(set(Dataset_process["MANUFACTURER"])))
    Manufacturer_.pack()

    Manufacturer_sector = tk.Label(frame, text="Manufacturer Sector:", font=("Arial", 16))
    Manufacturer_sector.pack()
    Manufacturer_sector_ = ttk.Combobox(frame, font=("Arial", 16), state="readonly", justify="center")
    Manufacturer_sector_['values'] = sorted(list(set(Dataset_process["MANUFACTURER_SECTOR"])))
    Manufacturer_sector_.pack()

    predict_translator_button = tk.Button(frame, text="Predict translator", command=lambda: get_info(new_window2, PM_, task_type_, Source_lang_, Target_lang_, Forecast_, Hourly_Rate_, Cost_, Quality_evaluation_, Manufacturer_, Manufacturer_sector_))
    predict_translator_button.pack()
    


def origen_lang():
    value_counts = Language_Dataset["SOURCE_LANG"].value_counts()
    num_unique_values = len(value_counts)
    colors = cm.rainbow_r(np.linspace(0, 1, num_unique_values))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    values = []
    for i, (value, count) in enumerate(value_counts.items()):
        bar = ax.bar([value], [np.log(count)], color=colors[i], label=f"{value} ({count})")
        values.append(bar)
        ax.text(bar[0].get_x() + bar[0].get_width() / 2.0, np.log(count), count,
                ha='center', va='bottom', fontsize=10)
    ax.legend(title="ORIGEN_LANG", bbox_to_anchor=(1.4, 1))
    ax.set_xlabel("ORIGEN_LANG")
    ax.set_ylabel("Frequency (log_scale)")
    plt.xticks(rotation=60, ha='right')
        
    new_window = tk.Toplevel(root)
    canvas = FigureCanvasTkAgg(fig, master=new_window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

def responsable_group():
    value_counts = Language_Dataset["PM"].value_counts()
    num_unique_values = len(value_counts)
    colors = cm.rainbow_r(np.linspace(0, 1, num_unique_values))
    
    # Create a histogram with separate bins and labels
    fig, ax = plt.subplots(figsize=(8, 5))
    values = []
    for i, (value, count) in enumerate(value_counts.items()):
        bar = ax.bar([value], [count], color=colors[i], label=f"{value} ({count})")
        ax.text(bar[0].get_x() + bar[0].get_width() / 2.0, np.log(count), count,
                ha='center', va='bottom', fontsize=10)
        values.append(bar)
        
    ax.legend(title="PM")
    ax.set_xlabel("PM")
    ax.set_ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    plt.title("Responsable Distribution")
    
        
    new_window = tk.Toplevel(root)
    canvas = FigureCanvasTkAgg(fig, master=new_window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

def return_top_10(input, num):
    Softmax = nn.Softmax(dim=1)
    output = model(input)
    output_softmax = Softmax(output.unsqueeze(0))
    sorted_values, sorted_indices = torch.sort(output_softmax, descending=True)
    sorted_values, sorted_indices = sorted_values[0][:num], sorted_indices[0][:num]
    mapped_list_comprehension = [(label2translator[int(num)], round(float(values), 2)) for num, values in zip(sorted_indices, sorted_values)]

    return pd.DataFrame(mapped_list_comprehension, columns=['Translator', 'Suitability'])

def obtain_input_vector(X, 
                        PM,
                        TASK_TYPE, 
                        SOURCE_LANG, 
                        TARGET_LANG, 
                        FORECAST, 
                        HOURLY_RATE,
                        COST, 
                        QUALITY_EVALUATION, 
                        MANUFACTURER, 
                        MANUFACTURER_SECTOR):
    
    new_row = pd.Series(np.zeros(len(X.columns)), index=X.columns)

    new_row['FORECAST'], new_row['HOURLY_RATE'], new_row['QUALITY_EVALUATION'], new_row['COST'] = FORECAST, HOURLY_RATE, QUALITY_EVALUATION, COST

    new_row['PM_' + PM]                                 = 1           
    new_row['TASK_TYPE_' + TASK_TYPE]                   = 1
    new_row['SOURCE_LANG_' + SOURCE_LANG]               = 1
    new_row['TARGET_LANG_' + TARGET_LANG]               = 1
    #new_row['MANUFACTURER_'+ MANUFACTURER]              = 1
    new_row['MANUFACTURER_SECTOR_'+MANUFACTURER_SECTOR] = 1

    return torch.tensor(new_row.values, dtype=torch.float32) 

def predict(PM_, task_type_, Source_lang_, Target_lang_, Forecast_, Hourly_Rate_, Cost_, Quality_evaluation_, Manufacturer_, Manufacturer_sector_):
    new_row = obtain_input_vector(One_hot_Dataframe.loc[:, One_hot_Dataframe.columns != 'TRANSLATOR'],
                    PM = str(PM_),
                    TASK_TYPE = str(task_type_), 
                    SOURCE_LANG = str(Source_lang_), 
                    TARGET_LANG = str(Target_lang_), 
                    FORECAST = float(Forecast_),
                    HOURLY_RATE = float(Hourly_Rate_),
                    COST = float(Cost_),
                    QUALITY_EVALUATION = float(Quality_evaluation_), 
                    MANUFACTURER = str(Manufacturer_), 
                    MANUFACTURER_SECTOR = str(Manufacturer_sector_))
    
    print(new_row.shape)
    Ordered_Translators = return_top_10(new_row, 10)
    
    return pd.DataFrame(Ordered_Translators, columns = ["Translator", "Suitability"])

   

def get_info(new_window2,PM_,task_type_,Source_lang_, Target_lang_, Forecast_, Hourly_Rate_, Cost_, Quality_evaluation_, Manufacturer_, Manufacturer_sector_):
    window = tk.Toplevel(new_window2)

    PM_ = PM_.get()
    task_type_ = task_type_.get()
    Source_lang_ = Source_lang_.get()
    Target_lang_ = Target_lang_.get()
    Forecast_ = Forecast_.get()
    Hourly_Rate_ = Hourly_Rate_.get()
    Cost_ = Cost_.get()
    Quality_evaluation_ = Quality_evaluation_.get()
    Manufacturer_ = Manufacturer_.get()
    Manufacturer_sector_ = Manufacturer_sector_.get()
        
    prediction = predict(PM_,task_type_,Source_lang_, Target_lang_, Forecast_, Hourly_Rate_, Cost_, Quality_evaluation_, Manufacturer_, Manufacturer_sector_)
    
    output = tk.Label(window, text = prediction, font = ("Arial",16),justify="left")
    output.pack()

def personal_info(new_window, personal_info):
    window = tk.Toplevel(new_window)

    name = personal_info.get()
    person = tk.Label(window, text=("Data for: "+name),font = ("Arial",16))
    person_tasks = get_translator_tasks(name)
    person.pack()

    task_quality = tk.Button(window, text="Quality of tasks", command= lambda: quality_tasks(person_tasks))
    task_quality.pack()

    task_type = tk.Button(window, text="Type of tasks", command= lambda: type_of_tasks(person_tasks))
    task_type.pack()

    task_cost = tk.Button(window, text="Cost of tasks", command = lambda: cost_of_tasks(person_tasks))
    task_cost.pack()

    source_lang = tk.Button(window, text="Source languages", command = lambda: source_languages(person_tasks))
    source_lang.pack()

    desti_lang = tk.Button(window, text="Destination languages", command = lambda: target_languages(person_tasks))
    desti_lang.pack()

def get_translator_tasks(translator_name):
    
    tasks = Language_Dataset[Language_Dataset['TRANSLATOR'] == translator_name]
    
    return tasks


def quality_tasks(tasks):
    """Generates visualizations for the dataframe of tasks."""

    # Drop NA values from quality evaluation and cost columns
    quality_eval = tasks['QUALITY_EVALUATION'].dropna()
    # Boxplot for quality of tasks
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=quality_eval)
    plt.title('\nQuality of Tasks\n')
    plt.show()

def type_of_tasks(tasks):
        # Pie chart for task types
    plt.figure(figsize=(10, 6))
    task_type_counts = tasks['TASK_TYPE'].value_counts()
    plt.pie(task_type_counts, labels=task_type_counts.index, autopct='%1.1f%%', pctdistance=1.1, labeldistance=1.2)
    plt.title('Types of Tasks')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0, 0.5, 1))
    plt.show()

def cost_of_tasks(tasks):
    cost = tasks['COST'].dropna()
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=cost)
    plt.title('\nCost of Tasks\n')
    plt.show()

def source_languages(tasks):
    plt.figure(figsize=(10, 6))
    target_lang_counts = tasks['SOURCE_LANG'].value_counts()
    plt.pie(target_lang_counts, labels=target_lang_counts.index, autopct='%1.1f%%', pctdistance=1.1, labeldistance=1.2)
    plt.title('\nSource Languages\n')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0, 0.5, 1))
    plt.show()

def target_languages(tasks):
    plt.figure(figsize=(10, 6))
    target_lang_counts = tasks['TARGET_LANG'].value_counts()
    plt.pie(target_lang_counts, labels=target_lang_counts.index, autopct='%1.1f%%', pctdistance=1.1, labeldistance=1.2)
    plt.title('\nTarget Languages\n')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0, 0.5, 1))
    plt.show()

def create_search_function(entry, options_set, button_list):
    def search_options():
        search_text = entry.get().lower()
        for button in button_list:
            button_text = button['text'].lower()
            if search_text in button_text:
                button.pack()
            else:
                button.pack_forget()

    entry.bind('<KeyRelease>', lambda event: search_options())

def update_entry(entry, value, button_list):
    for button in button_list:
        button.pack_forget()
    entry.delete(0, tk.END)
    entry.insert(tk.END, value)

def filter_buttons(event, search_entry, button_list):
    search_text = search_entry.get().lower()
    for button in button_list:
        if search_text and search_text not in button["text"].lower():
            button.pack_forget()
        else:
                button.pack()

root = tk.Tk()

# Create a PhotoImage object with your background image file

button1 = tk.Button(root, text="General info about the data", command=data_visual, font=("Arial", 16))
button1.pack()

# Load and resize the first image
image_file2 = r'C:\Users\34644\Desktop\Second Semester\Synthesis Project\Code_Project\Tkinter\data.jpg'
image2 = Image.open(image_file2)
width = 300
height = 300
resized_image2 = image2.resize((width, height), Image.ANTIALIAS)
photo2 = ImageTk.PhotoImage(resized_image2)

# Create the first image label
image_label2 = tk.Label(root, image=photo2)
image_label2.pack()

# Create the second button
button2 = tk.Button(root, text="Predict translator", command=predict_translator, font=("Arial", 16))
button2.pack()

# Load and resize the second image
image_file3 = r'C:\Users\34644\Desktop\Second Semester\Synthesis Project\Code_Project\Tkinter\translator.jpg'
image3 = Image.open(image_file3)
resized_image3 = image3.resize((width, height), Image.ANTIALIAS)
photo3 = ImageTk.PhotoImage(resized_image3)

# Create the second image label
image_label3 = tk.Label(root, image=photo3)
image_label3.pack()

root.mainloop()






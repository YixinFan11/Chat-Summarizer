
import numpy as np
import re
import datetime
import streamlit as st
from stqdm import stqdm
# Transformers
from transformers import pipeline
from tokenizers import Tokenizer
from summarizer import Summarizer,TransformerSummarizer
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

## Stream lit part -
st.title('Chat Summarizer')
## Show the header


## Collect data from the user just implement for 2 users and not for a group

user_1 = st.text_input(' Enter User name 1  ')
user_1 = str(user_1)
user_2 = st.text_input(' Enter User name 2  ')
user_2 = str(user_2)
# data = st.file_uploader("Add the exported whatsapp chat ( txt format only )!")
# data = data.read()
path  = st.text_input(' Enter the path of the data (Data must be TXT)')
start_date = st.text_input('Enter the start date for summarising chat ', 'DD/MM/YYYY')
end_date = st.text_input('Enter the end date for summarising chat ', 'DD/MM/YYYY')

with open(path, 'r') as file:
    data = file.read()

## Collect the data which needs to be summarized  - Check whether the chats have data for the dates if not then show wrong date


## Preprocessing chats and removing the emojis , links and special characters from the chat
def preprocess_chats(text):#, stem=False, lemmatize=False):
    # Make everything in lower case
    text = text.lower()

    # Refer - https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

    # chars_and_links= r'\d+|http?\S+|[^A-Za-z0-9]+'
    # text = re.sub(chars_and_links,' ',text)
    text = re.sub(emoji_pattern,' ',text)

    # Regular expression to remove emojis
    patterns = [r"\d{2}:\d{2}\s*",    #sun aug 19 13:02:10 2018
        r":\s*([\da-zA_Z0_9]+\/)+([a-zA-Z0-9\.]+)",                #URL
        r"([a-zA-Z_!]+)[\.!_]\d+:\s*",                          #word[._!]number:>=0space
        r":\d+",
        "[':,${}\[\].–|%\"=]"                                        #punctuations
        ]
    for p in patterns:
        text = re.sub(p,'', text)
    text = text.strip()
    # Returning the cleaned text
    return text

## Stream lit Preprocessing the chats
st.write('Preprocessing Chat')

# Bad words list can add many more things
bad_words = ['media omitted', 'this message was deleted','deleted this message','http','xd']
# Saving the user names in a list
user_list = [user_1, user_2]

# Function to remove the bad words and the users name from the chats
def remove(bad_words_list,text):
    ## Calling the preprocess function
    my_string=preprocess_chats(text)

    # Itreating in the list to remove bad words
    for words in bad_words_list:
        my_string=re.sub(".*"+words+".*"+"\n","",my_string)

    # Return the filtered list
    return my_string
     #return re.sub(r"(.*?)"+rem+"(.*?)$|\n", " ", my_string)

# Function call to remove all the useless words from the chats
# chat_dat_clean=remove(bad_words,user_list,data)


## Stream lit Validating whether the date entered data exists or not
st.write('Validating the date for the chats')
# To validate whether the date exists or not
def validate(date_text):
    try:
        if date_text != datetime.datetime.strptime(date_text, "%d/%m/%Y").strftime('%d/%m/%Y'):
            raise ValueError
        return True
    except ValueError:
        return False

##Stream lit the data dictionary is being created
def date_wise_history(bad_words_list,text):
    text=remove(bad_words,text)
    chat_history = {} # Intiallize the dict.
    split_text=text.split("\n")
    for lines in split_text:
        new_line=lines.split(" -")
        if validate(new_line[0])==True:
            if new_line[0] in chat_history.keys():
                chat_history[new_line[0]]+=new_line[1:len(new_line)+1]
            else:
                chat_history[new_line[0]]=new_line[1:len(new_line)+1]
        else:
            ind=split_text.index(lines)
            lines = re.sub('\t', '', lines)
            relative_chat=split_text[ind-1]
            date=relative_chat.split(' -')[0]
            chat_history[date].append(lines)
            split_text[ind]=date+' - '+lines
    return chat_history

## updates selection chat
def date_selection_chat(bad_words,user_list,start_dat, end_dat,text):
    start = datetime.datetime.strptime(start_dat, "%d/%m/%Y")
    end = datetime.datetime.strptime(end_dat, "%d/%m/%Y")
    date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days+1)]
    date_lst=list(map(lambda x: x.strftime("%d/%m/%Y"),date_generated))
    dic=date_wise_history(bad_words,text)
    key_list=list(dic.keys())
    first_date = datetime.datetime.strptime(key_list[0], "%d/%m/%Y")
    last_date = datetime.datetime.strptime(key_list[-1], "%d/%m/%Y")
    if start>end:
        select_chat=False
    elif last_date<start:
        select_chat=False
    elif first_date>end:
        select_chat=False
        #return False
    else:
        select_chat={}
        for key,values in dic.items():
            if key in date_lst:
                chats_of_the_day = ''
                for chat in values :
                    chats_of_the_day += chat;
                select_chat[key]=chats_of_the_day
    select_chat_1={}
    select_chat_2={}
    for key,value in select_chat.items():
        value_1=value
        value_2=value
        for user in user_list:
            value_1=re.sub(user," ", value_1)
            value_2=re.sub(user,user+":", value_2)
        select_chat_1[key]=value_1
        select_chat_2[key]=value_2
    return select_chat_1,select_chat_2

st.write('Formatting Chat History')

## Making summary using bert
## Creating a new dictonary to store the summarized chats
def distilbart_summary(bad_words,user_list,start_dat, end_dat,text):
    select_chat_dic,select_chat_dic_user=date_selection_chat(bad_words,user_list,start_dat, end_dat,text)
    if select_chat_dic_user == False:
        print('Current Date does not match your chat history, please enter a valid start date')
    elif bool(select_chat_dic_user)==False:
        print("No summary available")
    else:
        chat_sum_dict_distilbart = {}
        for key,value in select_chat_dic_user.items():
            summarization = pipeline("summarization",truncation=True, min_length = 5, max_length = 80,model="philschmid/distilbart-cnn-12-6-samsum")
            summary_text = summarization(value)[0]['summary_text']
            chat_sum_dict_distilbart[key] = str(summary_text)
        return chat_sum_dict_distilbart

st.write('Creating Summary...... Usually take long time when running it the first time')

answer = distilbart_summary(bad_words,user_list,start_date,end_date,data)

for key,vals in answer.items(): st.write(key) ; st.write('\nSummary - ') ; st.write(vals)

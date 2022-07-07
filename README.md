# Chat Summarizer

## About Data
The data used in this agent is the ***ChatHistory.txt*** downloaded from WhatsApp application. Due to the privacy reason, this txt will be uploaded through Moodles.

## About Agent

#### Step I: Install Packages
Install the requirement.txt
If it is not working, please install the following packages manually:

```
pip install transformers
pip install bert-extractive-summarizer 
pip install pytorch-transformers
pip install transformers torch sentencepiece

pip install streamlit
pip install stqdm
```

#### Step II: Run the agent
Use the command '***streamlit run app.py***' to run the agent in the Terminal. Then it will pop out a website which need to enter the five inputs(Two User names, text document of chat histoyr and Start and End date) as follow to create the final summary:
![Image](https://github.com/YixinFan11/DIA-Submission/blob/master/graph/Chat_Summary_1.png?raw=true)

#### Step III: View the summary
After waiting for some minutes, the summary will be shown as below:
![Image](https://github.com/YixinFan11/DIA-Submission/blob/master/graph/Chat_Summary_2.png?raw=true)


## About the Experiment
The code implementation of whole experiment process is on the Jupyter Notebook running on google colab. Run the ***WhatsApp_Chat_Summary_Final(DIA).ipynb*** to see experiment result.


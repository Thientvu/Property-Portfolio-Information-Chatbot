# partner-esi-llm-chatbot

## Setting up environment
To implement and test the code, use anaconda and install all required packages.

First, please create a new environment that use python 3.9.18 as follows:
```
conda create --name partneresi python=3.9.18
```

Once the environment has been created, run:
```
conda activate partneresi
```

Navigate to the repo folder on your local machine, and make sure to create a .env file contain:
```
OPENAI_API_KEY = "<The ChatGPT API Key>"
```

First, check if the environment has everything we need, we will use jupyter lab for quick test, run:
```
conda install pip jupyterlab
```

In the end, run the following command:

```
pip install -r requirements.txt
```

Once everything has been installed, run:

```
jupyter notebook
```

See if all the code block can run or not.

## Run demo
To run demo, make sure you are in the environment and the repo directory, and run:
```
streamlit run app.py 
```
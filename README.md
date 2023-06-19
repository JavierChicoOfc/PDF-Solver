# PDF-Solver

PDF-Solver is a web application that allows users to ask questions about the content of a PDF file using natural language. It is built upon GPT-3.5.

# Usage

1. Clone the repository
2. Create a virtual environment with `python3 -m venv venv`, conda or any other tool. It is needed for Streamlit to work properly.
3. Install the dependencies `pip install -r requirements.txt` in the virtual environment
4. {Optional} Go to config.py and write down your own key for OpenAI API (you can get it [here](https://beta.openai.com/))
5. Run the app `streamlit run app.py`
6. Follow the instructions on the web page

# Architecture design

![Architecture](final-scheme.png)

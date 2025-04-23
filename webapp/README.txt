How to run this Streamlit web app locally

1. Download repository
   Navigate to the webapp folder (path: BugBot/webapp) and download the contents

2. Folder Structure  
   Make sure your local setup looks like this:
   
   webapp/
   ├── app.py
   ├── model.h5
   ├── requirements.txt
   └── test images/ (optional for testing input images)

3. Set Up a Python Environment (only if you do not already have an environment set up with the requirements.txt):  

   python -m venv venv
   source venv/bin/activate

4. Install dependencies  

   pip install -r requirements.txt

5. Run Streamlit app
   First run app.py
   After running app.py, in the terminal, navigate to the `webapp` directory and 
   start the Streamlit app by typing and entering:

   streamlit run app.py

6. Stop the Streamlit app
   To stop the Streamlit server, press Ctrl + C in the terminal
